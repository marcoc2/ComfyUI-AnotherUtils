"""
SAM2 Segmentation Service for AnotherUtils.
Logic copied from comfyui-segment-anything-2/load_model.py (which works).
Monkey-patches SAM2Transforms to add JIT fallback (which the pip version lacks).
Supports both single-image and video segmentation modes.
"""
import numpy as np
import torch
import torch.nn as nn
import os
import yaml

from .utils import ModelDownloader, SAM2_URLS

# --- CRITICAL PATCH ---
# The pip version of sam2.utils.transforms.SAM2Transforms uses torch.jit.script
# without try/except, which crashes in embedded Python environments.
# The comfyui-segment-anything-2 node's bundled version has a try/except fallback.
# We apply the same fix here as a monkey-patch before importing SAM2ImagePredictor.
import sam2.utils.transforms as _sam2_transforms
from torchvision.transforms import Normalize, Resize, ToTensor

_original_sam2transforms_init = _sam2_transforms.SAM2Transforms.__init__

def _patched_sam2transforms_init(self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0):
    nn.Module.__init__(self)
    self.resolution = resolution
    self.mask_threshold = mask_threshold
    self.max_hole_area = max_hole_area
    self.max_sprinkle_area = max_sprinkle_area
    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]
    self.to_tensor = ToTensor()
    try:
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )
    except Exception as e:
        print(f"[AnotherUtils] torch.jit.script failed ({e}), using non-JIT transforms")
        self.transforms = nn.Sequential(
            Resize((self.resolution, self.resolution)),
            Normalize(self.mean, self.std),
        )

_sam2_transforms.SAM2Transforms.__init__ = _patched_sam2transforms_init
# --- END PATCH ---

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# --- PATCH 2: SAM2VideoPredictor.init_state ---
# The pip version of SAM 2.1 expects (video_path), but we need to pass image tensors directly.
# We create a patched init_state that takes the tensors but faithfully reproduces
# the exact inference_state dictionary structure expected by SAM 2.1 pip.
from collections import OrderedDict

@torch.inference_mode()
def _patched_init_state(self, images, video_height, video_width, device='cuda',
                        offload_video_to_cpu=False, offload_state_to_cpu=False,
                        async_loading_frames=False):
    """Initialize inference state from image tensors (patched for SAM 2.1 pip package)."""
    compute_device = self.device  # use model's device
    
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
        
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    
    self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state

SAM2VideoPredictor.init_state = _patched_init_state
# --- END PATCH 2 ---


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _resolve_yaml_path(model_name):
    """Find the correct YAML config for a given model name."""
    if "tiny" in model_name or "_t." in model_name:
        suffix = "t"
    elif "small" in model_name or "_s." in model_name:
        suffix = "s"
    elif "large" in model_name or "_l." in model_name:
        suffix = "l"
    else:
        suffix = "b+"

    version = "sam2.1" if "2.1" in model_name else "sam2"
    cfg_filename = f"{version}_hiera_{suffix}.yaml"

    comfy_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    primary_path = os.path.join(comfy_base, "custom_nodes", "comfyui-segment-anything-2", "sam2_configs", cfg_filename)
    if os.path.exists(primary_path):
        return primary_path

    import sam2 as _sam2_pkg
    fallback_path = os.path.join(os.path.dirname(_sam2_pkg.__file__), "configs", version, cfg_filename)
    if os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(f"SAM2 config not found for '{model_name}':\n  {primary_path}\n  {fallback_path}")


def _load_weights(checkpoint_path):
    """Load model weights from checkpoint (safetensors or pt)."""
    try:
        from comfy.utils import load_torch_file
        return load_torch_file(checkpoint_path)
    except ImportError:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(checkpoint_path, device="cpu")
        else:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            return sd.get("model", sd)


def _ensure_checkpoint(checkpoint_path):
    """Auto-download checkpoint if missing."""
    if not os.path.exists(checkpoint_path):
        filename = os.path.basename(checkpoint_path)
        if filename in SAM2_URLS:
            success = ModelDownloader.download(SAM2_URLS[filename], checkpoint_path)
            if not success:
                raise FileNotFoundError(f"Failed to download: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")


def _build_model_from_yaml(config, device, dtype, model_class=SAM2Base, is_video=False):
    """
    Build SAM2 model from parsed YAML config.
    Exact copy of comfyui-segment-anything-2/load_model.py logic.
    model_class: SAM2Base (image) or SAM2VideoPredictor (video).
    """
    mc = config['model']

    # Image Encoder
    tc = mc['image_encoder']['trunk']
    nc = mc['image_encoder']['neck']
    pec = nc['position_encoding']

    pos_enc = PositionEmbeddingSine(
        num_pos_feats=pec['num_pos_feats'], normalize=pec['normalize'],
        scale=pec['scale'], temperature=pec['temperature']
    )
    neck = FpnNeck(
        position_encoding=pos_enc, d_model=nc['d_model'],
        backbone_channel_list=nc['backbone_channel_list'],
        fpn_top_down_levels=nc['fpn_top_down_levels'],
        fpn_interp_model=nc['fpn_interp_model']
    )
    trunk_keys = ['embed_dim', 'num_heads', 'global_att_blocks', 'window_pos_embed_bkg_spatial_size', 'stages']
    trunk = Hiera(**{k: tc[k] for k in trunk_keys if k in tc})
    image_encoder = ImageEncoder(scalp=mc['image_encoder']['scalp'], trunk=trunk, neck=neck)

    # Memory Attention
    malc = mc['memory_attention']['layer']
    sa = RoPEAttention(**{k: malc['self_attention'][k] for k in
         ['rope_theta', 'feat_sizes', 'embedding_dim', 'num_heads', 'downsample_rate', 'dropout']})
    ca_cfg = malc['cross_attention']
    ca = RoPEAttention(**{k: ca_cfg[k] for k in
         ['rope_theta', 'feat_sizes', 'rope_k_repeat', 'embedding_dim', 'num_heads', 'downsample_rate', 'dropout', 'kv_in_dim']})
    mal = MemoryAttentionLayer(
        activation=malc['activation'], dim_feedforward=malc['dim_feedforward'],
        dropout=malc['dropout'], pos_enc_at_attn=malc['pos_enc_at_attn'],
        self_attention=sa, d_model=malc['d_model'],
        pos_enc_at_cross_attn_keys=malc['pos_enc_at_cross_attn_keys'],
        pos_enc_at_cross_attn_queries=malc['pos_enc_at_cross_attn_queries'],
        cross_attention=ca
    )
    mem_attn = MemoryAttention(
        d_model=mc['memory_attention']['d_model'],
        pos_enc_at_input=mc['memory_attention']['pos_enc_at_input'],
        layer=mal, num_layers=mc['memory_attention']['num_layers']
    )

    # Memory Encoder
    mec = mc['memory_encoder']
    pe_mem = PositionEmbeddingSine(
        num_pos_feats=mec['position_encoding']['num_pos_feats'],
        normalize=mec['position_encoding']['normalize'],
        scale=mec['position_encoding']['scale'],
        temperature=mec['position_encoding']['temperature']
    )
    mask_ds = MaskDownSampler(
        kernel_size=mec['mask_downsampler']['kernel_size'],
        stride=mec['mask_downsampler']['stride'],
        padding=mec['mask_downsampler']['padding']
    )
    fl = mec['fuser']['layer']
    fuser = Fuser(
        num_layers=mec['fuser']['num_layers'],
        layer=CXBlock(dim=fl['dim'], kernel_size=fl['kernel_size'],
                      padding=fl['padding'], layer_scale_init_value=float(fl['layer_scale_init_value']))
    )
    mem_enc = MemoryEncoder(position_encoding=pe_mem, mask_downsampler=mask_ds, fuser=fuser, out_dim=mec['out_dim'])

    # Assemble
    model = model_class(
        image_encoder=image_encoder, memory_attention=mem_attn, memory_encoder=mem_enc,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        num_maskmem=mc['num_maskmem'], image_size=mc['image_size'],
        sigmoid_scale_for_mem_enc=mc['sigmoid_scale_for_mem_enc'],
        sigmoid_bias_for_mem_enc=mc['sigmoid_bias_for_mem_enc'],
        use_mask_input_as_output_without_sam=mc['use_mask_input_as_output_without_sam'],
        directly_add_no_mem_embed=mc['directly_add_no_mem_embed'],
        use_high_res_features_in_sam=mc['use_high_res_features_in_sam'],
        multimask_output_in_sam=mc['multimask_output_in_sam'],
        iou_prediction_use_sigmoid=mc['iou_prediction_use_sigmoid'],
        use_obj_ptrs_in_encoder=mc['use_obj_ptrs_in_encoder'],
        add_tpos_enc_to_obj_ptrs=mc['add_tpos_enc_to_obj_ptrs'],
        only_obj_ptrs_in_the_past_for_eval=mc['only_obj_ptrs_in_the_past_for_eval'],
        pred_obj_scores=mc['pred_obj_scores'], pred_obj_scores_mlp=mc['pred_obj_scores_mlp'],
        fixed_no_obj_ptr=mc['fixed_no_obj_ptr'],
        multimask_output_for_tracking=mc['multimask_output_for_tracking'],
        use_multimask_token_for_obj_ptr=mc['use_multimask_token_for_obj_ptr'],
        compile_image_encoder=mc['compile_image_encoder'],
        multimask_min_pt_num=mc['multimask_min_pt_num'],
        multimask_max_pt_num=mc['multimask_max_pt_num'],
        use_mlp_for_obj_ptr_proj=mc['use_mlp_for_obj_ptr_proj'],
        proj_tpos_enc_in_obj_ptrs=mc['proj_tpos_enc_in_obj_ptrs'],
        no_obj_embed_spatial=mc['no_obj_embed_spatial'],
        use_signed_tpos_enc_to_obj_ptrs=mc['use_signed_tpos_enc_to_obj_ptrs'],
        binarize_mask_from_pts_for_mem_enc=is_video,
    ).to(dtype).to(device).eval()

    return model


# ─────────────────────────────────────────────────────────────
# Single-image service
# ─────────────────────────────────────────────────────────────

class SAM2Service:
    """Single-image SAM2 segmentation."""

    def __init__(self, model_cfg_path, checkpoint_path, device="auto"):
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        _ensure_checkpoint(checkpoint_path)

        yaml_path = _resolve_yaml_path(checkpoint_path)
        print(f"[AnotherUtils] SAM2 (image) config: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        model = _build_model_from_yaml(config, self.device, torch.float32, SAM2Base, False)
        model.load_state_dict(_load_weights(checkpoint_path))
        self.predictor = SAM2ImagePredictor(model)
        print("[AnotherUtils] SAM2 (image) loaded.")

    def set_image(self, image_np):
        self.predictor.set_image(image_np)

    def predict(self, points=None, labels=None, bboxes=None):
        if points is not None: points = np.array(points)
        if labels is not None: labels = np.array(labels)
        if bboxes is not None: bboxes = np.array(bboxes)
        masks, scores, logits = self.predictor.predict(
            point_coords=points, point_labels=labels, box=bboxes, multimask_output=False)
        if masks.ndim == 4: masks = masks.squeeze(1)
        return masks


# ─────────────────────────────────────────────────────────────
# Video service
# ─────────────────────────────────────────────────────────────

class SAM2VideoService:
    """
    Video SAM2 segmentation.
    Copied from comfyui-segment-anything-2 Sam2VideoSegmentationAddPoints + Sam2VideoSegmentation.
    """

    def __init__(self, model_cfg_path, checkpoint_path, device="auto"):
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.dtype = torch.float32
        _ensure_checkpoint(checkpoint_path)

        yaml_path = _resolve_yaml_path(checkpoint_path)
        print(f"[AnotherUtils] SAM2 (video) config: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model = _build_model_from_yaml(config, self.device, self.dtype, SAM2VideoPredictor, True)
        self.model.load_state_dict(_load_weights(checkpoint_path))
        self.inference_state = None
        print("[AnotherUtils] SAM2 (video) loaded.")

    def init_state(self, images_bchw, H, W):
        """
        Initialize video state from frames.
        images_bchw: [B, C, H, W] float tensor (resized to model input size)
        """
        self.model.to(self.device)
        if self.inference_state is not None:
            self.model.reset_state(self.inference_state)
        self.inference_state = self.model.init_state(images_bchw, H, W, device=self.device)
        return self.inference_state

    def add_points(self, frame_idx, obj_id, coords_pos, coords_neg=None):
        """
        Add point prompts on a specific frame for a specific object.
        coords_pos: list of (x, y)
        coords_neg: optional list of (x, y)
        """
        pos = np.atleast_2d(np.array(coords_pos))
        pos_labels = np.ones(len(pos))

        if coords_neg and len(coords_neg) > 0:
            neg = np.atleast_2d(np.array(coords_neg))
            combined = np.concatenate([pos, neg], axis=0)
            labels = np.concatenate([pos_labels, np.zeros(len(neg))], axis=0)
        else:
            combined = pos
            labels = pos_labels

        self.model.to(self.device)
        with torch.autocast(self.device, dtype=self.dtype):
            _, obj_ids, mask_logits = self.model.add_new_points(
                inference_state=self.inference_state,
                frame_idx=frame_idx, obj_id=obj_id,
                points=combined, labels=labels,
            )
        return obj_ids, mask_logits

    def propagate(self):
        """
        Propagate segmentation across all frames.
        Returns: [B, H, W] float mask tensor
        """
        self.model.to(self.device)
        segments = {}

        with torch.autocast(self.device, dtype=self.dtype):
            for frame_idx, obj_ids, mask_logits in self.model.propagate_in_video(self.inference_state):
                _, _, H, W = mask_logits.shape
                combined = np.zeros((H, W), dtype=bool)
                for i in range(len(obj_ids)):
                    combined = np.logical_or(combined, (mask_logits[i] > 0.0).cpu().numpy())
                segments[frame_idx] = combined

        masks = []
        for idx in sorted(segments.keys()):
            m = segments[idx]
            if m.ndim == 3: m = m[0]
            masks.append(torch.from_numpy(m.astype(np.float32)))

        return torch.stack(masks, dim=0)
