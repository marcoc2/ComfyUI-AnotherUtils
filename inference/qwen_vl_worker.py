"""
Qwen2.5-VL inference worker.

Runs as an isolated subprocess. The parent (QwenVLNode) sends JSON commands
on stdin and reads JSON responses on stdout. Logs and HuggingFace progress
bars flow to stderr, which is inherited from the parent.

Protocol (one JSON per line on stdin/stdout):

    parent -> worker:
        {"cmd":"init", "model_name":"...", "hf_cache":"..."}
        {"cmd":"infer", "image_b64":"...", "system_prompt":"...",
         "user_prompt":"...", "max_new_tokens":300}
        {"cmd":"exit"}

    worker -> parent:
        {"status":"ready"}
        {"response":"..."}
        {"error":"..."}

Killing the process frees all GPU/RAM allocations via the OS, which is the
whole point of running inference here instead of in-process.
"""

import sys
import os
import io
import json
import base64
import traceback


def _log(msg):
    print(f"[qwen-worker] {msg}", file=sys.stderr, flush=True)


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _do_init(req):
    model_name = req["model_name"]
    hf_cache = req["hf_cache"]

    os.environ["HF_HOME"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = (
        hf_cache if hf_cache.endswith("hub")
        else os.path.join(hf_cache, "hub")
    )

    _log(f"Loading {model_name}")

    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        cache_dir=hf_cache,
    ).to(target_device)
    model.eval()

    _log(f"Ready on {target_device}")
    return model, processor


def _do_infer(model, processor, req):
    import torch
    from PIL import Image

    img_bytes = base64.b64decode(req["image_b64"])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    messages = [
        {"role": "system", "content": req["system_prompt"]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": req["user_prompt"]},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.get("max_new_tokens", 300),
            do_sample=False,
            num_beams=1,
        )

    generated = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    del inputs, output_ids, generated
    return response


def main():
    model = None
    processor = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as e:
            _send({"error": f"invalid JSON: {e}"})
            continue

        cmd = req.get("cmd")

        if cmd == "init":
            try:
                model, processor = _do_init(req)
                _send({"status": "ready"})
            except Exception as e:
                _log(traceback.format_exc())
                _send({"error": f"init failed: {e}"})

        elif cmd == "infer":
            if model is None or processor is None:
                _send({"error": "worker not initialized"})
                continue
            try:
                response = _do_infer(model, processor, req)
                _send({"response": response})
            except Exception as e:
                _log(traceback.format_exc())
                _send({"error": f"infer failed: {e}"})

        elif cmd == "exit":
            _log("exit requested")
            return

        else:
            _send({"error": f"unknown cmd: {cmd}"})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log(f"fatal: {e}")
        _log(traceback.format_exc())
        sys.exit(1)
