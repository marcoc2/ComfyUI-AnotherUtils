"""Batch YuE2 covers: one source song, every LoRA in models/loras/YuE2.

- YuE2 Cover: Song    keeps each song's lyrics in input/yue2_lyrics/<song>.txt
                      and outputs the track length for max_duration.
- YuE2 Cover: LoRAs   outputs a LIST (model, clip, style, lyrics, filename) with one item per LoRA.
                      ComfyUI runs the rest of the graph once per list item.

Each LoRA's style and strengths live in models/loras/YuE2/yue2_cover_profiles.json, keyed by file
name, so a LoRA keeps its profile when it moves between subfolders. Running with a single LoRA
selected saves its profile. A LoRA without a profile gets an automatic one, built from the captions
of the AI Toolkit dataset it was trained on (found through the job name stored in the LoRA).
"""
import json
import logging
import math
import os
import re
import struct

import comfy.sd
import comfy.utils
import folder_paths
import yaml
from aiohttp import web
from server import PromptServer

from ..inference.deepseek_text import DEFAULT_INSTRUCTION, MODELS, cached_ask, has_section_tags

LORA_FOLDER = "YuE2"
ALL = "(all)"
FOLDER_PREFIX = "(folder) "
INSTRUMENTAL = "[Instrumental]"
AITK_OUTPUT_DIR = os.environ.get("AITK_OUTPUT_DIR", r"F:\workspace\ai-toolkit\output")


def _profiles_dir():
    return os.path.join(folder_paths.get_folder_paths("loras")[0], LORA_FOLDER)


def _profiles_file():
    return os.path.join(_profiles_dir(), "yue2_cover_profiles.json")


def _lyrics_dir():
    return os.path.join(folder_paths.get_input_directory(), "yue2_lyrics")


def _loras():
    """Every LoRA under the YuE2 folder, keyed by path inside it without extension:
    {"gaga": "YuE2\\gaga.safetensors", "new/grimes": "YuE2\\new\\grimes.safetensors"}."""
    out = {}
    for rel in folder_paths.get_filename_list("loras"):
        parts = re.split(r"[\\/]", rel)
        if len(parts) >= 2 and parts[0] == LORA_FOLDER:
            out["/".join(parts[1:-1] + [os.path.splitext(parts[-1])[0]])] = rel
    return dict(sorted(out.items()))


def _subfolders(loras):
    return sorted({key.rsplit("/", 1)[0] for key in loras if "/" in key})


def _select(choice, loras):
    """LoRA keys picked by the lora widget: (all) = top level, (folder) x = everything directly in x."""
    if choice == ALL:
        return [k for k in loras if "/" not in k]
    if choice.startswith(FOLDER_PREFIX):
        folder = choice[len(FOLDER_PREFIX):]
        return [k for k in loras if k.rsplit("/", 1)[0] == folder and "/" in k]
    return [choice]


def _profile_key(lora_key):
    return lora_key.rsplit("/", 1)[-1]


def _job_name(rel):
    """AI Toolkit job name stored in the LoRA header; the file itself may have been renamed."""
    path = folder_paths.get_full_path("loras", rel)
    try:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(size)).get("__metadata__", {})
    except (OSError, ValueError, struct.error):
        meta = {}
    return meta.get("name") or os.path.splitext(os.path.basename(path))[0]


def _read_captions(folder, ext):
    caps = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
        for name in files:
            if name.endswith("." + ext):
                with open(os.path.join(root, name), encoding="utf-8", errors="replace") as f:
                    text = f.read()
                cap = re.search(r"<CAPTION>(.*?)</CAPTION>", text, re.S)
                lyr = re.search(r"<LYRICS>(.*?)</LYRICS>", text, re.S)
                caps.append(((cap.group(1) if cap else text).strip(), (lyr.group(1) if lyr else "").strip()))
    return [c for c in caps if c[0]]


def _representative(captions):
    """The caption sharing the most vocabulary with the rest of the dataset."""
    words = [set(re.findall(r"[a-z]+", c.lower())) for c in captions]

    def score(i):
        return sum(len(words[i] & w) / max(1, len(words[i] | w)) for j, w in enumerate(words) if j != i)

    return captions[max(range(len(captions)), key=score)]


def _auto_profile(rel):
    """Profile from the AI Toolkit job that trained this LoRA, or None when the job is not found."""
    job = _job_name(rel)
    try:
        with open(os.path.join(AITK_OUTPUT_DIR, job, "config.yaml"), encoding="utf-8") as f:
            process = yaml.safe_load(f)["config"]["process"][0]
        dataset = process["datasets"][0]
    except (OSError, KeyError, IndexError, TypeError, yaml.YAMLError):
        return None
    pairs = _read_captions(dataset["folder_path"], dataset.get("caption_ext") or "txt")
    if not pairs:
        return None
    caption = _representative([c for c, _ in pairs])
    trigger = (process.get("trigger_word") or job).strip()
    return {"style": f"{trigger} {caption}", "strength_model": 0.95, "strength_clip": 0.65,
            "instrumental": all(l.lower() == INSTRUMENTAL.lower() for _, l in pairs), "auto": True}


def _read_profiles():
    try:
        with open(_profiles_file(), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _write_profiles(profiles):
    tmp = _profiles_file() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(profiles.items())), f, ensure_ascii=False, indent=1)
    os.replace(tmp, _profiles_file())


def _lyrics_file(song):
    return os.path.join(_lyrics_dir(), os.path.splitext(os.path.basename(song))[0] + ".txt")


def _read_lyrics(song):
    try:
        with open(_lyrics_file(song), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _clean_name(s):
    return re.sub(r'[<>:"/\\|?*]+', "_", s).strip(" .") or "unnamed"


def _load_audio(path):
    """Like comfy_extras.nodes_audio.load, but skips corrupt packets instead of failing on them.

    MP3s from the web often carry a broken frame ("Header missing"); PyAV raises on it and the
    core loader gives up on the whole file, while ffmpeg itself just drops that frame.
    """
    import av
    import torch
    from comfy_extras.nodes_audio import f32_pcm

    frames, skipped = [], 0
    # Tags in Latin-1 (common in MP3s from the web) must not stop the audio from loading.
    with av.open(path, metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {path}.")
        stream = container.streams.audio[0]
        sr, channels = stream.codec_context.sample_rate, stream.channels
        for packet in container.demux(stream):
            try:
                decoded = packet.decode()
            except av.error.InvalidDataError:
                skipped += 1
                continue
            for frame in decoded:
                buf = torch.from_numpy(frame.to_ndarray())
                if buf.shape[0] != channels:
                    buf = buf.view(-1, channels).t()
                frames.append(buf)
    if not frames:
        raise ValueError(f"No audio could be decoded from {path}.")
    if skipped:
        logging.warning(f"[YuE2 Cover] {os.path.basename(path)}: skipped {skipped} corrupt audio packet(s).")
    return f32_pcm(torch.cat(frames, dim=1)), sr


def _songs():
    folder = folder_paths.get_input_directory()
    return sorted(folder_paths.filter_files_content_types(os.listdir(folder), ["audio", "video"]))


class YuE2CoverSong:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (_songs(), {"audio_upload": True}),
                "use_deepseek_api": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Calls the DeepSeek API: lyrics without section tags ([Verse], [Chorus]...) are sent "
                               "there to add them. Lyrics that already have tags are used as they are."}),
                "lyrics": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Filled in with this song's saved lyrics. Paste raw lyrics here: "
                                                 "without section tags they are structured by DeepSeek on run. "
                                                 "Whatever runs is saved to input/yue2_lyrics/<song>.txt."}),
                "margin_s": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 300.0, "step": 1.0,
                                       "tooltip": "Seconds added to the track length for max_duration, so a cover that "
                                                  "comes out longer than the source is not cut off."}),
                "deepseek_model": (MODELS, {"default": MODELS[0]}),
                "deepseek_instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION,
                                                    "tooltip": "Sent to DeepSeek before the lyrics."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("audio", "lyrics", "song", "max_duration")
    FUNCTION = "run"
    CATEGORY = "audio/yue2 cover"

    @classmethod
    def IS_CHANGED(cls, audio, lyrics, margin_s, **kwargs):
        path = folder_paths.get_annotated_filepath(audio)
        return f"{os.path.getmtime(path)}|{os.path.getsize(path)}|{lyrics}|{margin_s}|{sorted(kwargs.items())}"

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Audio file not found: {audio}"
        return True

    def run(self, audio, lyrics, margin_s, use_deepseek_api=True, deepseek_model=MODELS[0],
            deepseek_instruction=DEFAULT_INSTRUCTION):
        path = folder_paths.get_annotated_filepath(audio)
        if lyrics.strip():
            # the saved file is the cache: once structured, the lyrics carry tags and skip DeepSeek
            if use_deepseek_api and not has_section_tags(lyrics):
                lyrics = cached_ask(deepseek_model, deepseek_instruction, lyrics.strip(), 0.3)
            os.makedirs(_lyrics_dir(), exist_ok=True)
            with open(_lyrics_file(audio), "w", encoding="utf-8") as f:
                f.write(lyrics.strip() + "\n")
        else:
            lyrics = _read_lyrics(audio) or ""
        if not lyrics.strip():
            raise ValueError(f"'{audio}' has no saved lyrics. Paste the structured lyrics ([Verse], [Chorus]...) "
                             f"or write {INSTRUMENTAL}.")

        waveform, sr = _load_audio(path)
        length = waveform.shape[-1] / sr
        song = os.path.splitext(os.path.basename(audio))[0]
        result = ({"waveform": waveform.unsqueeze(0), "sample_rate": sr}, lyrics.strip(), song,
                  float(math.ceil(length + margin_s)))
        # the frontend swaps the lyrics box for what actually ran, so the next run starts from it
        return {"ui": {"lyrics": [lyrics.strip()]}, "result": result}


_lora_cache = {}


def _load_lora(rel):
    path = folder_paths.get_full_path_or_raise("loras", rel)
    key = (path, os.path.getmtime(path))
    if key not in _lora_cache:
        for k in [k for k in _lora_cache if k[0] == path]:
            del _lora_cache[k]
        _lora_cache[key] = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    return _lora_cache[key]


class YuE2CoverLoras:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lyrics": ("STRING", {"forceInput": True}),
                "song": ("STRING", {"forceInput": True}),
                "lora": (cls._choices(), {
                    "tooltip": f"{ALL}: every LoRA directly in models/loras/YuE2. {FOLDER_PREFIX}<name>: every LoRA "
                               "in that subfolder (e.g. put new ones in 'new'). A single LoRA: runs only that one and "
                               "saves the style and strengths below as its profile."}),
                "style": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "Trigger + style prompt for the selected LoRA. Filled in with the saved "
                                                "(or automatic) profile. Ignored for groups."}),
                "strength_model": ("FLOAT", {"default": 0.95, "min": -2.0, "max": 2.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 0.65, "min": -2.0, "max": 2.0, "step": 0.01}),
                "instrumental": ("BOOLEAN", {"default": False,
                                             "tooltip": f"Instrumental LoRA: the lyrics become {INSTRUMENTAL}."}),
                "skip": ("STRING", {"default": "",
                                    "tooltip": "Groups only: comma-separated LoRA names to leave out."}),
            }
        }

    @staticmethod
    def _choices():
        loras = _loras()
        return [ALL] + [FOLDER_PREFIX + f for f in _subfolders(loras)] + list(loras)

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "style", "lyrics", "filename", "summary")
    OUTPUT_IS_LIST = (True, True, True, True, True, False)
    FUNCTION = "run"
    CATEGORY = "audio/yue2 cover"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # hand-edited profiles or a retrained LoRA also need a new run
        try:
            profiles = os.path.getmtime(_profiles_file())
        except OSError:
            profiles = 0
        loras = [(n, os.path.getmtime(folder_paths.get_full_path("loras", r))) for n, r in _loras().items()]
        return f"{profiles}|{loras}"

    def run(self, model, clip, lyrics, song, lora, style, strength_model, strength_clip, instrumental, skip):
        loras = _loras()
        profiles = _read_profiles()
        group = lora == ALL or lora.startswith(FOLDER_PREFIX)
        if not group and lora not in loras:
            raise ValueError(f"LoRA '{lora}' not found in models/loras/{LORA_FOLDER}.")

        if group:
            skipped = {p.strip() for p in skip.split(",") if p.strip()}
            keys = [k for k in _select(lora, loras) if _profile_key(k) not in skipped]
        else:
            keys = [lora]
            if style.strip():
                profiles[_profile_key(lora)] = {"style": style.strip(), "strength_model": round(strength_model, 3),
                                                "strength_clip": round(strength_clip, 3),
                                                "instrumental": bool(instrumental)}
                _write_profiles(profiles)

        # LoRAs without a profile get one from their training dataset, saved so it can be edited later
        created, no_profile = [], []
        for key in keys:
            name = _profile_key(key)
            if name not in profiles:
                auto = _auto_profile(loras[key])
                if auto:
                    profiles[name] = auto
                    created.append(name)
                else:
                    no_profile.append(name)
        if created:
            _write_profiles(profiles)
        keys = [k for k in keys if _profile_key(k) in profiles]
        if not keys:
            raise ValueError(f"No profile for {', '.join(no_profile) or lora}, and no AI Toolkit job found in "
                             f"{AITK_OUTPUT_DIR} to build one. Pick the LoRA alone and write its style.")

        out = {k: [] for k in ("model", "clip", "style", "lyrics", "filename")}
        lines = []
        for key in keys:
            name = _profile_key(key)
            p = profiles[name]
            sd, meta = _load_lora(loras[key])
            m, c = comfy.sd.load_lora_for_models(model, clip, sd, p["strength_model"], p["strength_clip"],
                                                 lora_metadata=meta)
            out["model"].append(m)
            out["clip"].append(c)
            out["style"].append(p["style"])
            out["lyrics"].append(INSTRUMENTAL if p.get("instrumental") else lyrics)
            out["filename"].append(f"YuE2_covers/{_clean_name(song)}/{_clean_name(song)} - {_clean_name(name)}")
            lines.append(f"{name}: model {p['strength_model']} / clip {p['strength_clip']}"
                         + (" / instrumental" if p.get("instrumental") else "")
                         + (" / auto profile" if p.get("auto") else ""))
        if created:
            lines.append("new automatic profiles: " + ", ".join(created))
        if no_profile:
            lines.append("no profile and no AI Toolkit job (left out): " + ", ".join(no_profile))
        return (out["model"], out["clip"], out["style"], out["lyrics"], out["filename"], "\n".join(lines))


@PromptServer.instance.routes.get("/another_utils/yue2cover/lyrics")
async def _lyrics_route(request):
    return web.json_response({"lyrics": _read_lyrics(request.query.get("song", ""))})


@PromptServer.instance.routes.get("/another_utils/yue2cover/profile")
async def _profile_route(request):
    """Saved profile, or the automatic one it would get (not saved until the LoRA runs)."""
    key = request.query.get("lora", "")
    profile = _read_profiles().get(_profile_key(key))
    if profile is None and key in _loras():
        profile = _auto_profile(_loras()[key])
    return web.json_response({"profile": profile})
