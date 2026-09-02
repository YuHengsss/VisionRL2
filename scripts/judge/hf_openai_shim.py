"""Minimal OpenAI-compatible /v1/chat/completions shim around HF
transformers Qwen3.5 — DEFAULT processor settings (no pixel overrides),
faithful to Vision-OPD's protocol where the serving side's defaults
govern the token budget. Text-only requests (judge) also supported.

Usage: MODEL_ID=Qwen/Qwen3.5-4B PORT=8123 python hf_openai_shim.py
"""
import base64
import io
import os
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")
PORT = int(os.environ.get("PORT", "8123"))
MAX_NEW_CAP = int(os.environ.get("MAX_NEW_CAP", "1024"))

print(f"[shim] loading {MODEL_ID} ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="flash_attention_2")
model.eval()
# Optional pixel budget override (defaults: model's processor config).
_pp = {}
if os.environ.get("SHIM_MAX_PIXELS"):
    _pp["max_pixels"] = int(os.environ["SHIM_MAX_PIXELS"])
if os.environ.get("SHIM_MIN_PIXELS"):
    _pp["min_pixels"] = int(os.environ["SHIM_MIN_PIXELS"])
processor = AutoProcessor.from_pretrained(MODEL_ID, **_pp)
print(f"[shim] ready (pixel override: {_pp}, MAX_NEW_CAP={MAX_NEW_CAP})", flush=True)

app = FastAPI()
_lock = threading.Lock()


def _decode_data_uri(uri):
    b64 = uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/chat/completions")
def chat(body: dict):
    messages = body.get("messages", [])
    max_tokens = min(int(body.get("max_tokens", 512)), MAX_NEW_CAP)
    tk = (body.get("chat_template_kwargs")
          or {}).get("enable_thinking", None)
    enable_thinking = bool(tk) if tk is not None else False

    hf_msgs, images = [], []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            hf_msgs.append({"role": m["role"], "content": [
                {"type": "text", "text": content}]})
            continue
        items = []
        for c in content:
            if c.get("type") == "image_url":
                img = _decode_data_uri(c["image_url"]["url"])
                images.append(img)
                items.append({"type": "image", "image": img})
            elif c.get("type") == "text":
                items.append({"type": "text", "text": c.get("text", "")})
        hf_msgs.append({"role": m["role"], "content": items})

    try:
        text = processor.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        text = processor.apply_chat_template(
            hf_msgs, tokenize=False, add_generation_prompt=True)

    with _lock:
        inputs = processor(text=[text], images=images or None,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, do_sample=False,
                                 max_new_tokens=max_tokens)
        _new = gen[0][inputs.input_ids.shape[1]:]
        out = processor.batch_decode(
            gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        # Real finish_reason. Previously hardcoded "stop", which masked
        # MAX_NEW_CAP truncation (default 1024) and made a non-terminating
        # generation indistinguishable from a clean EOS in saved answers.
        _eos = model.generation_config.eos_token_id
        _eos = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
        _eos.discard(None)
        _n_new = int(_new.shape[0])
        _hit_eos = bool(_new.numel() and int(_new[-1].item()) in _eos)
        _finish_reason = "stop" if _hit_eos else (
            "length" if _n_new >= max_tokens else "stop")

    return {
        "id": "shim", "object": "chat.completion",
        "created": int(time.time()), "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": _finish_reason,
                     "message": {"role": "assistant", "content": out}}],
        "usage": {"prompt_tokens": int(inputs.input_ids.shape[1]),
                  "completion_tokens": int(gen.shape[1] - inputs.input_ids.shape[1]),
                  "total_tokens": int(gen.shape[1])},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
