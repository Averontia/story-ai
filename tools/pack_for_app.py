# Packs the model in the app's format without the official converter (it crashes on Colab's newer setup).
# It copies the official app-ready copy's exact layout and only replaces the numbers. It first packs the
# ORIGINAL Llama and compares with the official copy, so we know the packing is right before using it.
import os, glob, json, hashlib, shutil, numpy as np, torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

# If Colab restarted, the settings from Step 1 and 3 are forgotten. Fill them back in.
if "SIZE" not in globals(): SIZE = "1B"
if "REPO_NAME" not in globals(): REPO_NAME = f"Llama-3.2-{SIZE}-StoryAI"
if "BASE" not in globals(): BASE = f"unsloth/Llama-3.2-{SIZE}-Instruct"
if not glob.glob("storyai-hf/*.safetensors"):
    raise SystemExit("Colab reset its computer, so the model you made is gone. "
                     "Menu: Runtime > Run all, and sign in again when the box appears.")

README = f"""---
library_name: mlc-llm
tags: [mlc-llm, web-llm]
base_model: meta-llama/Llama-3.2-{SIZE}-Instruct
license: llama3.2
---
# {REPO_NAME}

Llama 3.2 {SIZE} Instruct with its habit of holding back on dark fiction removed, packed for WebLLM, for the Story AI app.

Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.
See https://www.llama.com/llama3_2/license/ and the Acceptable Use Policy https://www.llama.com/llama3_2/use-policy/
"""

class Weights:
    """Reads a Hugging Face model's tensors one at a time, as float32 numpy arrays."""
    def __init__(self, folder):
        self.where = {}
        for f in glob.glob(os.path.join(folder, "*.safetensors")):
            with safe_open(f, "pt") as s:
                for k in s.keys(): self.where[k] = f
    def get(self, key):
        with safe_open(self.where[key], "pt") as s:
            return s.get_tensor(key).float().numpy()

def source(W, name):
    base = name.rsplit(".", 1)[0]          # drop .q_weight / .q_scale / .weight
    if base.endswith("qkv_proj"):
        p = base[: -len("qkv_proj")]; parts = [p + "q_proj", p + "k_proj", p + "v_proj"]
    elif base.endswith("gate_up_proj"):
        p = base[: -len("gate_up_proj")]; parts = [p + "gate_proj", p + "up_proj"]
    else:
        parts = [base]
    return base, np.concatenate([W.get(x + ".weight") for x in parts], 0)

def quantize(w, mdt, rows=8192):
    """4-bit groups of 32 along each row, 8 values per uint32, like the app's q4f16_1 / q4f32_1."""
    n, k = w.shape
    packed = np.zeros((n, k // 8), np.uint32)
    scale = np.zeros((n, k // 32), mdt)
    for a in range(0, n, rows):
        g = w[a:a + rows].astype(mdt).reshape(-1, k // 32, 32)
        sc = (np.abs(g).max(-1) / mdt(7)).astype(mdt)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (g / sc[..., None]).astype(mdt)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        q = np.clip(np.round((s + mdt(7)).astype(mdt)), 0, 14).astype(np.uint32).reshape(-1, k // 8, 8)
        p = np.zeros(q.shape[:2], np.uint32)
        for j in range(8): p |= q[:, :, j] << np.uint32(4 * j)
        packed[a:a + rows] = p
        scale[a:a + rows] = sc
    return packed, scale

def encode(arr, rec):
    if rec["dtype"] == "float32" and rec.get("format") == "f32-to-bf16":   # stored as bfloat16
        u = np.ascontiguousarray(arr, np.float32).view(np.uint32).astype(np.uint64)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    return np.ascontiguousarray(arr.astype(rec["dtype"])).tobytes()

def pack(W, cache, mdt, only=None):
    """Returns {shard file: bytes} (or, with `only`, {record name: bytes} for just those records)."""
    done, out = {}, {}
    for shard in cache["records"]:
        buf = None if only else bytearray(shard["nbytes"])
        for rec in shard["records"]:
            name = rec["name"]
            if only and name not in only: continue
            if name.endswith(".q_weight") or name.endswith(".q_scale"):
                base = name.rsplit(".", 1)[0]
                if base not in done:
                    _, w = source(W, name)
                    done[base] = dict(zip(("q_weight", "q_scale"), quantize(w, mdt)))
                arr = done[base][name.rsplit(".", 1)[1]]
            else:
                arr = source(W, name)[1]
            data = encode(arr.reshape(rec["shape"]), rec)
            assert len(data) == rec["nbytes"], (name, len(data), rec["nbytes"])
            if only: out[name] = data
            else: buf[rec["byteOffset"]:rec["byteOffset"] + len(data)] = data
        if not only: out[shard["dataPath"]] = bytes(buf)
    return out

original = Weights(snapshot_download(BASE, allow_patterns=["*.safetensors", "*.json"]))
mine = Weights("storyai-hf")

for q in ["q4f16_1", "q4f32_1"]:
    mdt = np.float16 if q == "q4f16_1" else np.float32
    out = f"pack-{q}"
    shutil.rmtree(out, ignore_errors=True)
    snapshot_download(f"mlc-ai/Llama-3.2-{SIZE}-Instruct-{q}-MLC", local_dir=out)
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)
    cache_file = next(os.path.join(out, f) for f in ["ndarray-cache.json", "tensor-cache.json"] if os.path.exists(os.path.join(out, f)))
    cache = json.load(open(cache_file))

    # 1. Check: pack a few parts of the ORIGINAL Llama and compare with the official copy, byte for byte.
    where = {r["name"]: (s["dataPath"], r) for s in cache["records"] for r in s["records"]}
    test = ["model.layers.0.self_attn.qkv_proj.q_weight", "model.layers.0.self_attn.qkv_proj.q_scale",
            "model.layers.0.input_layernorm.weight", "model.layers.3.mlp.down_proj.q_weight"]
    mine_bytes = pack(original, cache, mdt, only=set(test))
    for name in test:
        path, rec = where[name]
        with open(os.path.join(out, path), "rb") as fh:
            fh.seek(rec["byteOffset"]); official = fh.read(rec["nbytes"])
        # Compare values, not bytes: the official converter rounds the last digit slightly differently,
        # which doesn't matter. A layout mistake would make the values unrelated, which this catches.
        if rec["dtype"] == "uint32":
            a = np.frombuffer(official, np.uint32); b = np.frombuffer(mine_bytes[name], np.uint32)
            na = np.stack([(a >> np.uint32(4 * j)) & 15 for j in range(8)]).astype(int)
            nb = np.stack([(b >> np.uint32(4 * j)) & 15 for j in range(8)]).astype(int)
            same = (na == nb).mean(); close = (np.abs(na - nb) <= 1).mean()
            ok = same > 0.95 and close > 0.999
            print(f"  check {name}: {100 * same:.2f}% the same, {100 * close:.3f}% within one step")
        else:
            def vals(buf):
                if rec["dtype"] == "float32" and rec.get("format") == "f32-to-bf16":
                    return (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
                return np.frombuffer(buf, rec["dtype"]).astype(np.float32)
            a, b = vals(official), vals(mine_bytes[name])
            rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
            ok = np.quantile(rel, 0.999) < 0.02
            print(f"  check {name}: values differ by {100 * rel.mean():.4f}% on average")
        if not ok:
            raise SystemExit("The packing doesn't match the official copy. Copy the text above to Claude.")

    # 2. Pack your model in the same layout.
    for path, data in pack(mine, cache, mdt).items():
        with open(os.path.join(out, path), "wb") as fh: fh.write(data)
        for s in cache["records"]:
            if s["dataPath"] == path: s["md5sum"] = hashlib.md5(data).hexdigest()
    for name in ["ndarray-cache.json", "tensor-cache.json"]:
        with open(os.path.join(out, name), "w") as fh: json.dump(cache, fh)
    with open(os.path.join(out, "README.md"), "w") as fh: fh.write(README)
    print(q, "packed:", len(cache["records"]), "pieces\n")
