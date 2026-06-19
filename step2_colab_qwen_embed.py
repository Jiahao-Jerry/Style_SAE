"""
Step 2 (Colab GPU) — Extract Qwen2.5-3B layer-23 activations for all
original posts and their rewrites, then compute diff vectors.

Upload to Colab:
  - sae_clean_rewrites.json   (the cache from Step 1)

Download after running:
  - diff_vectors_4900.npy     (N_pairs, 2048) float32
  - axis_labels_4900.npy      (N_pairs,) string array
  - directions_4900.npy       (N_pairs,) string array
  - post_ids_4900.npy         (N_pairs,) string array

Runtime: ~10 min on T4
"""

# ── Cell 1: Install ───────────────────────────────────────────────
# !pip install transformers accelerate -q

# ── Cell 2: Run ───────────────────────────────────────────────────
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

CACHE_FILE = "sae_clean_rewrites.json"
MODEL_NAME = "Qwen/Qwen2.5-3B"
LAYER_IDX  = 23
MAX_LENGTH = 128
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.bfloat16
AXIS_NAMES = [
    "reading_level", "background", "abstract_concrete",
    "tone", "humor", "narrativity", "grounding",
]

with open(CACHE_FILE) as f:
    raw = json.load(f)

# Support new list format and old flat-dict format
if isinstance(raw, list):
    posts = [p for p in raw if len(p.get("rewrites", {})) == 7]
else:
    by_post = defaultdict(dict)
    for key, r in raw.items():
        by_post[r["post_id"]][r["axis"]] = r
    posts = [
        {"post_id": pid, "original": list(axes.values())[0]["original"], "rewrites":
         {ax: {"direction": r["direction"], "rewrite": r["rewrite"]} for ax, r in axes.items()}}
        for pid, axes in by_post.items() if len(axes) == 7
    ]

print(f"Posts with all 7 rewrites: {len(posts)}")

records = []
for post in posts:
    for ax in AXIS_NAMES:
        r = post["rewrites"][ax]
        records.append({
            "post_id":   post["post_id"],
            "axis":      ax,
            "direction": r["direction"],
            "original":  post["original"],
            "rewrite":   r["rewrite"],
        })
print(f"Total pairs: {len(records)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, low_cpu_mem_usage=True
)
model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}  ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")

captured = {}

def hook(module, inputs, output):
    captured["acts"] = output[0] if isinstance(output, tuple) else output

handle = model.model.layers[LAYER_IDX].register_forward_hook(hook)

def embed_texts(texts):
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            model(**enc, use_cache=False)
        acts   = captured["acts"].float()
        mask   = enc.attention_mask.unsqueeze(-1).float()
        pooled = (acts * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_vecs.append(pooled.cpu().numpy())
        if (i // BATCH_SIZE + 1) % 50 == 0:
            print(f"  {i + len(batch)}/{len(texts)} embedded")
    return np.vstack(all_vecs).astype(np.float32)

print("\nEmbedding originals...")
emb_orig = embed_texts([r["original"] for r in records])
print(f"  originals shape: {emb_orig.shape}")

print("\nEmbedding rewrites...")
emb_rew = embed_texts([r["rewrite"] for r in records])
print(f"  rewrites shape:  {emb_rew.shape}")

handle.remove()

diff_vectors = emb_rew - emb_orig
axis_labels  = np.array([r["axis"]      for r in records])
directions   = np.array([r["direction"] for r in records])
post_ids     = np.array([r["post_id"]   for r in records])

print(f"\nDiff vectors: {diff_vectors.shape}")
print(f"Mean diff norm: {np.linalg.norm(diff_vectors, axis=1).mean():.4f}")

n = len(records)
np.save(f"diff_vectors_{n}.npy", diff_vectors)
np.save(f"axis_labels_{n}.npy",  axis_labels)
np.save(f"directions_{n}.npy",   directions)
np.save(f"post_ids_{n}.npy",     post_ids)

print(f"\nSaved:")
print(f"  diff_vectors_{n}.npy  {diff_vectors.shape}")
print(f"  axis_labels_{n}.npy")
print(f"  directions_{n}.npy")
print(f"  post_ids_{n}.npy")
print(f"\nDownload these 4 files and run step3_train_sae.py locally.")

from google.colab import files
files.download(f"diff_vectors_{n}.npy")
files.download(f"axis_labels_{n}.npy")
files.download(f"directions_{n}.npy")
files.download(f"post_ids_{n}.npy")
