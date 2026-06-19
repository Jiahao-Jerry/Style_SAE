# Style SAE

A Sparse Autoencoder (SAE) trained on style-axis diff vectors extracted from Qwen2.5-3B, to discover whether LLM internals encode the same writing style dimensions a human annotator would label.

---

## Idea

Take 700 Bluesky posts. For each post × each of 7 style axes, use Claude Haiku to rewrite the post shifting that one axis (keeping content identical). Extract Qwen2.5-3B layer-23 activations for the original and rewrite, subtract to get a labeled diff vector. Train an SAE on the 4,900 diff vectors and check whether learned features align with the known axes.

---

## Pipeline

```
2550_posts.jsonl  (from github.com/Jiahao-Jerry/SURE)
        │
        ▼  step1_generate_rewrites.py     [Local]
        Claude Haiku rewrites 700 posts × 7 axes = 4,900 pairs
        Direction chosen by current score (shift toward opposite end)
        → sae_clean_rewrites.json
        │
        ▼  step2_colab_qwen_embed.py      [Colab T4]
        Qwen2.5-3B layer-23 activations for original + rewrite
        diff_vector = emb(rewrite) − emb(original)  →  shape (4900, 2048)
        → diff_vectors_4900.npy
        → axis_labels_4900.npy   (which axis was shifted)
        → directions_4900.npy    (up / down)
        → post_ids_4900.npy
        │
        ▼  step3_train_sae.py             [Local]
        SAE: 64 features, L1=0.10, 600 epochs, input_dim=2048
        Axis recovery measured by lift = P(active | axis=A) − P(active | axis≠A)
        → sae_qwen_model.pt
        → sae_qwen_results.png
```

---

## Scripts

| Script | Where | Description |
|---|---|---|
| `step1_generate_rewrites.py` | Local | Generates Claude Haiku rewrites; caches to JSON |
| `step2_colab_qwen_embed.py` | Google Colab (T4) | Extracts Qwen2.5-3B layer-23 diffs |
| `step3_train_sae.py` | Local | Trains SAE; reports per-axis feature lift |

---

## Data Files

| File | Shape / Size | Description |
|---|---|---|
| `sae_clean_rewrites.json` | 700 posts × 7 axes | Original + rewrite text pairs with axis metadata |
| `diff_vectors_4900.npy` | (4900, 2048) float32 | L2-normalized Qwen diff vectors |
| `axis_labels_4900.npy` | (4900,) | Axis name for each pair |
| `directions_4900.npy` | (4900,) | `up` or `down` shift direction |
| `post_ids_4900.npy` | (4900,) | Source post ID for each pair |
| `sae_qwen_model.pt` | — | Trained SAE weights |
| `sae_qwen_results.png` | — | Training loss, density, lift heatmap, per-axis coverage |

---

## 7 Style Axes

| Axis | 0 → 1 |
|---|---|
| `reading_level` | simple vocabulary → academic / complex |
| `background` | assumes no prior knowledge → assumes expert knowledge |
| `abstract_concrete` | vague general claims → specific facts / numbers |
| `tone` | analytical / neutral → emotional / charged |
| `humor` | earnest → witty / humorous |
| `narrativity` | pure argument → story / anecdote |
| `grounding` | direct statement → analogy / example-driven |

---

## Requirements

```
pip install anthropic sentence-transformers transformers accelerate torch numpy pandas matplotlib
```

Set your API key before running Step 1:
```
export ANTHROPIC_API_KEY=your_key_here
```

Step 2 runs on Google Colab — upload `sae_clean_rewrites.json` and download the four `.npy` files after it completes.

---

## Source Dataset

Posts come from [SURE](https://github.com/Jiahao-Jerry/SURE) — 2,550 curated Bluesky posts across 17 topics, each annotated with the 7 style axes above.
