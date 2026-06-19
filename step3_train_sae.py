"""
Step 3 (Local) — Train SAE on Qwen diff vectors and interpret features.

Inputs (downloaded from Colab after Step 2):
  - diff_vectors_NNNN.npy
  - axis_labels_NNNN.npy
  - directions_NNNN.npy

Outputs:
  - sae_qwen_model.pt
  - sae_qwen_results.png
"""

import json
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
N_FEATURES   = 64
L1_COEF      = 0.10
LR           = 1e-3
EPOCHS       = 600
BATCH_SIZE   = 128
SEED         = 42
DEAD_DENSITY = 0.01
CONFIRM_LIFT = 0.30
PARTIAL_LIFT = 0.15
OUT_MODEL    = "sae_qwen_model.pt"
OUT_PNG      = "sae_qwen_results.png"

AXIS_NAMES  = ["reading_level", "background", "abstract_concrete",
               "tone", "humor", "narrativity", "grounding"]
AXIS_SHORT  = ["read_lvl", "backgrnd", "abst_conc", "tone", "humor", "narrat", "ground"]

# ── Load diff vectors ─────────────────────────────────────────────
# Auto-detect the file from Colab output
diff_files = sorted(glob.glob("diff_vectors_*.npy"))
if not diff_files:
    raise FileNotFoundError("No diff_vectors_*.npy found. Run step2_colab_qwen_embed.py on Colab first.")

diff_file = diff_files[-1]          # use latest
n_str     = diff_file.split("_")[-1].replace(".npy", "")
print(f"Loading {diff_file}...")

diff_vecs   = np.load(diff_file).astype(np.float32)
axis_labels = np.load(f"axis_labels_{n_str}.npy")
directions  = np.load(f"directions_{n_str}.npy")

input_dim = diff_vecs.shape[1]      # 3584 for Qwen2.5-7B
print(f"  Diff vectors: {diff_vecs.shape}  (input_dim={input_dim})")
print(f"  Axis distribution: {dict(Counter(axis_labels))}")

# L2-normalize
norms     = np.linalg.norm(diff_vecs, axis=1, keepdims=True).clip(min=1e-8)
diff_vecs = diff_vecs / norms
print(f"  L2-normalized. Mean norm before: {norms.mean():.4f}")

# ── SAE ───────────────────────────────────────────────────────────
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, n_features, seed=42):
        super().__init__()
        gen   = torch.Generator().manual_seed(seed)
        w_enc = torch.empty(n_features, input_dim)
        nn.init.kaiming_uniform_(w_enc, a=5**0.5, generator=gen)
        self.W_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        w_dec      = torch.randn(input_dim, n_features, generator=gen)
        w_dec      = w_dec / w_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(w_dec)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x):
        return torch.relu(x @ self.W_enc.T + self.b_enc)

    def forward(self, x):
        f = self.encode(x)
        return f, f @ self.W_dec.T + self.b_dec

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.data.div_(norms)

# ── Train ─────────────────────────────────────────────────────────
print(f"\nTraining SAE (F={N_FEATURES}, L1={L1_COEF}, epochs={EPOCHS}, input_dim={input_dim})...")
torch.manual_seed(SEED)
model     = SparseAutoencoder(input_dim, N_FEATURES, SEED)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
X         = torch.from_numpy(diff_vecs)

loss_history    = []
density_history = []

for epoch in range(EPOCHS):
    perm        = torch.randperm(len(X))
    epoch_recon = 0.0
    for i in range(0, len(X), BATCH_SIZE):
        batch = X[perm[i:i+BATCH_SIZE]]
        optimizer.zero_grad()
        feats, recon = model(batch)
        recon_loss   = ((batch - recon)**2).sum(dim=1).mean()
        sparsity     = feats.abs().sum(dim=1).mean()
        loss         = recon_loss + L1_COEF * sparsity
        loss.backward()
        optimizer.step()
        model.normalize_decoder()
        epoch_recon += float(recon_loss.detach())

    with torch.no_grad():
        density = (model.encode(X) > 0).float().mean().item()
    loss_history.append(epoch_recon)
    density_history.append(density)

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:>4}/{EPOCHS}  recon={epoch_recon:.4f}  density={density:.3f}")

torch.save(model.state_dict(), OUT_MODEL)
print(f"  Saved model → {OUT_MODEL}")

# ── Activations ───────────────────────────────────────────────────
with torch.no_grad():
    acts = model.encode(X).numpy()
print(f"\nActivations: {acts.shape}  mean_density={(acts>0).mean():.3f}")

# ── Axis recovery ─────────────────────────────────────────────────
# Lift = P(feature active | axis=A) - P(feature active | axis!=A)
axis_arr    = np.array(axis_labels)
results     = []
lift_matrix = np.zeros((N_FEATURES, len(AXIS_NAMES)))

for f in range(N_FEATURES):
    feat = acts[:, f]
    dens = (feat > 0).mean()

    if dens < DEAD_DENSITY:
        results.append({"feature": f, "density": dens, "category": "dead",
                        "best_axis": None, "best_lift": 0.0, "lifts": {}})
        continue

    lifts = {}
    for ai, ax in enumerate(AXIS_NAMES):
        is_ax  = axis_arr == ax
        p_on   = (feat[is_ax]  > 0).mean() if is_ax.sum()  > 0 else 0.0
        p_off  = (feat[~is_ax] > 0).mean() if (~is_ax).sum() > 0 else 0.0
        lift   = float(p_on - p_off)
        lifts[ax]          = round(lift, 4)
        lift_matrix[f, ai] = lift

    best_ax   = max(lifts, key=lambda k: abs(lifts[k]))
    best_lift = lifts[best_ax]
    cat = ("confirms_axis"   if abs(best_lift) >= CONFIRM_LIFT else
           "partial_overlap" if abs(best_lift) >= PARTIAL_LIFT else
           "novel_candidate")

    results.append({"feature": f, "density": round(dens, 3), "category": cat,
                    "best_axis": best_ax, "best_lift": round(best_lift, 4),
                    "lifts": lifts})

# ── Print summary ─────────────────────────────────────────────────
cats = Counter(r["category"] for r in results)
print(f"\n=== Results ===")
print(f"Pairs: {len(diff_vecs)} | Features: {N_FEATURES} | Input dim: {input_dim}")
print(f"Categories: {dict(cats)}\n")

print(f"{'F':>3}  {'density':>8}  {'category':>16}  {'best_axis':>22}  {'lift':>8}")
print("─" * 68)
for r in sorted(results, key=lambda x: abs(x["best_lift"]), reverse=True)[:20]:
    print(f"{r['feature']:>3}  {r['density']:>8.3f}  {r['category']:>16}  "
          f"{str(r['best_axis']):>22}  {r['best_lift']:>8.4f}")

print("\nPer-axis coverage:")
by_axis = {}
for r in results:
    if r["category"] in ("confirms_axis", "partial_overlap"):
        by_axis.setdefault(r["best_axis"], []).append((r["feature"], r["best_lift"]))
for ax in AXIS_NAMES:
    hits   = by_axis.get(ax, [])
    status = "✓" if hits else "✗"
    print(f"  {status} {ax:<25} {[(f,l) for f,l in hits]}")

# ── Plot ──────────────────────────────────────────────────────────
print(f"\nGenerating plots → {OUT_PNG}")
DARK="#0f1117"; PANEL="#1e2130"; TEXT="#e8eaf0"; GRID="#2a2d3e"
ACC="#6c8ebf"; RED="#e05c5c"; GREEN="#5cb85c"; YEL="#e6a817"

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

# 1. Loss
ax1 = fig.add_subplot(gs[0,0]); style_ax(ax1)
ax1.plot(loss_history, color=ACC, linewidth=1.5)
ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True, color=GRID, lw=0.5)

# 2. Density
ax2 = fig.add_subplot(gs[0,1]); style_ax(ax2)
ax2.plot(density_history, color=YEL, linewidth=1.5)
ax2.axhline(0.15, color=GREEN, lw=1, ls="--", label="target ≤0.15")
ax2.set_title("Feature Density"); ax2.set_xlabel("Epoch")
ax2.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)
ax2.grid(True, color=GRID, lw=0.5)

# 3. Density histogram
ax3 = fig.add_subplot(gs[0,2]); style_ax(ax3)
densities = [(acts[:,f]>0).mean() for f in range(N_FEATURES)]
ax3.hist(densities, bins=15, color=ACC, edgecolor=DARK, alpha=0.85)
ax3.axvline(np.mean(densities), color=YEL, lw=1.5, ls="--",
            label=f"mean={np.mean(densities):.2f}")
ax3.set_title("Feature Density Distribution"); ax3.set_xlabel("Fraction Active")
ax3.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)
ax3.grid(True, color=GRID, lw=0.5)

# 4. Heatmap
ax4 = fig.add_subplot(gs[1,:]); style_ax(ax4)
order     = sorted(range(N_FEATURES), key=lambda f: abs(lift_matrix[f]).max(), reverse=True)
heat_data = lift_matrix[order]
vmax      = max(abs(heat_data).max(), 0.10)
im = ax4.imshow(heat_data.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax4.set_yticks(range(len(AXIS_SHORT))); ax4.set_yticklabels(AXIS_SHORT, color=TEXT, fontsize=9)
ax4.set_xticks(range(N_FEATURES))
ax4.set_xticklabels([str(order[i]) for i in range(N_FEATURES)],
                    rotation=90, fontsize=6, color=TEXT)
ax4.set_title("Axis Lift Heatmap (features sorted by max |lift|, red=positive, blue=negative)")
ax4.set_xlabel("Feature (sorted)"); ax4.set_ylabel("Axis")
cb = fig.colorbar(im, ax=ax4, fraction=0.015, pad=0.01)
cb.ax.tick_params(colors=TEXT, labelsize=7); cb.set_label("lift", color=TEXT)

# 5. Per-axis bar
ax5 = fig.add_subplot(gs[2,0]); style_ax(ax5)
c_counts = [sum(1 for r in results if r["best_axis"]==ax and r["category"]=="confirms_axis")  for ax in AXIS_NAMES]
p_counts = [sum(1 for r in results if r["best_axis"]==ax and r["category"]=="partial_overlap") for ax in AXIS_NAMES]
x = np.arange(len(AXIS_NAMES)); w = 0.35
ax5.bar(x-w/2, c_counts, w, label="confirms", color=GREEN, alpha=0.85)
ax5.bar(x+w/2, p_counts, w, label="partial",  color=YEL,   alpha=0.85)
ax5.set_xticks(x); ax5.set_xticklabels(AXIS_SHORT, rotation=30, ha="right", fontsize=7)
ax5.set_title("Features per Axis"); ax5.set_ylabel("# Features")
ax5.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)
ax5.grid(True, color=GRID, lw=0.5, axis="y")

# 6. Top features
ax6 = fig.add_subplot(gs[2,1]); style_ax(ax6)
top10 = sorted([r for r in results if r["best_lift"]!=0],
               key=lambda r: abs(r["best_lift"]), reverse=True)[:10]
ax6.barh(range(len(top10)),
         [r["best_lift"] for r in top10],
         color=[GREEN if r["best_lift"]>0 else RED for r in top10], alpha=0.85)
ax6.set_yticks(range(len(top10)))
ax6.set_yticklabels([f"F{r['feature']} {r['best_axis'][:7]}" for r in top10], fontsize=7)
ax6.set_title("Top-10 Features by |Lift|"); ax6.set_xlabel("Lift")
ax6.axvline(0, color=GRID, lw=1)
ax6.axvline( CONFIRM_LIFT, color=GREEN, lw=1, ls="--", alpha=0.5, label=f"confirm≥{CONFIRM_LIFT}")
ax6.axvline(-CONFIRM_LIFT, color=GREEN, lw=1, ls="--", alpha=0.5)
ax6.axvline( PARTIAL_LIFT, color=YEL,   lw=1, ls="--", alpha=0.5, label=f"partial≥{PARTIAL_LIFT}")
ax6.axvline(-PARTIAL_LIFT, color=YEL,   lw=1, ls="--", alpha=0.5)
ax6.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=6)
ax6.grid(True, color=GRID, lw=0.5, axis="x")

# 7. Pie
ax7 = fig.add_subplot(gs[2,2]); ax7.set_facecolor(PANEL)
cat_colors = {"confirms_axis": GREEN, "partial_overlap": YEL,
              "novel_candidate": ACC, "dead": "#555"}
ax7.pie(list(cats.values()), labels=list(cats.keys()),
        colors=[cat_colors.get(c, ACC) for c in cats],
        autopct="%1.0f%%", startangle=140,
        textprops={"color": TEXT, "fontsize": 8})
ax7.set_title("Feature Categories", color=TEXT)

fig.suptitle(
    f"SAE on Qwen2.5-7B Diff Vectors  |  {len(diff_vecs)} pairs  "
    f"·  F={N_FEATURES}  ·  L1={L1_COEF}  ·  {EPOCHS} epochs  ·  dim={input_dim}",
    color=TEXT, fontsize=11, y=0.98)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved → {OUT_PNG}")
import subprocess; subprocess.Popen(["open", OUT_PNG])
