"""Generate the two paper figures as vector PDF (+ PNG preview).

Figure 1 (fig-linker): the memory-linker pipeline — compile once offline,
then link at an arbitrary position: load, RoPE rebase, fuse.

Figure 2 (fig-regimes): E12 (paper section 5.6) — prefill cost scales with
available compute while restore cost is flat, so the precompilation advantage
peaks exactly where compute is scarce.

Usage: python paper/figs/make_figs.py   (writes PDFs/PNGs next to itself)
"""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # embed TrueType so arXiv/PDF viewers render text as text
})

HERE = os.path.dirname(os.path.abspath(__file__))

C_SRC   = "#f2e8d5"  # source text
C_ART   = "#c9d7e8"  # compiled artifact
C_CTX   = "#e4e4e4"  # live context
C_MOD   = "#9fc2e0"  # linked module
C_EDGE  = "#444444"
C_ACC   = "#8a3324"  # accent (rebase)


def box(ax, x, y, w, h, text, fc, fontsize=8.2, ec=C_EDGE, lw=1.0, style="round,pad=0.02,rounding_size=0.045"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, color=C_EDGE, lw=1.3, style="-|>", shrink=0.0, connectionstyle=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, shrinkA=shrink, shrinkB=shrink,
                                 connectionstyle=connectionstyle))


# ----------------------------------------------------------------------------
# Figure 1: the linker pipeline
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 3.4))
ax.set_xlim(0, 100)
ax.set_ylim(-1, 47)
ax.axis("off")

# --- offline half ---
ax.text(24, 45.0, "offline — compile once per (model, ABI)", fontsize=8.6, style="italic", ha="center")
box(ax, 2, 31, 14, 9, "memory.md\n(source text)", C_SRC)
arrow(ax, 16.5, 35.5, 23.5, 35.5)
ax.text(19.8, 37.6, "mdc compile", fontsize=7.4, ha="center", va="bottom")
box(ax, 23, 26, 26, 16,
    ".kmd module\nper-layer K/V rows\n+ recurrent state $(T_M, S_M)$\nid = sha256(weights | tokenizer\n| text | ABI)",
    C_ART, fontsize=7.2)
ax.text(36, 23.4, "compiled at positions $[p_0,\\ p_0{+}n)$", fontsize=7.0, ha="center", style="italic")

# divider
ax.plot([50.5, 50.5], [3, 46], color="#bbbbbb", lw=0.8, ls=(0, (4, 3)))

# --- runtime half ---
ax.text(76, 45.0, "runtime — link at any position, no re-evaluation for single-module linking", fontsize=8.2, style="italic", ha="center")

# live context bar
bar_y, bar_h = 32.5, 6.5
box(ax, 54, bar_y, 11, bar_h, "system", C_CTX, fontsize=7.4, style="round,pad=0.02,rounding_size=0.02")
box(ax, 65, bar_y, 16, bar_h, "conversation…", C_CTX, fontsize=7.4, style="round,pad=0.02,rounding_size=0.02")
box(ax, 81, bar_y, 14, bar_h, "module", C_MOD, fontsize=8.0, style="round,pad=0.02,rounding_size=0.02")
ax.text(88, bar_y - 2.4, "insertion at position $p$", fontsize=7.0, ha="center", style="italic")
ax.text(96.2, bar_y + bar_h / 2, "→", fontsize=9, va="center")

# module blob flying in: from artifact to slot
arrow(ax, 48.4, 34.5, 80.4, 35.7, color=C_EDGE, lw=1.2,
      connectionstyle="arc3,rad=-0.28")
ax.text(64, 41.2, "load + link", fontsize=7.2, ha="center", style="italic", color=C_ACC)

# steps
steps = [
    ("1. load", "blob → auxiliary sequence, at compiled positions $p_0$"),
    ("2. rebase", "RoPE re-rotation of K by $\\Delta = p - p_0$"),
    ("3. fuse", "merge sequences; recurrent state\n$S \\leftarrow S_M$ (naive) or $T_M S_P + S_M$ (affine)"),
]
sx = 54
for i, (t, d) in enumerate(steps):
    y = 22.5 - i * 7.4
    ax.text(sx, y, t, fontsize=8.2, weight="bold", va="center",
            color=C_ACC if i == 1 else "#222222")
    ax.text(sx + 11, y, d, fontsize=7.2, va="center")

ax.text(50.5, 0.2, "cost = O(module bytes), not O(prefill compute); multi-module composition adds splice-$k$ (~1/3 of the incoming module recomputed)",
        fontsize=7.0, ha="center", style="italic")

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(HERE, "fig-linker.pdf"))
fig.savefig(os.path.join(HERE, "fig-linker.png"), dpi=180)
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 2: compute-regime sweep (E12)
# ----------------------------------------------------------------------------
regimes  = ["CPU only\n(20 cores)", "partial offload\n(12/28 layers)", "full GPU\n(RTX 4070 Ti SUPER)"]
# E12, N=5 repetitions on machine B (fresh boot). Restore is cold NVMe:
# posix_fadvise(DONTNEED) evicts the .kmd from the OS page cache before each
# timed read. Bars = median; whiskers = min–max over the 5 runs. Recall 6/6 in
# every run and regime. Restore is dominated by state relocation (copy + K-shift),
# not the disk read, which is why the spread is sub-2 %.
prefill      = [18.875, 13.669, 5.454]
prefill_err  = [[0.308, 0.157, 0.006], [0.197, 0.095, 0.006]]   # [lower, upper] to min/max
restore      = [0.685, 0.719, 0.782]
restore_err  = [[0.007, 0.006, 0.010], [0.009, 0.006, 0.005]]
ratio        = ["27.6×", "19.0×", "7.0×"]

fig, ax = plt.subplots(figsize=(4.6, 2.9))
x = range(len(regimes))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], prefill, w, yerr=prefill_err, capsize=2.5,
            error_kw=dict(lw=0.8, ecolor="#333333"),
            label="prefill (recompute text)", color="#b0b0b0", edgecolor="#555555", lw=0.6)
b2 = ax.bar([i + w / 2 for i in x], restore, w, yerr=restore_err, capsize=2.5,
            error_kw=dict(lw=0.8, ecolor="#12314a"),
            label="restore .kmd (cold NVMe, in timer)", color="#5b8db8", edgecolor="#2e5878", lw=0.6)

for i, (p, r, t) in enumerate(zip(prefill, restore, ratio)):
    ax.text(i - w / 2, p + prefill_err[1][i] + 0.35, f"{p:.1f} s", ha="center", fontsize=8)
    ax.text(i + w / 2, r + 0.55, f"{r:.2f} s", ha="center", fontsize=8)
    ax.annotate(t, (i, p + prefill_err[1][i] + 2.6), ha="center", fontsize=10, weight="bold", color="#8a3324")

ax.set_xticks(list(x))
ax.set_xticklabels(regimes, fontsize=7.5)
ax.set_ylabel("setup time (s)", fontsize=9)
ax.set_ylim(0, 23.5)
ax.legend(fontsize=7.6, frameon=False, loc="upper right")
ax.set_title("Same 15.2k-token module (870 MB f16) — E12 (N=5, cold NVMe)", fontsize=8.2)

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "fig-regimes.pdf"))
fig.savefig(os.path.join(HERE, "fig-regimes.png"), dpi=180)
plt.close(fig)

print("written:", ", ".join(sorted(f for f in os.listdir(HERE) if f.startswith("fig-"))))
