"""Figures for analysis/results.md. Run after analysis/analyze.py:  python analysis/plots.py

Needs matplotlib. Writes analysis/figures/*.png (light theme, for a printed report).
Colors follow a validated reference palette: diverging blue <-> red with a gray midpoint,
blue as the single sequential hue, text in ink tokens, recessive grid and axes.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "analysis" / "figures"
S = json.load(open(ROOT / "analysis" / "out" / "summary.json", encoding="utf-8"))

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("ebd_div", [RED, MID, BLUE])
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
JUDGE_LABEL = {"claude-code": "Opus 5.5", "claude-code-fable": "Fable 5.2", "codex": "Astra", "antigravity": "Neon"}
MODEL_LABEL = {"haiku": "Haiku 5.5", "luna": "GPT-6 Luna", "deepseek": "DeepSeek\n(separate)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def strip(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def ink_on(value, vmax):
    return "#ffffff" if abs(value) > 0.6 * vmax else INK


def heatmap():
    rows = S["pair_rows"]
    qs = sorted({r["rubric_id"] for r in rows})
    models = ["haiku", "luna", "deepseek"]
    by = {(r["rubric_id"], r["model"]): r for r in rows}
    vmax = max(0.25, max(abs(r["diff_mean"]) for r in rows))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    xs = [0, 1, 2.25]  # a gap sets DeepSeek apart: it is analysed separately
    for x, m in zip(xs, models):
        for y, q in enumerate(qs):
            r = by[(q, m)]
            v = r["diff_mean"]
            ax.add_patch(plt.Rectangle((x - 0.48, y - 0.47), 0.96, 0.94, color=DIVERGING(norm(v)), lw=0))
            ax.text(x, y - 0.08, f"{v:+.3f}", ha="center", va="center", fontsize=10, fontweight="bold", color=ink_on(v, vmax))
            ax.text(x, y + 0.24, f"{r['judges_ebd_higher']}/4 judges", ha="center", va="center", fontsize=7.5, color=ink_on(v, vmax))
    ax.set_xlim(-0.6, 2.85)
    ax.set_ylim(len(qs) - 0.5, -0.5)
    ax.set_xticks(xs, [MODEL_LABEL[m] for m in models])
    ax.set_yticks(range(len(qs)), qs)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    strip(ax, keep=())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("EBD − baseline (score)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("EBD − baseline, mean of four judges\nblue: EBD scored higher · red: baseline scored higher", pad=34, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "1_matrix_ebd_minus_baseline.png", dpi=200)
    plt.close(fig)


def per_judge():
    rows = S["summary_rows"]  # [group, judge, n, ebd, base, diff, [lo, hi], higher, lower, p]
    groups = list(dict.fromkeys(r[0] for r in rows))
    fig, axes = plt.subplots(1, len(groups), figsize=(9, 3.4), sharex=True)
    for ax, g in zip(axes, groups):
        rs = [r for r in rows if r[0] == g]
        labels = [r[1] for r in rs]
        for y, r in enumerate(rs):
            lo, hi = r[6]
            strong = r[1] == "Judge mean"
            ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot([r[5]], [y], "o", ms=8 if strong else 7, color=BLUE if strong else SURFACE,
                    markeredgecolor=BLUE, markeredgewidth=2)
            ax.text(hi + 0.012, y, f"{r[5]:+.3f}  ({r[7]}/{r[2]}, p={r[9]:.2f})", va="center", fontsize=8, color=INK2)
        ax.axvline(0, color=BASE, lw=1)
        ax.set_yticks(range(len(rs)), labels)
        ax.set_ylim(len(rs) - 0.5, -0.6)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        strip(ax)
        ax.set_title(g, fontsize=10)
        ax.set_xlabel("Mean EBD − baseline, 95% bootstrap CI over pairs")
    axes[0].set_xlim(-0.16, 0.30)
    fig.suptitle("Every judge places EBD ahead in the main comparison; DeepSeek is level", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "2_per_judge_mean_diff.png", dpi=200)
    plt.close(fig)


def axes_breakdown():
    rows = S["axis_rows"]  # [axis, group_index, mean, n]
    axes_names = list(dict.fromkeys(r[0] for r in rows))
    titles = ["Main comparison (Haiku + Luna)", "DeepSeek (separate)"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, sharex=True)
    for gi, ax in enumerate(axs):
        for y, a in enumerate(axes_names):
            r = next(x for x in rows if x[0] == a and x[1] == gi)
            v, n = r[2], r[3]
            if v is None:
                continue
            ax.barh(y, v, height=0.6, color=BLUE if v >= 0 else RED, lw=0)
            ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:+.3f}  (n={n})", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
        ax.axvline(0, color=BASE, lw=1)
        ax.set_yticks(range(len(axes_names)), [a.replace(" & Citation Quality", " & citations") for a in axes_names])
        ax.set_ylim(len(axes_names) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        strip(ax)
        ax.set_title(titles[gi], fontsize=10)
        ax.set_xlabel("Mean EBD − baseline on the axis")
    axs[0].set_xlim(-0.25, 0.65)
    fig.suptitle("Where the difference comes from: rubric axes (n = pairs with that axis)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "3_axis_breakdown.png", dpi=200)
    plt.close(fig)


def agreement():
    judges = ["claude-code", "claude-code-fable", "codex", "antigravity"]
    k = {(a, b): r for a, b, *r in S["agree_rows"]}  # r = [exact, kappa, wkappa, corr]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for i, a in enumerate(judges):
        for j, b in enumerate(judges):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, color=GRID, lw=0))
                continue
            r = k.get((a, b)) or k.get((b, a))
            v = r[1]
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, color=SEQ((v - 0.5) / 0.5), lw=0))
            col = "#ffffff" if v > 0.75 else INK
            ax.text(j, i - 0.1, f"κ {v:.2f}", ha="center", va="center", fontsize=10, fontweight="bold", color=col)
            ax.text(j, i + 0.2, f"{r[0]:.0%} same", ha="center", va="center", fontsize=7.5, color=col)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_xticks(range(4), [JUDGE_LABEL[j] for j in judges])
    ax.set_yticks(range(4), [JUDGE_LABEL[j] for j in judges])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    strip(ax, keep=())
    ax.set_title(f"Agreement between judges over 834 gradings\nCohen's κ; Krippendorff's α across all four = {S['alpha']:.2f}", pad=30, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "4_judge_agreement.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    heatmap()
    per_judge()
    axes_breakdown()
    agreement()
    print("written:", sorted(p.name for p in FIG.glob("*.png")))
