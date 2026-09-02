"""Regenerate figures/token_curves_main.pdf on the AVG.6 convention.

Layout mirrors the previous paper figure: top row = two 6-bench-average
panels (4B, 9B) with dotted Base/SD-RPN @4096 reference levels; below =
2x6 per-benchmark grid (row 1 = 4B, row 2 = 9B). Palette harmonized
with figures/acc_latency.pdf (base gray squares, SD-RPN orange circles,
ours blue stars). Data: draft_design.md per-bench cap grids (fin/placebo
RL, v4mix PA, HF base); token x-ladders = measured suite means (src+crop).
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CAPS = [576, 1024, 2048, 4096]

TOK = {
    "4B": {"base": [553, 992, 1967, 3484], "PA": [980, 1602, 2798, 4206],
           "RL": [870, 1487, 2725, 4200]},
    "9B": {"base": [553, 992, 1967, 3483], "PA": [726, 1276, 2327, 3648],
           "RL": [865, 1459, 2674, 4139]},
}

# RL @4096 cells = cc4096a2_* (sigma auto2, paper convention; fin ckpts
# verified). RL <=2048 unchanged (auto == auto2 there). PA @4096 = the
# 2026-08-06 v4mix+auto2 re-run (pa4096a2v4mix_*; AVG6 71.14/76.09) --
# supersedes both stale-auto and the invalid May-ckpt cc4096a2 PA legs.
PERBENCH = {
    "V* Bench": {
        "4B": {"base": [65.97, 74.35, 78.01, 82.20], "PA": [82.72, 81.68, 85.86, 85.86], "RL": [85.34, 87.96, 89.01, 91.62]},
        "9B": {"base": [69.11, 76.96, 84.82, 86.91], "PA": [83.25, 90.05, 92.15, 91.10], "RL": [89.53, 90.58, 94.76, 96.34]},
    },
    "MME-RW-Lite": {
        "4B": {"base": [40.96, 42.05, 45.49, 47.58], "PA": [48.93, 50.86, 52.11, 52.37], "RL": [50.96, 52.11, 53.88, 54.19]},
        "9B": {"base": [45.34, 52.37, 50.50, 56.18], "PA": [56.44, 56.59, 58.31, 57.48], "RL": [56.75, 57.63, 59.20, 59.30]},
    },
    "HR-Bench 4K": {
        "4B": {"base": [63.50, 68.50, 72.50, 75.62], "PA": [71.12, 72.88, 74.50, 77.00], "RL": [77.38, 77.88, 78.63, 78.62]},
        "9B": {"base": [64.12, 67.37, 71.37, 76.38], "PA": [76.00, 76.50, 79.00, 79.50], "RL": [78.50, 79.38, 81.37, 82.00]},
    },
    "HR-Bench 8K": {
        "4B": {"base": [56.37, 60.75, 65.62, 67.62], "PA": [63.25, 66.87, 69.75, 70.12], "RL": [70.87, 76.25, 76.25, 75.50]},
        "9B": {"base": [56.87, 61.12, 63.00, 69.12], "PA": [70.25, 72.25, 74.37, 79.00], "RL": [73.38, 75.38, 79.50, 80.88]},
    },
    "ZoomBench": {
        "4B": {"base": [40.47, 45.44, 49.94, 52.31], "PA": [55.62, 55.62, 57.99, 58.22], "RL": [61.78, 66.15, 67.57, 66.27]},
        "9B": {"base": [46.86, 47.69, 52.19, 53.02], "PA": [59.29, 62.13, 63.43, 63.31], "RL": [62.72, 65.21, 67.10, 66.63]},
    },
    "InfoVQA": {
        "4B": {"base": [69.82, 77.89, 79.82, 81.62], "PA": [78.10, 81.60, 82.91, 83.28], "RL": [80.45, 84.51, 84.91, 85.32]},
        "9B": {"base": [74.77, 81.47, 83.66, 84.09], "PA": [82.72, 85.24, 85.72, 86.16], "RL": [84.08, 86.02, 87.12, 87.12]},
    },
}

def avg6(model, stack):
    rows = [PERBENCH[b][model][stack] for b in PERBENCH]
    return [round(sum(r[i] for r in rows) / 6, 2) for i in range(4)]

STYLE = {
    "base": dict(color="#8a8a8a", marker="s", ls="--", label="Base"),
    "PA":   dict(color="#d99143", marker="o", ls="-", label="SD-RPN"),
    "RL":   dict(color="#2e6fa3", marker="*", ls="-", label="Ours"),
}

fig = plt.figure(figsize=(10.4, 8.6))
gs = fig.add_gridspec(3, 6, height_ratios=[1.35, 1, 1], hspace=0.42, wspace=0.48)

# top: AVG.6 panels
for k, model in enumerate(["4B", "9B"]):
    ax = fig.add_subplot(gs[0, 3*k:3*k+3])
    for stack in ["base", "PA", "RL"]:
        st = dict(STYLE[stack])
        ms = 12 if st["marker"] == "*" else 6.5
        ax.plot(TOK[model][stack], avg6(model, stack), marker=st.pop("marker"),
                markersize=ms, lw=1.8, **{kk: v for kk, v in st.items() if kk != "label"},
                label=st["label"])
    for ref_stack in ["base", "PA"]:
        ax.axhline(avg6(model, ref_stack)[3], color=STYLE[ref_stack]["color"],
                   ls=":", lw=1.0, alpha=0.65)
    for cap, xi, yi in zip(CAPS, TOK[model]["RL"], avg6(model, "RL")):
        ax.annotate(f"@{cap}", (xi, yi), textcoords="offset points",
                    xytext=(5, -11), fontsize=8, color=STYLE["RL"]["color"])
    ax.set_xlim(300, 4900)
    ax.set_title(f"Qwen3.5-{model}")
    ax.set_xlabel("Mean visual tokens per sample")
    ax.set_ylabel("Six-benchmark average")
    ax.grid(alpha=0.28, lw=0.5)
    if k == 0:
        ax.legend(loc="lower right")

# bottom: 2x6 per-bench grid
for row, model in enumerate(["4B", "9B"]):
    for col, bench in enumerate(PERBENCH):
        ax = fig.add_subplot(gs[1 + row, col])
        for stack in ["base", "PA", "RL"]:
            st = STYLE[stack]
            ms = 8 if st["marker"] == "*" else 4.5
            ax.plot(TOK[model][stack], PERBENCH[bench][model][stack],
                    color=st["color"], marker=st["marker"], ls=st["ls"],
                    markersize=ms, lw=1.3)
        if row == 0:
            ax.set_title(bench, fontsize=9.5)
        if col == 0:
            ax.set_ylabel(f"Qwen3.5-{model}\naccuracy", fontsize=9)
        if row == 1:
            ax.set_xlabel("Visual tokens", fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7.5)

out = r"C:\YH Files\submissions\ICLR2027\RL_SD_RPN\figures\token_curves_main.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
for m in ["4B", "9B"]:
    print(m, {s: avg6(m, s) for s in ["base", "PA", "RL"]})
