"""Supplement figure: per-benchmark accuracy-vs-token curves (2x6 grid,
row 1 = 4B, row 2 = 9B), split out of the former token_curves_main.pdf.
Style matched to the main-text tradeoff figure (make_tradeoff_main.py):
same palette/markers, solid lines, white-edged markers. Series: base /
SD-RPN / ours (VOPD per-bench rows live in vopd_curve/results.md but the
supplement mirrors the original main-text grid).
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Per-bench mean visual tokens (src+crop), recomputed 2026-08-17 from the
# same runs as the accuracy cells (base == VOPD verified identical;
# PA-9B@576 from latsweep first-100 legs). Caps 576/1024/2048/4096.
PERBENCH_TOK = {
    "V* Bench": {
        "4B": {"base": [550, 989, 2023, 3328], "PA": [659, 1152, 2217, 3545], "RL": [831, 1371, 2529, 3822]},
        "9B": {"base": [550, 989, 2023, 3328], "PA": [663, 1184, 2232, 3528], "RL": [857, 1389, 2529, 3844]},
    },
    "MME-RW-Lite": {
        "4B": {"base": [543, 976, 1785, 2707], "PA": [652, 1135, 1954, 2784], "RL": [830, 1391, 2356, 3244]},
        "9B": {"base": [543, 976, 1785, 2707], "PA": [697, 1176, 2041, 2922], "RL": [859, 1429, 2418, 3316]},
    },
    "HR-Bench 4K": {
        "4B": {"base": [559, 998, 2004, 4040], "PA": [770, 1348, 2541, 4637], "RL": [883, 1541, 2918, 4928]},
        "9B": {"base": [559, 998, 2004, 4040], "PA": [819, 1451, 2728, 4594], "RL": [891, 1554, 2985, 5014]},
    },
    "HR-Bench 8K": {
        "4B": {"base": [559, 998, 2004, 4040], "PA": [770, 1348, 2541, 4637], "RL": [883, 1541, 2918, 4928]},
        "9B": {"base": [559, 998, 2004, 4040], "PA": [819, 1451, 2728, 4594], "RL": [891, 1554, 2985, 5014]},
    },
    "ZoomBench": {
        "4B": {"base": [551, 993, 2018, 3347], "PA": [669, 1160, 2203, 3477], "RL": [849, 1404, 2571, 3865]},
        "9B": {"base": [551, 993, 2018, 3347], "PA": [693, 1192, 2237, 3532], "RL": [877, 1422, 2565, 3879]},
    },
    "InfoVQA": {
        "4B": {"base": [539, 914, 1574, 2329], "PA": [679, 1116, 1814, 2578], "RL": [842, 1360, 2208, 2964]},
        "9B": {"base": [539, 914, 1574, 2329], "PA": [687, 1095, 1777, 2503], "RL": [845, 1355, 2201, 2950]},
    },
}

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

STYLE = {
    "base": dict(color="#8a8a8a", marker="s", label="Base (uniform scaling)"),
    "PA":   dict(color="#d99143", marker="o", label="SD-RPN"),
    "RL":   dict(color="#2e6fa3", marker="*", label="Ours"),
}

fig, axes = plt.subplots(2, 6, figsize=(12.5, 4.6))
for row, model in enumerate(["4B", "9B"]):
    for col, bench in enumerate(PERBENCH):
        ax = axes[row][col]
        for stack in ["base", "PA", "RL"]:
            st = STYLE[stack]
            ms = 9 if st["marker"] == "*" else 5
            xs = PERBENCH_TOK[bench][model][stack]
            ax.plot(xs, PERBENCH[bench][model][stack],
                    color=st["color"], lw=1.3, alpha=0.85, zorder=2)
            ax.scatter(xs, PERBENCH[bench][model][stack],
                       s=ms**2, color=st["color"], marker=st["marker"],
                       edgecolors="white", linewidths=0.5, zorder=3,
                       label=st["label"])
        if row == 0:
            ax.set_title(bench, fontsize=9.5)
        if col == 0:
            ax.set_ylabel(f"Qwen3.5-{model}\naccuracy", fontsize=9)
        if row == 1:
            ax.set_xlabel("Visual tokens", fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7.5)
handles, labels = axes[0][0].get_legend_handles_labels()
by = dict(zip(labels, handles))
order = ["Ours", "SD-RPN", "Base (uniform scaling)"]
fig.legend([by[o] for o in order], order, loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, -0.035))
fig.tight_layout(w_pad=1.0, h_pad=1.4)
for ext in ["pdf", "png"]:
    out = rf"C:\YH Files\submissions\ICLR2027\RL_SD_RPN\figures\token_curves_perbench.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    print("saved", out)
