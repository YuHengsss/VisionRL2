"""Accuracy vs latency scatter (Efficiency Analysis figure).

All points self-measured on the same CityU GPU, training-aligned protocol.
Acc = 6-bench mean (V*, MME-Lite, HR4k, HR8k, ZB, InfoVQA) computed from
the per-bench cap grids in draft_design.md (fin4b/fin9b RL rows, v4mix PA,
HF base) and vopd_curve/results.md (VOPD released weights).
Lat = latsweep AVG5 estimator (first-100 minus warmup-3); the base-4B@576
V* cell uses its median (0.14) instead of the smoke-leg-inflated mean.
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CAPS = ["576", "1024", "2048", "4096"]

def avg6(rows):
    return [round(sum(c[i] for c in rows) / len(rows), 2) for i in range(4)]

# per-bench rows: V*, MME, HR4k, HR8k, ZB, Info (caps 576/1024/2048/4096)
acc = {
    ("4b", "base"): avg6([[65.97,74.35,78.01,82.20],[40.96,42.05,45.49,47.58],
                          [63.50,68.50,72.50,75.62],[56.37,60.75,65.62,67.62],
                          [40.47,45.44,49.94,52.31],[69.82,77.89,79.82,81.62]]),
    ("4b", "pa"):   avg6([[82.72,81.68,85.86,85.86],[48.93,50.86,52.11,52.37],
                          [71.12,72.88,74.50,77.00],[63.25,66.87,69.75,70.12],
                          [55.62,55.62,57.99,58.22],[78.10,81.60,82.91,83.28]]),
    ("4b", "rl"):   avg6([[85.34,87.96,89.01,91.62],[50.96,52.11,53.88,54.19],
                          [77.38,77.88,78.63,78.62],[70.87,76.25,76.25,75.50],
                          [61.78,66.15,67.57,66.27],[80.45,84.51,84.91,85.32]]),
    ("4b", "vopd"): [57.49, 62.23, 65.73, 69.06],
    ("9b", "base"): avg6([[69.11,76.96,84.82,86.91],[45.34,52.37,50.50,56.18],
                          [64.12,67.37,71.37,76.38],[56.87,61.12,63.00,69.12],
                          [46.86,47.69,52.19,53.02],[74.77,81.47,83.66,84.09]]),
    ("9b", "pa"):   avg6([[83.25,90.05,92.15,91.10],[56.44,56.59,58.31,57.48],
                          [76.00,76.50,79.00,79.50],[70.25,72.25,74.37,79.00],
                          [59.29,62.13,63.43,63.31],[82.72,85.24,85.72,86.16]]),
    ("9b", "rl"):   avg6([[89.53,90.58,94.76,96.34],[56.75,57.63,59.20,59.30],
                          [78.50,79.38,81.37,82.00],[73.38,75.38,79.50,80.88],
                          [62.72,65.21,67.10,66.63],[84.08,86.02,87.12,87.12]]),
    ("9b", "vopd"): [63.15, 67.71, 72.53, 75.72],
}

# Per-sample MEDIAN latency, AVG over the 5 task runs (2026-08-06 fix:
# means were dominated by rare 8-10s allocator/warmup outliers, biasing
# series unevenly; median verified DP-invariant to +-0.5%).
# base-4b@576 V* cell = sibling-task median 0.162 (smoke leg polluted
# beyond trimming; single-leg re-run queued).
lat = {
    ("4b", "base"): [0.162, 0.231, 0.400, 0.634],
    ("4b", "pa"):   [0.291, 0.420, 0.612, 0.879],
    ("4b", "rl"):   [0.274, 0.400, 0.641, 0.976],
    ("4b", "vopd"): [0.266, 0.343, 0.495, 0.734],
    ("9b", "base"): [0.248, 0.363, 0.617, 0.940],
    ("9b", "pa"):   [0.414, 0.596, 0.974, 1.477],
    ("9b", "rl"):   [0.406, 0.591, 1.011, 1.547],
    ("9b", "vopd"): [0.211, 0.325, 0.598, 0.992],
}

STYLE = {
    "base": dict(color="#8a8a8a", marker="s", label="Base (uniform scaling)"),
    "vopd": dict(color="#5e9c76", marker="^", label="Vision-OPD (released)"),
    "pa":   dict(color="#d99143", marker="o", label="SD-RPN"),
    "rl":   dict(color="#2e6fa3", marker="*", label="Ours"),
}
CAP_LBL = {"576": "576", "1024": "1K", "2048": "2K", "4096": "4K"}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
for ax, scale, title in zip(axes, ["4b", "9b"], ["Qwen3.5-4B", "Qwen3.5-9B"]):
    for stack in ["base", "vopd", "pa", "rl"]:
        st = STYLE[stack]
        xs, ys = lat[(scale, stack)], acc[(scale, stack)]
        ms = 13 if st["marker"] == "*" else 7
        ax.plot(xs, ys, color=st["color"], lw=1.4, alpha=0.85, zorder=2)
        ax.scatter(xs, ys, s=ms**2, color=st["color"], marker=st["marker"],
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label=st["label"])
        for x, y, c in zip(xs, ys, CAPS):
            dy = 9 if stack == "rl" else -11
            ax.annotate(CAP_LBL[c], (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8,
                        color=st["color"], zorder=4)
    # reference: SD-RPN@4096 level
    ref = acc[(scale, "pa")][3]
    ax.axhline(ref, color="#d99143", ls=":", lw=1.0, alpha=0.6, zorder=1)
    ax.set_title(title)
    ax.set_xlabel("Latency (s / sample)")
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel("Six-benchmark average accuracy")
handles, labels = axes[0].get_legend_handles_labels()
by = dict(zip(labels, handles))
order = ["Ours", "SD-RPN", "Vision-OPD (released)", "Base (uniform scaling)"]
axes[1].legend([by[o] for o in order], order, loc="lower right", framealpha=0.9)
fig.tight_layout(w_pad=2.0)
out = r"C:\YH Files\submissions\ICLR2027\RL_SD_RPN\figures\acc_latency.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
for k in sorted(acc):
    print(k, "acc", acc[k], "lat", lat[k])
