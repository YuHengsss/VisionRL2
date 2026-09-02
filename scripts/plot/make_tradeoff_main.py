"""Merged main-text efficiency figure: AVG.6 accuracy vs tokens (top row)
and vs median latency (bottom row), columns = 4B / 9B. Replaces the former
token_curves_main.pdf (avg panels) + acc_latency.pdf pair; per-bench grid
moved to the supplement (make_token_curves_perbench.py). Style follows the
old acc_latency.pdf (solid lines, white-edged markers, cap point labels).
Data: results.md AVG.6 grids + median latency table; VOPD from
vopd_curve/results.md (released weights, our protocol).
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 14, "axes.labelsize": 15.5, "axes.titlesize": 16,
    "legend.fontsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "font.weight": "normal",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CAPS = ["576", "1024", "2048", "4096"]
CAP_LBL = {"576": "576", "1024": "1K", "2048": "2K", "4096": "4K"}

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
    # VOPD-4B: MME-RW-Lite cells are unified-v1 judge-rescued (format
    # non-compliance; 2026-08-17 NCI rescue: 36.06/38.72/40.39/41.22 vs
    # protocol 32.88/35.12/36.89/37.83). AVG6 recomputed accordingly.
    ("4b", "vopd"): avg6([[70.68,78.53,82.72,88.48],[36.06,38.72,40.39,41.22],
                          [64.00,68.00,71.00,75.00],[56.00,60.00,65.00,70.00],
                          [51.24,53.85,57.63,61.89],[70.00,78.00,81.00,81.00]]),
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

# AVG.6 suite means (per-task means over V*, MME, HR4K, HR8K, ZB, Info;
# src+crop), recomputed 2026-08-17 from the SAME runs as the accuracy
# grids (pack*_fin / cc4096a2_rl / pa*_v4mix_capcurve / pa4096a2v4mix /
# vopdcurve). base == VOPD per-task means verified identical (both
# single-pass, deterministic preprocessing). PA-9B@576 = latsweep
# first-100 legs (no full-run leg exists; see results.md note).
tok = {
    ("4b", "base"): [550, 978, 1901, 3298],
    ("4b", "pa"):   [700, 1210, 2212, 3610],
    ("4b", "rl"):   [853, 1434, 2583, 3958],
    ("4b", "vopd"): [550, 978, 1901, 3298],
    ("9b", "base"): [550, 978, 1901, 3298],
    ("9b", "pa"):   [730, 1258, 2290, 3612],
    ("9b", "rl"):   [870, 1450, 2614, 4003],
    ("9b", "vopd"): [550, 978, 1901, 3298],
}

# per-sample median latency, AVG over 5 task runs (see latency_sweep/results.md).
# VOPD switched 2026-08-17 to the first-100 estimator computed from the
# existing vopdcurve jsonls (same criterion as the base/PA/RL latsweep legs).
# @4096 base/PA/RL cells switched 2026-08-18 to the rtprof stage-timed
# protocol totals (canonical for the routing table; deltas vs latsweep
# 1-5%), so figure and routing table share identical numbers at the
# common operating point.
lat = {
    # @576 cell: V* leg re-measured 2026-08-18 (_rerun tag; old smoke leg
    # was JIT-polluted and the cell borrowed the sibling median 0.162)
    ("4b", "base"): [0.155, 0.231, 0.400, 0.623],
    ("4b", "pa"):   [0.291, 0.420, 0.612, 0.857],
    ("4b", "rl"):   [0.274, 0.400, 0.641, 0.945],
    ("4b", "vopd"): [0.270, 0.349, 0.507, 0.712],
    ("9b", "base"): [0.248, 0.363, 0.617, 0.918],
    ("9b", "pa"):   [0.414, 0.596, 0.974, 1.396],
    ("9b", "rl"):   [0.406, 0.591, 1.011, 1.470],
    ("9b", "vopd"): [0.212, 0.328, 0.581, 0.902],
}

# Bold single-row style (user request 2026-08-17): thick lines, bold
# labels, @cap annotations on Ours only, base dashed gray; panels
# [4B tokens, 9B tokens, 4B latency, 9B latency]. Freed vertical space
# goes to the qualitative comparison figure.
STYLE = {
    "base": dict(color="#9a9a9a", marker="s", ls="--", lw=2.0, ms=8,
                 label="Base"),
    "vopd": dict(color="#5e9c76", marker="^", ls="-", lw=2.0, ms=8,
                 label="Vision-OPD"),
    "pa":   dict(color="#d99143", marker="o", ls="-", lw=2.4, ms=8.5,
                 label="SD-RPN"),
    "rl":   dict(color="#2e6fa3", marker="*", ls="-", lw=2.6, ms=17,
                 label="Ours"),
}

# Panels paired by model scale; the latency panel of each pair shares its
# y-axis with the token panel (drops two redundant y-tick columns ->
# larger panels). Smaller canvas => everything renders larger at textwidth.
PANELS = [("4b", tok, "Visual tokens / sample", "Qwen3.5-4B"),
          ("4b", lat, "Latency (s / sample)", "Qwen3.5-4B"),
          ("9b", tok, "Visual tokens / sample", "Qwen3.5-9B"),
          ("9b", lat, "Latency (s / sample)", "Qwen3.5-9B")]

from matplotlib.ticker import MultipleLocator

# Tight pairs: near-zero gap inside each shared-y pair, spacer column
# between the pairs; taller canvas (user: y-axis was too thin).
fig = plt.figure(figsize=(14.6, 4.5))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.14, 1, 1], wspace=0.10)
axes = [fig.add_subplot(gs[0, 0]), None, fig.add_subplot(gs[0, 3]), None]
axes[1] = fig.add_subplot(gs[0, 1], sharey=axes[0])
axes[3] = fig.add_subplot(gs[0, 4], sharey=axes[2])
for k, (scale, xdata, xlabel, title) in enumerate(PANELS):
    ax = axes[k]
    for stack in ["base", "vopd", "pa", "rl"]:
        st = STYLE[stack]
        xs, ys = xdata[(scale, stack)], acc[(scale, stack)]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=st["lw"],
                marker=st["marker"], markersize=st["ms"], zorder=3,
                label=st["label"])
    if k == 0:  # cap labels only in the first panel (collide elsewhere)
        pts = list(zip(xdata[(scale, "rl")], acc[(scale, "rl")], CAPS))
        for j, (x, y, c) in enumerate(pts):
            last = j == len(pts) - 1
            ax.annotate(f"@{c}", (x, y), textcoords="offset points",
                        xytext=(5 if last else 6, 11 if last else -14),
                        ha="right" if last else "left",
                        fontsize=12, fontweight="bold",
                        color=STYLE["rl"]["color"], zorder=4)
    for ref in ["pa", "base"]:
        ax.axhline(acc[(scale, ref)][3], color=STYLE[ref]["color"],
                   ls=":", lw=1.3, alpha=0.7, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(alpha=0.28, lw=0.6)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    if xdata is tok:
        ax.set_xticks([1000, 2000, 3000, 4000])
        ax.set_xticklabels(["1K", "2K", "3K", "4K"])
    if k in (1, 3):  # shared-y partner: drop redundant tick labels
        ax.tick_params(labelleft=False)
# explicit headroom for the @4096 label (sharey autoscale ignores
# per-axes margins set before the partner panel plots)
top4b = max(acc[("4b", "rl")])
axes[0].set_ylim(top=top4b + 3.0)
axes[0].set_ylabel("Six-benchmark average")
axes[2].set_ylabel("Six-benchmark average")
handles, labels = axes[0].get_legend_handles_labels()
by = dict(zip(labels, handles))
order = ["Ours", "SD-RPN", "Vision-OPD", "Base"]
axes[0].legend([by[o] for o in order], order, loc="lower right",
               framealpha=0.9, handlelength=2.0, borderpad=0.35,
               labelspacing=0.35, handletextpad=0.5)
# manual margins: tight_layout would override the gridspec pair spacing
fig.subplots_adjust(left=0.055, right=0.995, top=0.905, bottom=0.155)
for ext in ["pdf", "png"]:
    out = rf"C:\YH Files\submissions\ICLR2027\RL_SD_RPN\figures\tradeoff_main.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    print("saved", out)
for k in sorted(acc):
    print(k, "acc", acc[k], "tok", tok[k], "lat", lat[k])
