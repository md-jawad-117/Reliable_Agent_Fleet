"""Plot results/ablations.csv -> results/fig_ablations.png, results/ablations_summary.md"""
import csv
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ablation_experiments import SCENARIOS, VARIANTS
from plot_results import INK, MUTED, SURFACE, plt

LABEL = {
    "full": "full runtime (ours)",
    "no_fencing": "- fencing",
    "no_leases": "- leases",
    "no_camera": "- camera reconcile",
    "no_dedup": "- command dedup",
    "no_done_retry": "- DONE retry",
    "no_fencing+no_leases": "- fencing - leases",
    "no_dedup+no_leases": "- dedup - leases",
    "no_done_retry+no_camera": "- DONE retry - camera",
    "baseline": "baseline (retry + timeout)",
}
COLS = list(SCENARIOS) + ["random"]
COL_LABEL = {"zombie": "zombie arm\n(crash, reassigned)", "crash_in_air": "crash inside\nairspace",
             "partition": "partition\nmid-reach", "lost_done": "lost DONE", "dup_command": "repeated\ncommand",
             "random": "random faults\n(2-4 faults)"}
CMAP = LinearSegmentedColormap.from_list("unsafe", [SURFACE, "#f3c3ae", "#e0603a", "#a3271b"])
CMAP_H = LinearSegmentedColormap.from_list("hung", [SURFACE, "#c9d8ee", "#5b8fd1", "#1f4f91"])


def load():
    rows = [r for r in csv.DictReader(open(os.path.join("results", "ablations.csv"))) if not r["error"]]
    for r in rows:
        r["unsafe"] = int(int(r["duplicate_shipments"]) > 0 or int(r["collisions"]) > 0)
        r["hung"] = int(r["finished"] != "True")
    return rows


def grid(rows, key):
    g = np.full((len(VARIANTS), len(COLS)), np.nan)
    n = np.zeros_like(g)
    for i, v in enumerate(VARIANTS):
        for j, c in enumerate(COLS):
            rs = [r for r in rows if r["variant"] == v and r["scenario"] == c]
            if rs:
                g[i, j] = 100 * np.mean([r[key] for r in rs])
                n[i, j] = len(rs)
    return g, n


def draw(ax, g, cmap, title):
    ax.imshow(g, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if v > 55 else (INK if v > 0 else MUTED), fontweight="bold" if v > 0 else None)
    ax.set_xticks(range(len(COLS)), [COL_LABEL[c] for c in COLS], fontsize=8.5)
    ax.set_yticks(range(len(VARIANTS)), [LABEL[v] for v in VARIANTS], fontsize=9)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    # separators: full | single ablations | pairs | baseline
    for y in (0.5, 5.5, 8.5):
        ax.axhline(y, color=SURFACE, lw=3)
    ax.axvline(len(SCENARIOS) - 0.5, color=SURFACE, lw=3)
    ax.set_title(title, loc="left")


def main():
    rows = load()
    g_u, n_u = grid(rows, "unsafe")
    g_h, _ = grid(rows, "hung")
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(wspace=0.05))
    draw(axs[0], g_u, CMAP, "Runs with a duplicate shipment or collision (%)")
    draw(axs[1], g_h, CMAP_H, "Runs that never finished (%)")
    axs[1].set_yticklabels([])
    nt = int(n_u[0, 0])
    nr = int(n_u[0, -1])
    fig.text(0.5, -0.02, f"Targeted scenarios: {nt} runs per cell (network seeds, 5% loss). "
                         f"Random faults: {nr} runs per cell (2 or 4 faults x 5% or 15% loss).",
             ha="center", color=MUTED, fontsize=9)
    fig.savefig(os.path.join("results", "fig_ablations.png"), dpi=160, bbox_inches="tight")

    lines = ["| variant | " + " | ".join(COL_LABEL[c].replace("\n", " ") for c in COLS) + " |",
             "|---|" + "---|" * len(COLS)]
    for i, v in enumerate(VARIANTS):
        cells = [f"{g_u[i, j]:.0f}%" + (f" ({g_h[i, j]:.0f}% hung)" if g_h[i, j] > 0 else "")
                 for j in range(len(COLS))]
        lines.append(f"| {LABEL[v]} | " + " | ".join(cells) + " |")
    with open(os.path.join("results", "ablations_summary.md"), "w") as f:
        f.write("Unsafe runs (duplicate or collision), with never-finished runs in brackets:\n\n")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
