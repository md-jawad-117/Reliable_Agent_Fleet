"""Plot results/runs.csv -> results/fig_overview.png, results/fig_unsafe_by_loss.png, results/summary.md"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"baseline": ("#2a78d6", "o", "Baseline (retry + timeout)"),
          "reliable": ("#eb6834", "s", "Ours (leases + fencing + reconcile)")}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]
    for r in rows:
        for k in ("orders_ok", "duplicate_shipments", "collisions", "orders_failed", "n_faults", "seed"):
            r[k] = int(r[k])
        r["loss"], r["makespan"] = float(r["loss"]), float(r["makespan"])
        r["finished"] = r["finished"] == "True"
        r["unsafe"] = int(r["duplicate_shipments"] > 0 or r["collisions"] > 0)
    return rows


def mean_ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return v.mean(), 0.0
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def line(ax, rows, key, scale=1.0, label_end=True):
    xs = sorted({r["n_faults"] for r in rows})
    for mode, (col, mk, name) in SERIES.items():
        ms, cs = [], []
        for x in xs:
            m, c = mean_ci([r[key] * scale for r in rows if r["mode"] == mode and r["n_faults"] == x])
            ms.append(m)
            cs.append(c)
        ms, cs = np.array(ms), np.array(cs)
        ax.fill_between(xs, ms - cs, ms + cs, color=col, alpha=0.12, linewidth=0)
        ax.plot(xs, ms, color=col, linewidth=2, marker=mk, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        if label_end:
            ax.annotate("baseline" if mode == "baseline" else "ours", (xs[-1], ms[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xlabel("injected faults per run")
    ax.set_xlim(xs[0] - 0.2, xs[-1] + 0.8)


def main():
    rows = load(os.path.join("results", "runs.csv"))
    out = "results"

    # ---- overview: 4 single-axis panels
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    line(axs[0], rows, "unsafe", 100)
    axs[0].set_title("Runs with an unsafe event (%)")
    axs[0].set_ylabel("% of runs with a duplicate or collision")
    axs[0].set_ylim(-3, 103)
    line(axs[1], rows, "duplicate_shipments")
    axs[1].set_title("Duplicate shipments per run")
    axs[1].set_ylim(bottom=-0.05)
    line(axs[2], rows, "collisions")
    axs[2].set_title("Arm-arm collisions per run")
    axs[2].set_ylim(bottom=-0.03)
    line(axs[3], rows, "makespan")
    axs[3].set_title("Time to finish all orders (s)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02, "Mean over 90 runs per point (30 seeds x 3 message-loss rates); bands = 95% CI.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(os.path.join(out, "fig_overview.png"), dpi=160, bbox_inches="tight")

    # ---- unsafe % by message loss (small multiples, shared y)
    losses = sorted({r["loss"] for r in rows})
    fig, axs = plt.subplots(1, len(losses), figsize=(11, 3.6), sharey=True)
    for ax, l in zip(axs, losses):
        line(ax, [r for r in rows if r["loss"] == l], "unsafe", 100)
        ax.set_title(f"{int(l * 100)}% message loss")
        ax.set_ylim(-3, 103)
    axs[0].set_ylabel("% runs with duplicate or collision")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "fig_unsafe_by_loss.png"), dpi=160, bbox_inches="tight")

    # ---- summary table
    lines = ["| system | runs | unsafe runs | duplicate shipments | collisions | orders correct (of 9) "
             "| unfinished runs | mean time (s) |", "|---|---|---|---|---|---|---|---|"]
    for mode in SERIES:
        rs = [r for r in rows if r["mode"] == mode]
        lines.append(
            f"| {mode} | {len(rs)} | {sum(r['unsafe'] for r in rs)} ({100 * np.mean([r['unsafe'] for r in rs]):.0f}%) "
            f"| {sum(r['duplicate_shipments'] for r in rs)} | {sum(r['collisions'] for r in rs)} "
            f"| {np.mean([r['orders_ok'] for r in rs]):.2f} | {sum(not r['finished'] for r in rs)} "
            f"| {np.mean([r['makespan'] for r in rs]):.1f} |")
    lines += ["", "Faulty runs only (n_faults >= 1):", "",
              "| system | unsafe runs | mean time (s) |", "|---|---|---|"]
    for mode in SERIES:
        rs = [r for r in rows if r["mode"] == mode and r["n_faults"] > 0]
        lines.append(f"| {mode} | {sum(r['unsafe'] for r in rs)}/{len(rs)} "
                     f"| {np.mean([r['makespan'] for r in rs]):.1f} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
