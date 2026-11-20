"""Plot results/agents.csv -> results/fig_agents.png, results/fig_agent_rate.png, results/agents_summary.md"""
import csv
import os

import numpy as np

from plot_results import INK2, MUTED, SERIES, SURFACE, mean_ci, plt

AGENTS = ["impatient", "forgetful", "confused", "hallucinating", "mixed"]


def load():
    rows = [r for r in csv.DictReader(open(os.path.join("results", "agents.csv"))) if not r["error"]]
    for r in rows:
        for k in ("seed", "n_faults", "orders_ok", "duplicate_shipments", "collisions", "agent_intents",
                  "intent_unsafe_redo", "refused_unsafe_redo", "intent_repeat", "intent_wrong_arm",
                  "intent_invalid", "agent_refused"):
            r[k] = int(float(r[k] or 0))
        r["rate"], r["makespan"] = float(r["rate"]), float(r["makespan"])
        r["finished"] = r["finished"] == "True"
        r["unsafe"] = int(r["duplicate_shipments"] > 0 or r["collisions"] > 0)
        r["redo_reached"] = r["intent_unsafe_redo"] - r["refused_unsafe_redo"]
    return rows


def bars(ax, rows, key, scale=1.0, fmt="{:.0f}"):
    w = 0.38
    x = np.arange(len(AGENTS))
    for i, (mode, (col, _, name)) in enumerate(SERIES.items()):
        ms, cs = zip(*[mean_ci([r[key] * scale for r in rows if r["agent"] == a and r["mode"] == mode])
                       for a in AGENTS])
        pos = x + (i - 0.5) * w
        ax.bar(pos, ms, w * 0.92, color=col, label=name, yerr=cs, error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        for p, m, c in zip(pos, ms, cs):
            ax.annotate(fmt.format(m), (p, m + c), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK2)
    ax.set_xticks(x, AGENTS)
    ax.grid(axis="x", visible=False)


def main():
    rows = load()
    A = [r for r in rows if r["study"] == "A"]
    B = [r for r in rows if r["study"] == "B"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    bars(axs[0], A, "unsafe", 100)
    axs[0].set_title("Runs with a duplicate or collision (%)")
    axs[0].set_ylim(0, 105)
    bars(axs[1], A, "redo_reached", fmt="{:.1f}")
    axs[1].set_title("Unsafe agent commands that reached a robot, per run")
    bars(axs[2], A, "makespan", fmt="{:.0f}")
    axs[2].set_title("Time to finish all orders (s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, -0.03, f"Each agent fault at rate 0.3; {len(A) // 10} runs per bar "
                         "(20 seeds x 0 or 2 robot/network faults, 5% message loss); error bars = 95% CI.",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(os.path.join("results", "fig_agents.png"), dpi=160, bbox_inches="tight")

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    rates = sorted({r["rate"] for r in B})
    for key, ax, title, scale in (("unsafe", axs[0], "Runs with a duplicate or collision (%)", 100),
                                  ("makespan", axs[1], "Time to finish all orders (s)", 1)):
        for mode, (col, mk, name) in SERIES.items():
            ms, cs = map(np.array, zip(*[mean_ci([r[key] * scale for r in B if r["mode"] == mode and r["rate"] == x])
                                         for x in rates]))
            ax.fill_between(rates, ms - cs, ms + cs, color=col, alpha=0.12, linewidth=0)
            ax.plot(rates, ms, color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.5, label=name)
            ax.annotate("baseline" if mode == "baseline" else "ours", (rates[-1], ms[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK2, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("agent error rate (mixed agent faults)")
        ax.set_xlim(-0.02, rates[-1] + 0.08)
    axs[0].set_ylim(-3, 103)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join("results", "fig_agent_rate.png"), dpi=160, bbox_inches="tight")

    lines = ["| agent | system | runs | unsafe runs | unsafe commands issued | ...reached a robot "
             "| orders correct (of 9) | unfinished | mean time (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for a in AGENTS:
        for mode in SERIES:
            rs = [r for r in A if r["agent"] == a and r["mode"] == mode]
            lines.append(f"| {a} | {mode} | {len(rs)} | {sum(r['unsafe'] for r in rs)} "
                         f"| {sum(r['intent_unsafe_redo'] for r in rs)} | {sum(r['redo_reached'] for r in rs)} "
                         f"| {np.mean([r['orders_ok'] for r in rs]):.2f} | {sum(not r['finished'] for r in rs)} "
                         f"| {np.mean([r['makespan'] for r in rs]):.1f} |")
    lines += ["", "Dose-response (mixed agent, 2 robot faults):", "",
              "| error rate | baseline unsafe runs | ours unsafe runs | baseline time (s) | ours time (s) |",
              "|---|---|---|---|---|"]
    for x in rates:
        b = [r for r in B if r["rate"] == x and r["mode"] == "baseline"]
        o = [r for r in B if r["rate"] == x and r["mode"] == "reliable"]
        lines.append(f"| {x:.2f} | {sum(r['unsafe'] for r in b)}/{len(b)} | {sum(r['unsafe'] for r in o)}/{len(o)} "
                     f"| {np.mean([r['makespan'] for r in b]):.1f} | {np.mean([r['makespan'] for r in o]):.1f} |")
    with open(os.path.join("results", "agents_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
