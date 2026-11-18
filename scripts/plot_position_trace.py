"""Regenerate the causal-tracing figure from the saved measurements."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), layout="constrained", sharey="col")
    colours = {"A": "#0072B2", "B": "#C65416"}

    def series(ax, x, seed_values, colour, ls="-", marker="o"):
        v = np.array(seed_values)
        ax.fill_between(x, v.min(0), v.max(0), color=colour, alpha=.1)
        ax.plot(x, v.mean(0), ls, marker=marker, ms=4, color=colour)

    for i, model in enumerate(("RNN", "JEPA")):
        a, b, c = axes[i]
        sites = ("flash", "gap", "final") if i == 0 else ("slot", "mix", "final")
        for end, col in colours.items():
            rows = [v["rows"] for v in data["records"] if v["model"] == model and v["end"] == end]
            series(a, range(2, 8), [r["full"]["by_hidden_frame_px"] for r in rows], col)
            ts = [k for k in sites if k+"/position" in rows[0]]
            xs = np.array([sites.index(k) for k in ts]) + (-.04 if end == "A" else .04)

            def fractions(keys):
                return [[100*np.mean([r[k]["shift_px"] for k in kk])/r["full"]["shift_px"]
                         for kk in keys] for r in rows]
            names = [[k+"/position"] for k in ts]
            null = [[f"{k}/random{n}" for n in range(data["random_controls"])] for k in ts]
            series(b, xs, fractions(names), col)
            series(b, xs, fractions(null), col, ls=":", marker="x")
            if "route/all" in rows[0]:
                budgets = ("64", "1pct", "5pct")
                names = [["route/ranked_"+v] for v in budgets]
                null = [[f"route/null{n}_{v}" for n in range(data["random_controls"])]
                        for v in budgets]
                series(c, np.arange(3), fractions(names), col)
                series(c, np.arange(3), fractions(null), col, ls=":", marker="x")
        a.set(xticks=range(2, 8), ylabel=f"{model}\nShift along edit (px)", xlabel="Hidden frame")
        b.set(xticks=range(3), xticklabels=(["After A", "Before B", "Final"] if i == 0
                                            else ["Flash slot", "First mix", "Final"]),
              ylabel="Intact position shift (%)")
        c.set(xticks=range(3), xticklabels=["64 edges", "1%", "5%"], xlabel="Selected learned weights",
              ylabel="Intact position shift (%)")
        for ax in axes[i]:
            ax.axhline(0, lw=.7, color=".5")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.16)
        for ax in (b, c):
            ax.axhline(100, lw=.8, ls="--", color=".55")
    for ax, title in zip(axes[0], ("1   Move one flash by 4 px", "2   Swap two position directions",
                                  "3   Swap selected connections")):
        ax.set_title(title, fontsize=11, loc="left")
    handles = [Line2D([0], [0], color=col, lw=2) for col in colours.values()]
    handles.append(Line2D([0], [0], color=".4", ls=":", marker="x"))
    fig.legend(handles, ["Start A moved", "End B moved", "Random, same rank or count"],
               ncols=3, loc="outside lower center", frameon=False)
    fig.suptitle("Which internal changes move the predicted position?\n"
                 f"Directions and connections selected on squares/circles · tested on "
                 f"{data['test_pairs_per_endpoint']} triangle pairs per endpoint · bands span seeds",
                 fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/position_trace.json"))
    p.add_argument("--out", type=Path, default=Path("figures/position_trace.png"))
    args = p.parse_args()
    plot(json.loads(args.results.read_text()), args.out)
