"""Render neuron tuning, population and latent dynamics figures from saved measurements."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

MODELS = ("RNN", "JEPA")
BLUE, ORANGE = "#0072B2", "#BB521A"


def band(ax, x, vals, color, label, ls="-", marker="o"):
    v = np.asarray(vals)
    ax.fill_between(x, v.min(0), v.max(0), color=color, alpha=.12)
    ax.plot(x, v.mean(0), ls, marker=marker, ms=4, color=color, label=label)


def groups(data, model, stage="trained"):
    return [r for r in data["records"] if r["model"] == model and r["stage"] == stage]


def save(fig, path):
    fig.savefig(path, dpi=170)
    plt.close(fig)


def single(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.4]})
    for i, m in enumerate(MODELS):
        row = groups(data, m)[0]
        maps = row["maps"]
        up, down = [np.array(maps[k]) for k in ("bend_up", "bend_down")]
        extent = [maps["x"][0]-.5, maps["x"][-1]+.5, maps["y"][-1]+.5, maps["y"][0]-.5]
        for j, (im, desc) in enumerate([((up[0]+down[0])/2, "Mean activation"),
                                        (down[1]-up[1], "Down minus up")]):
            ax = axes[i, j]
            lim = max(abs(im.min()), abs(im.max()), 1e-10)
            handle = ax.imshow(im, extent=extent, cmap="RdBu_r" if j else "viridis",
                               norm=TwoSlopeNorm(0, -lim, lim) if j else None)
            fig.colorbar(handle, ax=ax, shrink=.75, label=desc)
            ax.set(title=f"{m} unit {maps['units'][j]} · "+("start" if not j else "bend"),
                   xlabel="Chord midpoint x (px)", ylabel="Midpoint y (px)")
        ax = axes[i, 2]
        rs = groups(data, m)
        for j, (key, col) in enumerate((("start", BLUE), ("bend", ORANGE))):
            for z, stage in enumerate(("initial", "trained")):
                vals = [r["examples"][key]["test_r2"] for r in groups(data, m, stage)]
                pos = j+(-.12 if not z else .12)
                ax.vlines(pos, min(vals), max(vals), color=col)
                ax.scatter(pos, np.mean(vals), color=col if z else "white", edgecolor=col,
                           marker="o", s=56)
        ax.set(xticks=[0, 1], xticklabels=["Start x,y", "Cue offset x,y"],
               xlim=(-.5, 1.5), ylabel="Held-out R²", title=f"Best fit-ranked unit · {len(rs)} seeds")
        ax.axhline(0, color=".6", ls=":", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    lo = min(ax.get_ylim()[0] for ax in axes[:, 2])
    hi = max(ax.get_ylim()[1] for ax in axes[:, 2])
    for ax in axes[:, 2]:
        ax.set_ylim(lo, hi)
    fig.suptitle("Individual units respond to location and path bend\n"
                 "Seed 0 maps: translated horizontal paths · held-out crosses · open = untrained, filled = trained",
                 fontsize=12)
    save(fig, path)


def populations(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.1), layout="constrained", sharey="col")
    for i, m in enumerate(MODELS):
        rs, init = groups(data, m), groups(data, m, "initial")
        sizes = [v["k"] for v in rs[0]["populations"]["start"]]
        x = np.arange(len(sizes))
        for ax, key in zip(axes[i, :2], ("start", "bend")):
            for src, metric, col, label, ls in [(rs, "r2", BLUE, "Fit-ranked", "-"),
                                                (rs, "random_r2", ORANGE, "Random units", "--"),
                                                (init, "r2", "#62656E", "Untrained, ranked", ":")]:
                band(ax, x, [[v[metric] for v in r["populations"][key]] for r in src],
                     col, label, ls)
            ax.set(ylabel=f"{m} · held-out R²", title="Decode "+("start position" if key == "start"
                                                                 else "cue offset"))
            ax.axhline(0, lw=.7, color=".6")
        ax = axes[i, 2]
        for metric, col, label, ls in (("copied_px", BLUE, "Copy ranked units", "-"),
                                       ("removed_px", "#76579B", "Block ranked units", "-."),
                                       ("random_px", ORANGE, "Copy random units", "--")):
            band(ax, x, [[v[metric] for v in r["swaps"]["sizes"]] for r in rs], col, label, ls)
        ax.set(ylabel="Output bend (px)", title="Change the prediction")
        for ax in axes[i]:
            ax.set(xticks=x, xticklabels=[str(v) for v in sizes], xlabel="Number of units")
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.suptitle("Reading a population is different from making it move the prediction\n"
                 "Readouts fit on squares/circles, tested on new crosses · bands span seeds", fontsize=13)
    save(fig, path)


def dynamics(data, path):
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 6.5), layout="constrained")
    times = np.arange(1, 9)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#c6c7cb")
    for i, m in enumerate(MODELS):
        gs = [r["geometry"] for r in groups(data, m)]
        ax, line, travel, pc = axes[i]
        heat = np.mean([g["cross_time_r2"]["start"] for g in gs], axis=0)
        im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, extent=(.5, 8.5, 8.5, .5))
        ax.set(xticks=times, yticks=times, xlabel="Test frame", ylabel=f"{m} · fit frame",
               title="Does the same start readout work?")
        fig.colorbar(im, ax=ax, shrink=.75, label="Held-out R²", extend="min")
        band(line, times, [g["pair_separation_rms"] for g in gs], BLUE, "Opposite-bend states")
        band(line, times[1:], [g["step_rms"] for g in gs], ORANGE, "Step from previous state", "--")
        line.set(title="Full-space distances", xlabel="Reveal frame", ylabel="RMS / final-state spread",
                 xticks=times, ylim=(0, None))
        line.legend(frameon=False, fontsize=8)
        line.spines[["top", "right"]].set_visible(False)
        rs = groups(data, m)
        for key, colour, label, ls in (("true", "#555960", "True position", ":"),
                                       ("time_blind", BLUE, "One fixed readout", "-"),
                                       ("frozen_cue_plus_clock", ORANGE, "Cue state + clock", "--")):
            band(travel, [5, 6, 7], [r["blank_motion"][key]["projected_travel"] for r in rs],
                 colour, label, ls)
        travel.set(title="Does decoded position travel?", xlabel="Blank frame",
                   ylabel="Fraction of true 5→7 travel", xticks=[5, 6, 7])
        travel.legend(frameon=False, fontsize=8)
        travel.spines[["top", "right"]].set_visible(False)
        g = gs[0]
        pts = np.array(g["example_pcs"])
        for j, (colour, label) in enumerate(((BLUE, "One cue"), (ORANGE, "Opposite cue"))):
            a = pts[j]
            pc.plot(*a.T, color=colour, lw=1.7, marker=".", ms=4, label=label)
            for t, mk in ((0, "o"), (3, "^"), (7, "s")):
                pc.scatter(*a[t], marker=mk, s=44, edgecolor=colour, facecolor="white", zorder=3)
        pc.set(xlabel="PC1", ylabel="PC2", aspect="equal",
               title=f"One paired path · PCs cover {g['pc12_variance_fraction']:.0%}")
        pc.spines[["top", "right"]].set_visible(False)
    lim = max(a.get_ylim()[1] for a in axes[:, 1])
    for a in axes[:, 1]:
        a.set_ylim(0, lim)
    fig.suptitle("How the internal state changes as flashes arrive\n"
                 "A at frame 1 (circle), cue at 4 (triangle), B at 8 (square) · "
                 "RNN recurrent state vs JEPA context recomputed from flashes revealed so far", fontsize=13)
    save(fig, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/motion_unit_dynamics.json"))
    p.add_argument("--dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.results.read_text())
    for fn, suffix in ((single, "single_units"), (populations, "populations"), (dynamics, "latent_dynamics")):
        fn(data, args.dir/f"motion_{suffix}.png")
