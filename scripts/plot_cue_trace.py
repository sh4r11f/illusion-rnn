"""Render the fixed-endpoint cue-transfer comparison from saved results."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(data, path):
    fig, (ex, curve, patch) = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained",
                                          gridspec_kw={"width_ratios": (1, 1.2, 1.5)})
    grey = "#535D69"
    for side, images in enumerate(data["example_flashes_yx"]):
        positions = []
        for t, mask in enumerate(images):
            xy = np.array(mask)[:, ::-1]
            positions.append(xy.mean(0))
            if t != 1 and side:
                continue
            color = grey if t != 1 else ("#679179", "#96558C")[side]
            ex.scatter(*xy.T, c=color, s=6, alpha=.5, marker="s", lw=0)
            ex.annotate(("A", ("C", "C′")[side], "B")[t], positions[-1],
                        xytext=(4, -10), textcoords="offset points", weight="bold")
        ex.plot(*np.array(positions).T, color=grey, ls=":" if side else "--", alpha=.6, lw=1)
    ex.set(aspect="equal", xlim=(-.5, 31.5), ylim=(31.5, -.5), xticks=[], yticks=[],
           title="A   Only the middle flash moves", xlabel="One test pair · same A and B")
    frames = np.array(data["scored_1based"])
    cases = [("intact", "Move real cue"), ("plane/AB", "Endpoint planes together"),
             ("plane/cue_fit", "Cue-fitted plane"), ("route/A", "Start-ranked edges*"),
             ("route/cue_fit", "Cue-ranked edges")]
    for m, col, offset in (("RNN", "#B95215", -.12), ("JEPA", "#006EA8", .12)):
        rs = [r["rows"] for r in data["records"] if r["model"] == m]
        vals = np.array([r["intact"]["bend_px_by_frame"] for r in rs])[:, frames-1]
        curve.plot(frames, vals.mean(0), marker="o", color=col, label=m)
        curve.fill_between(frames, vals.min(0), vals.max(0), color=col, alpha=.15)
        if m == "RNN":
            ref = np.array(rs[0]["true"]["bend_px_by_frame"])
            curve.plot(frames, ref[frames-1], ls=":", color=grey, marker="x", label="True movie")
        for i, (k, _) in enumerate(cases):
            v = np.array([r[k]["bend_px"] for r in rs])
            patch.hlines(i+offset, v.min(), v.max(), color=col, lw=2)
            patch.scatter(v.mean(), i+offset, color=col, s=28)
            if k == "intact":
                continue
            null_key = ("plane/null4_" if k == "plane/AB" else "plane/null2_"
                        if k.startswith("plane/") else "route/null_")
            null = np.mean([r[key]["bend_px"] for r in rs for key in r if key.startswith(null_key)])
            patch.scatter(null, i+offset, color=col, s=18, marker="x")
    curve.axvline(4, color=grey, lw=.8, alpha=.6)
    curve.set(title="B   Do the hidden positions bend?", xticks=range(1, 9), xlabel="Frame (4 = cue)",
              ylabel="Bend toward moved cue (px)", ylim=(0, None))
    curve.legend(frameon=False, fontsize=8)
    patch.set(yticks=range(len(cases)), yticklabels=[s for _, s in cases],
              xlabel="Mean bend across the five hidden frames (px)",
              title="C   How much do internal swaps reproduce?")
    patch.invert_yaxis()
    patch.axvline(0, color=grey, lw=.6)
    for ax in (curve, patch):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax == curve else "x", alpha=.18)
    fig.suptitle("A moved cue bends the reconstruction, but the endpoint circuits explain only part\n"
                 f"{data['test_pairs']} shared triangle pairs · endpoint drift subtracted · "
                 "ranges span model seeds · × in panel C = matched random swaps", fontsize=13)
    patch.text(0, -.22, "*RNN: same weights. JEPA: corresponding cue-slot weights.",
               transform=patch.transAxes, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/cue_trace.json"))
    p.add_argument("--out", type=Path, default=Path("figures/cue_trace.png"))
    args = p.parse_args()
    plot(json.loads(args.results.read_text()), args.out)
