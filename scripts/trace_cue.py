"""Do endpoint position circuits transfer to a bend cued *between* fixed endpoints?"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from illusion_rnn import trajectories as tr
from illusion_rnn.analysis import fit_ridge
from illusion_rnn.jepa import MotionJEPA
from scripts import trace_position as ep
from scripts.run_trajectory_experiment import CUE, SCORE, load

SHOWN = (0, CUE, tr.FRAMES-1)
FIT_SEED, TEST_SEED = 130401, 130402


@dataclass
class CuePairs:
    original: torch.Tensor
    moved: torch.Tensor
    axis: torch.Tensor
    truth: torch.Tensor  # rendered centre displacement; never passed to the model


def cue_pairs(n, seed, shapes):
    rng = np.random.default_rng(seed)
    seen, old, new, changes, axes = set(), [], [], [], []
    while len(old) < n:
        chords = tr.sample_chords(rng, 64)
        chords.shapes = rng.choice(shapes, len(chords.shapes))
        left, right = (tr.render(chords, "steady", p) for p in ("arc_left", "arc_right"))
        for i in range(len(chords.shapes)):
            a, b = (left, right) if len(old) % 2 == 0 else (right, left)
            key = np.packbits(a.frames[i, [0, -1]].astype(bool)).tobytes()
            if key in seen:
                continue
            seen.add(key)
            old.append(a.frames[i, list(SHOWN)])
            new.append(b.frames[i, list(SHOWN)])
            delta = b.centres[i]-a.centres[i]
            changes.append(delta)
            dx, dy = a.centres[i, -1]-a.centres[i, 0]
            normal = np.array([-dy, dx])
            axes.append(normal/np.linalg.norm(normal)*np.sign(normal @ delta[CUE]))
            if len(old) == n:
                break
    return CuePairs(*(torch.from_numpy(np.stack(x)).float() for x in (old, new, axes, changes)))


def chord_residual(shift, first, last):
    t = torch.arange(tr.FRAMES, dtype=shift.dtype)
    w = (t-first)/(last-first)
    return shift - ((1-w)*shift[:, first:first+1] + w*shift[:, last:last+1])


def bend_weights():
    # Weights of the endpoint-adjusted mean hidden bend, for ranking on cue-fit pairs only.
    return chord_residual(torch.eye(tr.FRAMES), 0, tr.FRAMES-1)[:, SCORE].mean(1)


def per_pair(delta, axis):
    side = (delta*axis[:, None]).sum(-1)
    residual = chord_residual(side, 0, tr.FRAMES-1)
    hidden = chord_residual(side, SCORE[0], SCORE[-1])
    return {"bend_px": residual[:, SCORE].mean(1), "shift_px_by_frame": side,
            "bend_px_by_frame": residual, "hidden_only_bend_px": hidden[:, SCORE[1:-1]].mean(1),
            "endpoint_motion_px": delta[:, (0, -1)].norm(dim=-1)}


def cue_states(model, flashes, time=CUE):
    return ep.flash_states(model, flashes, SHOWN, time)


def plane_set(model, fits, endpoint_fit, cal, previous, seed):
    a, shift, d, _ = endpoint_fit
    jepa = isinstance(model, MotionJEPA)
    planes, orders = {}, {}
    for e, k in enumerate("AB"):
        u, v = (ep.states(model, images, e) for images in (a, ep.donor(a, shift, e)))
        planes[k] = ep.basis(v["final"]-u["final"], d)[0]
        if jepa or e == 0:
            order = ep.route_ranking(model, u, v, d, e, cal)
            width = ep.route_inputs(model, u, v, e)[2].size(1)
            ids = {(int(idx % width), int(idx // width)) for idx in order[:64]}
            if ids != set(map(tuple, previous[k]["ranked_edges_source_target"])):
                raise ValueError(f"endpoint circuit mismatch at seed {seed}, {k}; check checkpoints")
            orders[k] = order
    planes["AB"] = torch.linalg.svd(torch.cat([planes[k] for k in "AB"], dim=1),
                                    full_matrices=False).U
    u, v = (cue_states(model, images) for images in (fits.original, fits.moved))
    step = fits.truth[:, CUE]
    planes["cue_fit"] = ep.basis(v["final"]-u["final"], step)[0]
    orders["cue_fit"] = ep.route_ranking(model, u, v, fits.axis*ep.SHIFT, 0, cal,
                                         slot_time=CUE, time_weights=bend_weights())
    rng = torch.Generator().manual_seed(130410+seed)
    for i in range(ep.NULLS):
        q = torch.linalg.qr(torch.randn(planes["AB"].shape, generator=rng)).Q
        planes[f"null2_{i}"], planes[f"null4_{i}"] = q[:, :2], q
        orders[f"null_{i}"] = torch.randperm(len(orders["cue_fit"]), generator=rng)
    w = ep.route_inputs(model, u, v, 0, slot_time=CUE)[2]
    weights = {k: ep.mask_weights(w, idx, round(.05*w.numel())) for k, idx in orders.items()}
    return planes, weights


def score(model, test, planes, weights, cal):
    measures = {}
    for i in range(0, len(test.axis), 64):
        sl = slice(i, i+64)
        x, y = test.original[sl], test.moved[sl]
        u, v = (cue_states(model, imgs) for imgs in (x, y))
        xy = ep.output(model, u["final"], cal, all_frames=True)[1]

        def record(key, delta):
            for metric, vals in per_pair(delta, test.axis[sl]).items():
                measures.setdefault(key, {}).setdefault(metric, []).append(vals.numpy())

        def edited(key, h):
            record(key, ep.output(model, h, cal, all_frames=True)[1]-xy)

        record("true", test.truth[sl])
        edited("intact", v["final"])
        for k, q in planes.items():
            piece = ((v["final"]-u["final"]) @ q) @ q.T
            edited(f"plane/{k}", u["final"]+piece)
            if not k.startswith("null"):
                edited(f"block_plane/{k}", v["final"]-piece)
        pre0, dx, w = ep.route_inputs(model, u, v, 0, slot_time=CUE)
        pre1 = ep.route_inputs(model, v, u, 0, slot_time=CUE)[0]
        for k, mask in {**weights, "all": w}.items():
            inc = F.linear(dx, mask)
            edited(f"route/{k}", ep.route_finish(model, u, pre0+inc))
            if not k.startswith("null"):
                edited(f"block_route/{k}", ep.route_finish(model, v, pre1-inc))
        if isinstance(model, MotionJEPA):
            # The old absolute A/B edges are distinct columns from the cue-slot weights.
            # Copy even the *entire* original endpoint slots; their contributions cannot change.
            slots = u["_slots"].clone()
            for t in (0, tr.FRAMES-1):
                cols = ep.time_slice(model, t)
                slots[:, cols] = v["_slots"][:, cols]
            edited("literal_endpoint_slots", model.context[1:](slots))
    data = {k: {m: np.concatenate(buf) for m, buf in vals.items()} for k, vals in measures.items()}
    rows = {k: {**{m: a.mean(0, dtype=np.float64).tolist() for m, a in vals.items()},
                "fraction_bent_toward_cue": float((vals["bend_px"] > 0).mean())}
            for k, vals in data.items()}
    return rows, {k: vals["bend_px"] for k, vals in data.items()}


def intervals(per_seed):
    """Paired movie bootstrap conditional on fixed seeds; compare to matched random edits."""
    rng = np.random.default_rng(130420)
    curves = {k: np.mean([r[k] for r in per_seed], axis=0) for k in per_seed[0]}
    n = len(curves["intact"])
    idx = rng.integers(n, size=(2000, n))
    out = {}
    for k, a in curves.items():
        if k.split("/")[-1].startswith("null") or k.startswith("block_"):
            continue
        if k.startswith(("plane/", "route/")) and k != "route/all":
            fam, which = k.split("/")
            prefix = ("null4_" if which == "AB" else "null2_") if fam == "plane" else "null_"
            a = a - np.mean([curves[f"{fam}/{prefix}{i}"] for i in range(ep.NULLS)], axis=0)
            k += "_minus_null"
        dist = a[idx].mean(1)
        out[k] = {"mean_px": float(a.mean()), "ci95_px": np.quantile(dist, [.025, .975]).tolist()}
    return out


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    prior = json.loads(args.prior.read_text())
    basefit = ep.pairs(prior["fit_pairs_per_endpoint"], prior["fit_seed"], prior["fit_shapes"])
    fit = cue_pairs(args.fit, FIT_SEED, ("square", "circle"))
    test = cue_pairs(args.test, TEST_SEED, ("triangle",))
    records, samples = [], {"RNN": [], "JEPA": []}
    for seed in sorted({r["seed"] for r in prior["records"]}):
        rnn = load(torch.load(args.rnn/f"held-out_seed{seed}.pt", map_location="cpu", weights_only=True))
        jepa = MotionJEPA()
        jepa.load_state_dict(torch.load(args.jepa/f"jepa_seed{seed}.pt", map_location="cpu",
                                        weights_only=True)["model"])
        cal = fit_ridge(jepa.target(basefit[0]).flatten(0, 1).numpy(), basefit[3].flatten(0, 1).numpy())
        for name, model in (("RNN", rnn.eval()), ("JEPA", jepa.eval())):
            previous = {r["end"]: r for r in prior["records"] if r["seed"] == seed and r["model"] == name}
            planes, weights = plane_set(model, fit, basefit, cal, previous, seed)
            rows, values = score(model, test, planes, weights, cal)
            cue_selected = weights["cue_fit"] != 0
            overlap = {k: float(((w != 0) & cue_selected).sum()/cue_selected.sum())
                       for k, w in weights.items() if k in ("A", "B")}
            records.append({"seed": seed, "model": name, "rows": rows,
                            "fraction_of_cue_edges_also_endpoint_ranked": overlap})
            samples[name].append(values)
    return {"fit_pairs": args.fit, "test_pairs": args.test, "fit_seed": FIT_SEED, "test_seed": TEST_SEED,
            "fit_shapes": ["square", "circle"], "test_shape": "triangle", "shown_1based": [t+1 for t in SHOWN],
            "scored_1based": [t+1 for t in SCORE], "records": records,
            "example_flashes_yx": [[torch.nonzero(im).tolist() for im in pair[0]]
                                   for pair in (test.original, test.moved)],
            "paired_bend": {m: intervals(per_seed) for m, per_seed in samples.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rnn", type=Path, required=True)
    p.add_argument("--jepa", type=Path, required=True)
    p.add_argument("--prior", type=Path, default=Path("results/position_trace.json"))
    p.add_argument("--fit", type=int, default=1024)
    p.add_argument("--test", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("results/cue_trace.json"))
    args = p.parse_args()
    if min(args.fit, args.test) < 2:
        p.error("need at least two pairs in each split")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(run(args), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
