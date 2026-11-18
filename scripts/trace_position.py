"""Causal traces of endpoint position through the frozen two-flash motion models."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from illusion_rnn import apparent as ap
from illusion_rnn.analysis import fit_ridge
from illusion_rnn.jepa import MotionJEPA
from scripts.run_trajectory_experiment import load

T, SIDE, SHIFT, K, NULLS = 8, 32, 4, 8, 4


def pairs(n, seed, shapes):
    """Same base for A-only and B-only interventions; balance ±x, ±y."""
    rng = np.random.default_rng(seed)
    ends, moved, steps, centres, seen = [], [], [], [], set()
    directions = SHIFT*np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    while len(ends) < n:
        batch = ap.sample_motion(rng, 64, shapes=shapes, p_static=0)
        for movie, xy in zip(batch.frames, batch.centres):
            d = directions[len(ends) % 4]
            images = movie[[0, -1]].astype(bool)
            c = xy[[0, -1]]
            # No wrap, clipping, or movement lengths outside the original range.
            _, yy, xx = np.nonzero(images)
            if not ((0 <= xx+d[0]) & (xx+d[0] < SIDE) &
                    (0 <= yy+d[1]) & (yy+d[1] < SIDE)).all():
                continue
            lengths = np.linalg.norm(c[1]-c[0] + np.array([-d, d]), axis=-1)
            if (lengths < 8).any() or (lengths > 24).any():
                continue
            shifted = np.roll(images, tuple(d[::-1]), axis=(1, 2))
            keys = {np.packbits(v).tobytes() for v in
                    (images, [shifted[0], images[1]], [images[0], shifted[1]])}
            if keys & seen:
                continue
            seen.update(keys)
            ends.append(images)
            moved.append(shifted)
            steps.append(d)
            centres.append(c)
            if len(ends) == n:
                break
    return tuple(torch.from_numpy(np.stack(v)).float() for v in (ends, moved, steps, centres))


def donor(base, moved, end):
    out = base.clone()
    out[:, end] = moved[:, end]
    return out


def input_movie(ends):
    frames = ends.new_zeros(len(ends), T, SIDE, SIDE)
    frames[:, (0, -1)] = ends
    return frames


def states(model, ends, end):
    """Only actual endpoint images are accepted; hidden target images do not exist here."""
    return flash_states(model, ends, (0, T-1), (0, T-1)[end])


def flash_states(model, images, times, slot_time):
    """Construct the movie from visible flashes only. Times include the final frame."""
    x = images.new_zeros(len(images), T, SIDE, SIDE)
    x[:, times] = images
    if isinstance(model, MotionJEPA):
        mask = torch.zeros(len(x), T, dtype=torch.bool)
        mask[:, times] = True
        enc = model.encoder(x).masked_fill(~mask[..., None], 0)
        slots = torch.cat((enc, mask[..., None]), dim=-1).flatten(1)
        mid = model.context[1:3](slots)
        return {"slot": enc[:, slot_time], "mix": mid, "final": model.context[3:](mid),
                "_slots": slots}
    h, _ = model.rnn(x.flatten(2).transpose(0, 1))
    return {"flash": h[0], "gap": h[-2], "final": h[-1], "_last": x[:, -1].flatten(1)}


def slot_slice(model, end):
    return time_slice(model, (0, T-1)[end])


def time_slice(model, time):
    dim = model.encoder.project[-1].normalized_shape[0]
    start = time*(dim+1)
    return slice(start, start+dim)


def resume(model, state, key, edited, end):
    if key == "final":
        return edited
    if isinstance(model, MotionJEPA):
        if key == "slot":
            slots = state["_slots"].clone()
            slots[:, slot_slice(model, end)] = edited
            edited = model.context[1:3](slots)
        return model.context[3:](edited)
    if key == "flash":
        blank = torch.zeros_like(state["_last"])
        for _ in range(T-2):
            edited = model.rnn.recurrence(blank, edited)
    return model.rnn.recurrence(state["_last"], edited)


def basis(diff, step):
    # Least-squares encoding directions in activation units. No output gradients,
    # coordinate decoder weights or test examples are used for localization.
    slopes = torch.linalg.lstsq(step.double(), diff.double()).solution.T
    q = torch.linalg.qr(slopes).Q
    return q.float(), slopes.norm(dim=1).argsort(descending=True)


def edge_state(model, base, other, end, src, dst):
    """Change only src→dst weight contributions; all other edges keep base values."""
    if isinstance(model, MotionJEPA):
        lin = model.context[1]
        dx = (other["slot"]-base["slot"])[:, src]
        change = F.linear(dx, lin.weight[:, slot_slice(model, end)][dst][:, src])
        pre = F.linear(base["_slots"], lin.weight[dst], lin.bias[dst])
        mid = base["mix"].clone()
        mid[:, dst] = model.context[2](pre+change)
        return model.context[3:](mid)
    # Isolate learned recurrent edges, NOT the direct leaky-identity path.
    rec = model.rnn
    delta = (other["gap"]-base["gap"])[:, src]
    pre = rec.input2h(base["_last"])[:, dst] + F.linear(
        base["gap"], rec.h2h.weight[dst], rec.h2h.bias[dst])
    change = F.linear(delta, rec.h2h.weight[dst][:, src])
    h = base["final"].clone()
    h[:, dst] += rec.alpha*((pre+change).relu()-pre.relu())
    return h


def output(model, h, calibration, *, all_frames=False):
    frames = slice(None) if all_frames else slice(1, -1)
    if isinstance(model, MotionJEPA):
        raw = model.predictor(h).reshape(len(h), T, -1)[:, frames]
        xy = calibration(raw.reshape(-1, raw.size(-1)).numpy()).reshape(len(h), raw.size(1), 2)
        return raw, torch.from_numpy(xy).float()
    raw = model.fc(h).reshape(len(h), T, SIDE, SIDE)[:, frames].sigmoid()
    axis = torch.arange(SIDE)
    m = raw.sum((-2, -1)).clamp_min(1e-20)
    xy = torch.stack([(raw.sum(k)*axis).sum(-1)/m for k in (-2, -1)], dim=-1)
    return raw, xy


def route_inputs(model, base, other, end, *, slot_time=None):
    if isinstance(model, MotionJEPA):
        lin = model.context[1]
        sl = slot_slice(model, end) if slot_time is None else time_slice(model, slot_time)
        return lin(base["_slots"]), other["slot"]-base["slot"], lin.weight[:, sl]
    rnn = model.rnn
    return (rnn.input2h(base["_last"]) + rnn.h2h(base["gap"]),
            other["gap"]-base["gap"], rnn.h2h.weight)


def route_finish(model, base, pre):
    if isinstance(model, MotionJEPA):
        return model.context[2:](pre)
    return (1-model.rnn.alpha)*base["gap"] + model.rnn.alpha*pre.relu()


def route_ranking(model, f0, f1, step, end, cal, *, slot_time=None, time_weights=None):
    """Rank learned edges on fit pairs by mean signed first-order effect at the midpoint."""
    pre, dx, w = route_inputs(model, f0, f1, end, slot_time=slot_time)
    mid = (pre + .5*F.linear(dx, w)).detach()
    if isinstance(model, MotionJEPA):
        dim = model.encoder.project[-1].normalized_shape[0]
        # Jacobian of the fixed affine single-image calibration; no test labels.
        jac = torch.tensor(cal(np.eye(dim))-cal(np.zeros((1, dim))), dtype=w.dtype)
    scores = torch.zeros_like(w)
    for i in range(0, len(step), 64):
        sl = slice(i, i+64)
        with torch.enable_grad():
            p = mid[sl].detach().requires_grad_()
            base = {} if isinstance(model, MotionJEPA) else {"gap": f0["gap"][sl]}
            h = route_finish(model, base, p)
            if isinstance(model, MotionJEPA):
                z = model.predictor(h).reshape(len(p), T, -1)
                xy = (z[:, 1:-1] if time_weights is None else z) @ jac
            else:
                _, xy = output(model, h, None, all_frames=time_weights is not None)
            measure = xy.mean(1) if time_weights is None else (xy*time_weights[None, :, None]).sum(1)
            objective = (measure*step[sl]/SHIFT).sum()
            grad, = torch.autograd.grad(objective, p)
        scores += grad.T @ dx[sl]
    return (w*scores).abs().flatten().argsort(descending=True)


def mask_weights(weight, order, n):
    mask = torch.zeros_like(weight).contiguous()
    mask.view(-1)[order[:n]] = 1
    return weight*mask


def evaluate(model, fit, test, end, seed, *, control=False):
    a, shifted, step, coords = fit
    f0, f1 = (states(model, x, end) for x in (a, donor(a, shifted, end)))
    jepa = isinstance(model, MotionJEPA)
    if jepa:
        sites = ("slot", "mix", "final")
    else:
        sites = ("flash", "gap", "final") if end == 0 else ("final",)
    if control:
        sites = ("final",)
    rng = torch.Generator().manual_seed(33000+seed)
    spaces, ranks, randoms = {}, {}, {}
    for key in sites:
        q, ranks[key] = basis(f1[key]-f0[key], step)
        spaces[key] = q
        randoms[key] = [torch.linalg.qr(torch.randn(q.shape, generator=rng)).Q for _ in range(NULLS)]
    cal = None
    if jepa:
        z = model.target(a).flatten(0, 1).numpy()
        cal = fit_ridge(z, coords.flatten(0, 1).numpy())
    edge = not control and (jepa or end == 0)
    route = ("slot", "mix") if jepa else ("gap", "final")
    edges, weights, ranked_pairs = {}, {}, []
    if edge:
        edges["selected"] = tuple(ranks[k][:K] for k in route)
        for i in range(NULLS):
            edges[f"random{i}"] = tuple(torch.randperm(len(ranks[k]), generator=rng)[:K] for k in route)
        _, _, w = route_inputs(model, f0, f1, end)
        order = route_ranking(model, f0, f1, step, end, cal)
        # Store [source, receiver] unit identities for the 64-edge candidate.
        ranked_pairs = [[int(x % w.size(1)), int(x // w.size(1))] for x in order[:64]]
        weights["all"] = w
        for label, indices in [("ranked", order),
                               *((f"null{i}", torch.randperm(w.numel(), generator=rng))
                                 for i in range(NULLS))]:
            for budget, count in (("64", 64), ("1pct", round(.01*w.numel())),
                                  ("5pct", round(.05*w.numel()))):
                weights[f"{label}_{budget}"] = mask_weights(w, indices, count)
    # The same test movies and donor pairs are used by all models and controls.
    a, shifted, step, _ = test
    totals = {}
    for start in range(0, len(a), 64):
        sl = slice(start, start+64)
        u, v = (states(model, x, end) for x in (a[sl], donor(a[sl], shifted[sl], end)))
        raw0, xy0 = output(model, u["final"], cal)
        raw1, xy1 = output(model, v["final"], cal)
        axis = step[sl]/SHIFT

        def score(name, h):
            raw, xy = output(model, h, cal)
            along = ((xy-xy0)*axis[:, None]).sum(-1).sum(0)
            across = ((xy-xy0)*torch.stack((-axis[:, 1], axis[:, 0]), dim=1)[:, None]).sum(-1)
            vals = np.r_[along.numpy(), across.abs().sum().item(),
                         (raw-raw1).square().sum().item(), (raw0-raw1).square().sum().item(),
                         (xy-xy1).square().sum().item(), (xy0-xy1).square().sum().item()]
            totals[name] = totals.get(name, np.zeros_like(vals)) + vals

        score("full", v["final"])
        for key in sites:
            diff = v[key]-u[key]
            for label, q in [("position", spaces[key]),
                             *((f"random{i}", q) for i, q in enumerate(randoms[key]))]:
                piece = (diff @ q) @ q.T
                score(f"{key}/{label}", resume(model, u, key, u[key]+piece, end))
                if label == "position":
                    score(f"{key}/blocked", resume(model, v, key, v[key]-piece, end))
        for label, (src, dst) in edges.items():
            score(f"edges/{label}", edge_state(model, u, v, end, src, dst))
            if label == "selected":
                score("edges/blocked", edge_state(model, v, u, end, src, dst))
        if edge:
            pre0, dx, _ = route_inputs(model, u, v, end)
            pre1, _, _ = route_inputs(model, v, u, end)
            for label, weight in weights.items():
                inc = F.linear(dx, weight)
                score(f"route/{label}", route_finish(model, u, pre0+inc))
                if label.startswith("ranked") or label == "all":
                    score(f"block/{label}", route_finish(model, v, pre1-inc))
    rows = {}
    for key, values in totals.items():
        along = values[:T-2]/len(a)
        rows[key] = {"shift_px": float(along.mean()), "by_hidden_frame_px": along.tolist(),
                     "abs_orthogonal_px": float(values[-5]/(len(a)*(T-2))),
                     "native_recovery": float(1-np.sqrt(values[-4]/values[-3]))
                     if values[-3] > 0 else None,
                     "xy_recovery": float(1-np.sqrt(values[-2]/values[-1]))
                     if values[-1] > 0 else None}
    selected = {k: v.tolist() for k, v in zip(route, edges.get("selected", ()))}
    return {"rows": rows, "edge_units": selected, "ranked_edges_source_target": ranked_pairs}


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    fit = pairs(args.fit, 120301, ("square", "circle"))
    test = pairs(args.test, 120303, ("triangle",))
    records, untrained = [], []
    for seed in range(args.seeds):
        rnn = load(torch.load(args.rnn / f"held-out_seed{seed}.pt", weights_only=True,
                              map_location="cpu")).eval()
        jepa = MotionJEPA().eval()
        jepa.load_state_dict(torch.load(args.jepa / f"jepa_seed{seed}.pt", weights_only=True,
                                        map_location="cpu")["model"])
        for name, model in (("RNN", rnn), ("JEPA", jepa)):
            torch.manual_seed(seed)
            init = (ap.ImputationNet(SIDE**2, rnn.rnn.hidden_size, T) if name == "RNN"
                    else MotionJEPA()).eval()
            for end in range(2):
                records.append({"model": name, "seed": seed, "end": "AB"[end],
                                **evaluate(model, fit, test, end, seed)})
                untrained.append({"model": name, "seed": seed, "end": "AB"[end],
                                  **evaluate(init, fit, test, end, seed, control=True)})
        print(f"Scored seed {seed}", flush=True)
    return {"fit_pairs_per_endpoint": args.fit, "test_pairs_per_endpoint": args.test,
            "fit_shapes": ["square", "circle"], "test_shape": "triangle", "shift_px": SHIFT,
            "fit_seed": 120301, "test_seed": 120303,
            "rank": 2, "units_per_side": K, "random_controls": NULLS, "records": records,
            "untrained": untrained}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rnn", type=Path, required=True)
    p.add_argument("--jepa", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--fit", type=int, default=1024)
    p.add_argument("--test", type=int, default=384)
    p.add_argument("--out", type=Path, default=Path("results/position_trace.json"))
    args = p.parse_args()
    if min(args.fit, args.test) < 4 or args.fit % 4 or args.test % 4 or args.seeds < 1:
        p.error("fit/test counts must be positive multiples of four; need at least one seed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(run(args), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
