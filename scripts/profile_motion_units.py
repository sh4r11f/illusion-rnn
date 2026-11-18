"""Single units, population readouts and reveal-time dynamics on matched apparent-motion paths."""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap
from illusion_rnn import trajectories as tr
from illusion_rnn.analysis import fit_ridge
from illusion_rnn.jepa import MotionJEPA
from scripts import trace_cue as cue
from scripts import trace_position as ep
from scripts.run_trajectory_experiment import load

FIT, TEST, SIZES, N_RANDOM = 141001, 141002, (1, 4, 16, 64), 4


@dataclass
class Movies:
    flashes: torch.Tensor  # adjacent rows = opposite bends of one chord
    path: np.ndarray       # scoring labels, never used as network input

    @property
    def centres(self):
        return self.path[:, list(cue.SHOWN)]

    @property
    def targets(self):
        c = self.centres[:, 1]
        return {"start": self.centres[:, 0], "bend": c-np.repeat((c[::2]+c[1::2])/2, 2, axis=0)}

    @property
    def normals(self):
        chord = self.centres[::2, -1]-self.centres[::2, 0]
        n = chord[:, ::-1]*[1, -1]
        n *= np.sign((n*(self.centres[1::2, 1]-self.centres[::2, 1])).sum(1, keepdims=True))
        return torch.tensor(n / np.linalg.norm(n, axis=1, keepdims=True), dtype=torch.float32)


def movies(n, seed, shapes, seen):
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    while len(frames) < n:
        ch = tr.sample_chords(rng, 64)
        ch.shapes = rng.choice(shapes, len(ch.shapes))
        a, b = (tr.render(ch, "steady", p) for p in ("arc_left", "arc_right"))
        for i in range(len(ch.sizes)):
            # Exclude repeated endpoint centres, irrespective of appearance or width.
            key = tuple(a.centres[i, [0, -1]].flatten())
            if key in seen:
                continue
            seen.add(key)
            frames.append([v.frames[i, list(cue.SHOWN)] for v in (a, b)])
            labels.append([v.centres[i] for v in (a, b)])
            if len(frames) == n:
                break
    return Movies(torch.from_numpy(np.stack(frames).reshape(-1, 3, tr.SIDE, tr.SIDE)),
                  np.stack(labels).reshape(-1, tr.FRAMES, 2))


@torch.no_grad()
def activities(model, images):
    """RNN: actual recurrent states. JEPA: separate contexts for successively revealed prefixes."""
    hist, sites, uncued = [], [], []
    for ends in images.split(128):
        frames = ends.new_zeros(len(ends), tr.FRAMES, tr.SIDE, tr.SIDE)
        frames[:, cue.SHOWN] = ends
        if isinstance(model, MotionJEPA):
            mask = torch.zeros(frames.shape[:2], dtype=torch.bool)
            states = []
            for t in range(tr.FRAMES):
                if t in cue.SHOWN:
                    mask[:, t] = True
                    h = model(frames, mask)[0]
                states.append(h)
            hist.append(torch.stack(states, dim=1).numpy())
            sites.append(cue.cue_states(model, ends)["mix"].numpy())  # actual GELU neurons
            uncued.append(ep.states(model, ends[:, (0, -1)], 0)["mix"].numpy())
        else:
            h = model.rnn(frames.flatten(2).transpose(0, 1))[0].transpose(0, 1)
            hist.append(h.numpy())
            sites.append(h[:, -1].numpy())
            frames[:, cue.CUE] = 0
            uncued.append(model.rnn(frames.flatten(2).transpose(0, 1))[1].numpy())
    return tuple(np.concatenate(z) for z in (hist, sites, uncued))


def r2(y, pred):
    return float(1-np.square(y-pred).sum()/np.square(y-y.mean(0)).sum())


def unit_fits(x, y, test, truth):
    """All univariate affine decoders, independently; rank only by fitting-set R²."""
    mu, bias = x.mean(0, dtype=np.float64), y.mean(0, dtype=np.float64)
    z, v = x-mu, y-bias
    cov = z.T @ v
    slope = cov/(z*z).sum(0).clip(min=1e-12)[:, None]
    fit_scores = (slope*cov).sum(1)/(v*v).sum()
    pred = (test-mu)[:, :, None]*slope[None]+bias
    scores = 1-((pred-truth[:, None])**2).sum((0, 2))/((truth-truth.mean(0))**2).sum()
    return fit_scores, scores, np.argsort(-fit_scores, kind="stable")


def population(x, y, test, truth, nocue, order, rng):
    rows = []
    for k in [s for s in SIZES if s < x.shape[1]] + [x.shape[1]]:
        cols = order[:k]
        probe = fit_ridge(x[:, cols], y)
        score = r2(truth, probe(test[:, cols]))
        null = []
        if k < len(order):
            for _ in range(N_RANDOM):
                c = rng.choice(x.shape[1], k, replace=False)
                null.append(r2(truth, fit_ridge(x[:, c], y)(test[:, c])))
        rows.append({"k": k, "r2": score, "random_r2": float(np.mean(null)) if null else score,
                     "no_cue_r2": r2(truth, probe(nocue[:, cols]))})
    return rows


def geometry(hist, test, targets, truth):
    """Cross-time readouts and cue-pair separation in the full latent, plus a train-only PCA view."""
    reads = {}
    for name in targets:
        # Float32 rounding of equal predictions on paired bends must not manufacture selectivity.
        reads[name] = [[r2(truth[name], probe(test[:, s])) for s in range(tr.FRAMES)]
                       for t in range(tr.FRAMES) for probe in [fit_ridge(hist[:, t], targets[name])]]
    norm = np.sqrt(np.square(test[:, -1]-test[:, -1].mean(0)).sum(1).mean())
    speed = np.sqrt(np.square(np.diff(test, axis=1)).sum(-1).mean(0))/norm
    sep = np.sqrt(np.square(test[::2]-test[1::2]).sum(-1).mean(0))/norm
    # PCA of within-movie changes, preventing static identity/location offsets from dominating.
    def anchored(x):
        return x-x[:, :1]
    flat = anchored(hist).reshape(-1, hist.shape[-1]).astype(np.float64)
    mu = flat.mean(0)
    z = flat-mu
    eig, vec = np.linalg.eigh(z.T @ z)
    eig = eig[::-1].clip(min=0)
    pcs = vec[:, -2:][:, ::-1].copy()
    pcs *= np.sign(pcs[np.argmax(abs(pcs), axis=0), np.arange(2)])
    exemplar = ((anchored(test[:2])-mu) @ pcs)/norm
    return {"cross_time_r2": reads, "step_rms": speed.tolist(), "pair_separation_rms": sep.tolist(),
            "pc12_variance_fraction": float(eig[:2].sum()/eig.sum()),
            "participation_rank": float(eig.sum()**2/(eig**2).sum()), "example_pcs": exemplar.tolist()}


def blank_motion(hist, test, fit_xy, test_xy):
    """One time-blind position decoder, compared with decoding a frozen cue state plus a clock."""
    times = np.arange(cue.CUE+1, tr.FRAMES-1)
    actual = test_xy[:, times]
    delta = actual[:, -1]-actual[:, 0]
    scale = np.square(delta).sum()
    def measure(xy):
        change = xy-xy[:, :1]
        gain = (change*delta[:, None]).sum(axis=(0, 2))/scale
        return {"mean_error_px": float(np.linalg.norm(xy-actual, axis=-1).mean()),
                "projected_travel": gain.tolist()}
    probe = fit_ridge(hist[:, times].reshape(-1, hist.shape[-1]), fit_xy[:, times].reshape(-1, 2))
    pred = np.stack([probe(test[:, t]) for t in times], axis=1)
    # This readout is explicitly told the blank index, but never receives another model state.
    clock = np.stack([fit_ridge(hist[:, cue.CUE], fit_xy[:, t])(test[:, cue.CUE])
                      for t in times], axis=1)
    return {"time_blind": measure(pred), "frozen_cue_plus_clock": measure(clock),
            "true": measure(actual),
            "hold_visible_cue": measure(np.broadcast_to(test_xy[:, cue.CUE:cue.CUE+1], actual.shape))}


@torch.no_grad()
def swaps(model, sites, order, normal, calibration, rng):
    """Causal effects on native output, not on the position/bend probe."""
    a, b = (torch.from_numpy(np.ascontiguousarray(z)) for z in (sites[::2], sites[1::2]))
    def xy(h):
        if isinstance(model, MotionJEPA):
            h = model.context[3:](h)
        return ep.output(model, h, calibration, all_frames=True)[1]
    selections = {}
    for k in [s for s in SIZES if s < len(order)] + [len(order)]:
        selections[k] = [order[:k], *(rng.choice(len(order), k, replace=False)
                                     for _ in range(N_RANDOM) if k < len(order))]
    totals = {k: np.zeros((len(lists), 2)) for k, lists in selections.items()}
    full = 0.
    for i in range(0, len(a), 64):
        sl = slice(i, i+64)
        x, y = a[sl], b[sl]
        orig, moved = xy(x), xy(y)
        def bend(delta):
            return cue.per_pair(delta, normal[sl])["bend_px"].sum().item()
        full += bend(moved-orig)
        for k, lists in selections.items():
            for j, c in enumerate(lists):
                edited = x.clone()
                edited[:, c] = y[:, c]
                totals[k][j, 0] += bend(xy(edited)-orig)
                if j == 0:
                    blocked = y.clone()
                    blocked[:, c] = x[:, c]
                    totals[k][j, 1] += bend(moved-xy(blocked))
    return {"intact_px": full/len(a), "sizes": [
        {"k": k, "copied_px": a[0, 0]/len(normal), "removed_px": a[0, 1]/len(normal),
         "random_px": float(a[1:, 0].mean() if len(a) > 1 else a[0, 0])/len(normal)}
        for k, a in totals.items()]}


def unit_maps(model, ids):
    """Translate identical horizontal paths. Vary perpendicular bend independently on each grid point."""
    xs, ys = np.arange(9, 23), np.arange(7, 25)
    xy = np.array([(x, y) for y in ys for x in xs])
    shape = ap._mask("cross", 5)
    result = []
    for dy in (-3, 3):
        frames = np.zeros((len(xy), 3, tr.SIDE, tr.SIDE), dtype=np.float32)
        for i, (cx, cy) in enumerate(xy):
            for j, (dx, off) in enumerate(((-6, 0), (-1, dy), (6, 0))):
                x, y = cx+dx-2, cy+off-2
                frames[i, j, y:y+5, x:x+5] = shape
        _, units, _ = activities(model, torch.from_numpy(frames))
        result.append(units[:, ids].T.reshape(len(ids), len(ys), len(xs)))
    return {"x": xs.tolist(), "y": ys.tolist(), "units": list(map(int, ids)),
            "bend_up": result[0].tolist(), "bend_down": result[1].tolist()}


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    seen = set()
    fit = movies(args.fit, FIT, ("square", "circle"), seen)
    test = movies(args.test, TEST, ("cross",), seen)
    records, units = [], []
    for seed in range(args.seeds):
        for name in ("RNN", "JEPA"):
            torch.manual_seed(seed)
            model = (ap.ImputationNet(tr.SIDE**2, 512, tr.FRAMES) if name == "RNN" else MotionJEPA()).eval()
            for stage in ("initial", "trained"):
                if stage == "trained":
                    if name == "RNN":
                        model = load(torch.load(args.rnn/f"held-out_seed{seed}.pt",
                                                weights_only=True, map_location="cpu")).eval()
                    else:
                        model.load_state_dict(torch.load(args.jepa/f"jepa_seed{seed}.pt",
                                                         weights_only=True, map_location="cpu")["model"])
                h, x, _ = activities(model, fit.flashes)
                ht, xt, no = activities(model, test.flashes)
                out = {"model": name, "stage": stage, "seed": seed, "populations": {}, "examples": {}}
                rng = np.random.default_rng(141010+seed)
                orders = {}
                for key, y in fit.targets.items():
                    truth = test.targets[key]
                    f, scores, rank = unit_fits(x, y, xt, truth)
                    orders[key] = rank
                    for idx in range(len(rank)):
                        units.append([name, stage, seed, idx, key, float(f[idx]), float(scores[idx])])
                    out["examples"][key] = {"unit": int(rank[0]), "test_r2": float(scores[rank[0]]),
                                            "n_positive_0p1": int((scores >= .1).sum()), "n_units": len(rank)}
                    out["populations"][key] = population(x, y, xt, truth, no, rank, rng)
                out["geometry"] = geometry(h, ht, fit.targets, test.targets)
                out["blank_motion"] = blank_motion(h, ht, fit.path, test.path)
                cal = None
                if name == "JEPA":
                    # Each fitting flash gets its own true xy; no test or path labels used.
                    z = model.target(fit.flashes).flatten(0, 1).numpy()
                    cal = fit_ridge(z, fit.centres.reshape(-1, 2))
                out["swaps"] = swaps(model, xt, orders["bend"], test.normals, cal, rng)
                # Pair-preserving shuffled-label control; decorrelate direction and location.
                perm = rng.permutation(args.fit)
                idx = (2*perm[:, None]+np.array([0, 1])[None]).ravel()
                out["shuffled_r2"] = {k: r2(test.targets[k], fit_ridge(x, y[idx])(xt))
                                      for k, y in fit.targets.items()}
                if seed == 0 and stage == "trained":
                    out["maps"] = unit_maps(model, [orders[k][0] for k in ("start", "bend")])
                records.append(out)
                print(name, stage, seed, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"fit_chords": args.fit, "test_chords": args.test, "fit_seed": FIT,
                                    "test_seed": TEST, "test_shape": "cross", "records": records},
                                   allow_nan=False, indent=2)+"\n")
    args.units.parent.mkdir(parents=True, exist_ok=True)
    with args.units.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["model", "stage", "seed", "unit", "target", "fit_r2", "test_r2"])
        writer.writerows(units)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=int, default=1536)
    p.add_argument("--test", type=int, default=512)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rnn", type=Path, required=True)
    p.add_argument("--jepa", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/motion_unit_dynamics.json"))
    p.add_argument("--units", type=Path, default=Path("results/motion_unit_scores.csv"))
    args = p.parse_args()
    if min(args.fit, args.test) < 2 or args.seeds < 1:
        p.error("need at least two independent chords per split and one network seed")
    run(args)
