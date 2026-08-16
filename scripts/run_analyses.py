#!/usr/bin/env python3
"""Run the dynamics/RSA analysis suite on a TAM checkpoint.

Evaluates the model shape-balanced (square/circle/triangle) per stimulus
variant, then computes PCA trajectories, binned loadings, top-unit
timecourses, unit RDMs, second-order condition RDMs, and the cross-variant
second-order RDM. Saves compact artifacts to --out for the visualization
notebooks (03_dynamics, 04_rsa).

Usage:
    uv run python scripts/run_analyses.py [--n-trials 300] [--seed 0]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from illusion_rnn import SHAPES, __version__, evaluate, load_rnn, make_env  # noqa: E402
from illusion_rnn import analysis as an  # noqa: E402

DIRECTION_NAMES = {1: "left", 2: "middle", 3: "right"}
CONDITIONS = np.array([1, 2, 3])
DT_MS = 50.0  # default TAMTask dt; envs below are constructed with defaults


def collect_variant(model, variant, n_trials, seed, device):
    """Shape-balanced evaluation; returns (activity, labels, mean accuracy)."""
    activities, labels, accuracies = [], [], []
    per_shape = max(1, n_trials // len(SHAPES))
    for shape in SHAPES:
        env = make_env(
            "tam", box_shape=shape, variant=variant,
            stim_ori="horizontal", img_size=64,
        )
        env.seed(seed)
        result = evaluate(model, env, n_trials=per_shape, device=device)
        activity, trial_labels = an.stack_activity(result)
        activities.append(activity)
        labels.append(trial_labels)
        accuracies.append(result.accuracy)
    return (
        np.concatenate(activities, axis=0),
        np.concatenate(labels, axis=0),
        float(np.mean(accuracies)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/rnn-pixel_h2048_tam-horiz.pt"))
    # "basic" is deliberately excluded: all 24 of its images are byte-identical
    # to TAM_task images (21 share filenames, 3 are internal duplicates), so it
    # is not a held-out variant -- see tests/test_stimuli_integrity.py.
    parser.add_argument("--variants", nargs="+",
                        default=["standard", "outline"])
    parser.add_argument("--n-trials", type=int, default=300,
                        help="per variant (split across the 3 shapes)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff", type=int, default=3,
                        help="first analyzed timestep (frame2 onset)")
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10,
                        help="top loading units per PC per bin")
    parser.add_argument("--rdm-top-k", type=int, default=30,
                        help="units in the stored RDM heatmaps")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists() or args.checkpoint.stat().st_size <= 1024:
        msg = (
            f"{args.checkpoint} is missing or an LFS pointer; run "
            "'git lfs install --local && git lfs checkout' first"
        )
        raise SystemExit(msg)

    model = load_rnn(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- evaluate ----------------------------------------------------------
    runs = {}
    accuracy = {}
    for variant in args.variants:
        activity, labels, acc = collect_variant(
            model, variant, args.n_trials, args.seed, args.device,
        )
        cond_mean, cond_sem, _ = an.condition_average(
            activity, labels, conditions=CONDITIONS,
        )
        runs[variant] = (activity, labels, cond_mean, cond_sem)
        accuracy[variant] = round(acc, 3)
        print(f"{variant}: n={len(labels)} accuracy={acc:.3f}")

    # ---- dynamics ----------------------------------------------------------
    basis_variant = "standard" if "standard" in runs else args.variants[0]
    basis = an.run_pca(runs[basis_variant][0], n_components=2)
    n_timepoints = next(iter(runs.values()))[0].shape[1]
    dynamics = {
        "components": basis.components.astype(np.float32),
        "pca_mean": basis.mean.astype(np.float32),
        "t_ms": np.arange(n_timepoints) * DT_MS,
        "conditions": CONDITIONS,
        "basis_variant": np.array(basis_variant),
    }
    for variant, (activity, labels, cond_mean, cond_sem) in runs.items():
        evr = an.run_pca(activity, n_components=min(20, activity.shape[-1]))
        binned = an.binned_pca_loadings(
            activity, n_bins=args.n_bins, n_components=2,
        )
        top = an.top_units(binned.loadings, k=args.top_k)
        unit_ids = an.unique_units(top)
        dynamics.update({
            f"{variant}_trajectories": basis.transform(activity).astype(np.float32),
            f"{variant}_labels": labels,
            f"{variant}_mean_trajectories": basis.transform(cond_mean),
            f"{variant}_evr": evr.explained_variance_ratio,
            f"{variant}_loadings": binned.loadings.astype(np.float32),
            f"{variant}_top_units": top,
            f"{variant}_unit_ids": unit_ids,
            f"{variant}_unit_mean": cond_mean[:, :, unit_ids],
            f"{variant}_unit_sem": cond_sem[:, :, unit_ids],
        })
    dynamics["bin_edges"] = binned.bin_edges
    np.savez_compressed(args.out / "dynamics.npz", **dynamics)

    # ---- rsa ---------------------------------------------------------------
    cond_means = [runs[v][2] for v in args.variants]
    mask = an.active_unit_mask(cond_means, cutoff=args.cutoff)
    active_ids = np.where(mask)[0]
    if len(active_ids) < args.rdm_top_k:
        msg = f"only {len(active_ids)} active units (< --rdm-top-k {args.rdm_top_k})"
        raise SystemExit(msg)
    pooled_rate = np.mean(
        [cm[:, args.cutoff:, :].mean(axis=(0, 1)) for cm in cond_means], axis=0,
    )
    order = np.argsort(pooled_rate[active_ids])[::-1]
    rdm_unit_ids = active_ids[order[: args.rdm_top_k]]

    rsa = {
        "active_units": mask,
        "rdm_unit_ids": rdm_unit_ids,
        "cutoff": np.array(args.cutoff),
    }
    cross = {}
    for variant in args.variants:
        cond_mean = runs[variant][2]
        rsa[f"{variant}_unit_rdms"] = an.compute_rdms(
            cond_mean, cutoff=args.cutoff, units=rdm_unit_ids,
        )
        full_rdms = an.compute_rdms(cond_mean, cutoff=args.cutoff, units=active_ids)
        rsa[f"{variant}_condition_rdm"] = an.second_order_rdm(full_rdms)
        for i, condition in enumerate(CONDITIONS):
            cross[f"{variant}-{DIRECTION_NAMES[int(condition)]}"] = full_rdms[i]
    keys, cross_rdm = an.cross_variant_rdm(cross)
    rsa["cross_keys"] = np.array(keys)
    rsa["cross_rdm"] = cross_rdm
    np.savez_compressed(args.out / "rsa.npz", **rsa)

    # ---- meta --------------------------------------------------------------
    meta = {
        "checkpoint": str(args.checkpoint),
        "variants": list(args.variants),
        "n_trials": args.n_trials,
        "seed": args.seed,
        "cutoff": args.cutoff,
        "n_bins": args.n_bins,
        "top_k": args.top_k,
        "rdm_top_k": args.rdm_top_k,
        "device": args.device,
        "accuracy": accuracy,
        "n_active_units": int(mask.sum()),
        "package_version": __version__,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out}/dynamics.npz, rsa.npz, meta.json "
          f"({mask.sum()} active units)")


if __name__ == "__main__":
    main()
