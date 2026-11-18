"""Train frame-level JEPA and pixel-control checkpoints with paired movies and masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import torch

from illusion_rnn import apparent as ap, trajectories as tr
from illusion_rnn.jepa import MotionJEPA, PixelControl


MODELS = {"jepa": MotionJEPA, "pixel": PixelControl}
DECAY = .99


def train(job):
    name, seed, args = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = MODELS[name]()
    history = []
    for sampler, steps, data_seed, lr in ((ap.sample_motion, args.straight_steps, seed, 1e-3),
                                          (tr.sample_trajectories, args.varied_steps, 51000+seed,
                                           3e-4)):
        rng = np.random.default_rng(data_seed)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        losses = []
        for _ in range(steps):
            frames = torch.from_numpy(sampler(rng, args.batch_size).frames)
            visible = torch.from_numpy(ap.sample_masks(rng, len(frames), tr.FRAMES, max_gap=4))
            if visible.all():
                # Nothing to predict in this batch. Keep the curriculum and step count paired.
                visible[0, -1] = False
            optimizer.zero_grad()
            loss = model.objective(frames, visible)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            if isinstance(model, MotionJEPA):
                model.update_target(DECAY)
            losses.append(loss.item())
        history.append({"steps": steps, "data_seed": data_seed, "lr": lr,
                        "first_100_loss": float(np.mean(losses[:100])),
                        "last_100_loss": float(np.mean(losses[-100:]))})
        print(f"{name} {seed}: {steps} {sampler.__name__} steps, loss {history[-1]['last_100_loss']:.4f}",
              flush=True)
    torch.save({"model": model.state_dict(), "objective": name, "seed": seed, "training": history,
                "ema_decay": DECAY if name == "jepa" else None, "batch_size": args.batch_size,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)},
               args.checkpoints / f"{name}_seed{seed}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--straight-steps", type=int, default=1000)
    parser.add_argument("--varied-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if min(args.seeds, args.straight_steps, args.varied_steps, args.batch_size, args.workers) < 1:
        parser.error("counts must be positive")
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(train, [(name, seed, args) for name, seed in product(MODELS, range(args.seeds))]))


if __name__ == "__main__":
    main()
