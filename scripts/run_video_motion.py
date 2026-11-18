"""Train on masked real video or evaluate a saved model on endpoints alone (single CUDA GPU)."""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from illusion_rnn.video_data import VideoClips, check_splits, read_manifest, visible_mask
from illusion_rnn.video_models import VideoConfig, VideoConvGRU, VideoJEPA, interpolate_endpoints


MODELS = {"convgru": VideoConvGRU, "jepa": VideoJEPA}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train", help="train and evaluate once per epoch")
    test = sub.add_parser("eval", help="evaluate an existing checkpoint without training")
    for p in (train, test):
        p.add_argument("--val-manifest", required=True, type=Path)
        p.add_argument("--checkpoint", required=True, type=Path)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    train.add_argument("--model", choices=MODELS, required=True)
    train.add_argument("--train-manifest", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--fps", type=int, default=8)
    train.add_argument("--frames", type=int, default=8)
    train.add_argument("--side", type=int, default=128)
    train.add_argument("--width", type=int, default=256, help="ConvGRU hidden channels")
    train.add_argument("--dim", type=int, default=384, help="JEPA embedding dimension")
    train.add_argument("--heads", type=int, default=6, help="JEPA attention heads")
    train.add_argument("--resume", action="store_true",
                       help="continue this checkpoint (epochs is the desired total)")
    return root


def loader(clips, cfg, fps, args, *, shuffle=False, seed=0):
    dataset = VideoClips(clips, cfg.frames, cfg.side, fps)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                      generator=torch.Generator().manual_seed(seed), shuffle=shuffle,
                      pin_memory=True)


def cuda_settings(precision):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA-enabled PyTorch build and check the driver.")
    supported = torch.cuda.is_bf16_supported()
    if precision == "bf16" and not supported:
        raise ValueError("bf16 not supported on this GPU; use fp32")
    return torch.device("cuda"), supported and precision != "fp32"


def prepare(batch, device):
    return batch.to(device=device, dtype=torch.float32, non_blocking=True).div_(255)


@torch.no_grad()
def evaluate(model, batches, device, amp):
    model.eval()
    sums = torch.zeros(2, device=device, dtype=torch.float64)
    n = 0
    for batch in batches:
        clip = prepare(batch, device)
        vis = visible_mask(*clip.shape[:2], device, endpoints_only=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred, target = model.validation_pair(clip, vis)
        target = target.float()
        baseline = interpolate_endpoints(target)
        hidden = target[~vis]
        sums[0] += (pred[~vis].float() - hidden).square().sum(dtype=torch.float64)
        sums[1] += (baseline[~vis] - hidden).square().sum(dtype=torch.float64)
        n += hidden.numel()
    model_mse, baseline_mse = (sums / n).tolist()
    return {"space": "teacher_embedding" if isinstance(model, VideoJEPA) else "rgb_0_to_1",
            "hidden_mse": model_mse, "endpoint_crossfade_mse": baseline_mse}


def train_epoch(model, batches, opt, device, amp):
    model.train()
    total, count = 0., 0
    for batch in batches:
        clip = prepare(batch, device)
        vis = visible_mask(*clip.shape[:2], device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = model.objective(clip, vis)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if isinstance(model, VideoJEPA):
            model.update_target()
        total += loss.detach().item() * len(clip)
        count += len(clip)
    return total / count


def main(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    device, amp = cuda_settings(args.precision)
    val = read_manifest(args.val_manifest)
    ckpt = None
    if args.action == "eval" or args.resume:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if args.action == "eval":
        cfg = VideoConfig(**ckpt["config"])
        model = MODELS[ckpt["model"]](cfg).to(device)
        model.load_state_dict(ckpt["weights"])
        metrics = evaluate(model, loader(val, cfg, ckpt["fps"], args), device, amp)
        print(json.dumps(metrics, allow_nan=False))
        return

    if args.epochs < 1 or args.fps < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs, fps and lr must be positive")
    if args.checkpoint.exists() and not args.resume:
        raise FileExistsError(f"{args.checkpoint} exists; specify --resume or choose another path")
    cfg = VideoConfig(frames=args.frames, side=args.side, width=args.width, dim=args.dim, heads=args.heads)
    train = read_manifest(args.train_manifest)
    check_splits(train, val)
    spec = {"config": asdict(cfg), "fps": args.fps, "model": args.model, "seed": args.seed}
    if ckpt and any(ckpt[k] != v for k, v in spec.items()):
        raise ValueError("resume must use the checkpoint's model, dimensions, seed and fps")
    torch.manual_seed(args.seed)
    model = MODELS[args.model](cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    start = 0
    if ckpt:
        model.load_state_dict(ckpt["weights"])
        opt.load_state_dict(ckpt["optimizer"])
        start = ckpt["epoch"]
        del ckpt
        if start >= args.epochs:
            raise ValueError("checkpoint already reached the requested number of epochs")
    val_batches = loader(val, cfg, args.fps, args)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    for epoch in range(start, args.epochs):
        seed = args.seed + epoch
        torch.manual_seed(seed)
        batches = loader(train, cfg, args.fps, args, shuffle=True, seed=seed)
        loss = train_epoch(model, batches, opt, device, amp)
        metrics = evaluate(model, val_batches, device, amp)
        row = {"epoch": epoch + 1, "train_loss": loss, **metrics}
        print(json.dumps(row, allow_nan=False), flush=True)
        torch.save({**spec, "weights": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch + 1, "metrics": metrics}, tmp)
        tmp.replace(args.checkpoint)


if __name__ == "__main__":
    main(parser().parse_args())
