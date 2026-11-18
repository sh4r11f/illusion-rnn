"""Fixed-rate RGB clips for the after-sequence imputation task."""
import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Clip:
    video: Path
    start_seconds: float


def read_manifest(path: Path) -> list[Clip]:
    clips = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        if not {"video", "start_seconds"}.issubset(rows.fieldnames or ()):
            raise ValueError(f"{path}: expected CSV columns video,start_seconds")
        for row in rows:
            try:
                video = (path.parent / row["video"]).resolve(strict=True)
                start = float(row["start_seconds"])
                if not video.is_file() or not math.isfinite(start) or start < 0:
                    raise ValueError("expected a local video file and a finite nonnegative start")
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{rows.line_num}: {exc}") from exc
            clips.append(Clip(video, start))
    if not clips:
        raise ValueError(f"{path}: no clips")
    return clips


def check_splits(train: list[Clip], val: list[Clip]) -> None:
    overlap = {c.video for c in train} & {c.video for c in val}
    if overlap:
        raise ValueError(f"train and validation share a source video: {min(overlap)}")


class VideoClips(Dataset):
    """Independent seeks; decode only the bounded RGB output for the requested window."""
    def __init__(self, clips: list[Clip], frames: int, side: int, fps: int):
        if min(frames, side, fps) < 1:
            raise ValueError("frames, side and fps must be positive")
        decoder = shutil.which("ffmpeg")
        if not decoder:
            raise RuntimeError("VideoClips requires ffmpeg on PATH")
        self.ffmpeg, self.clips = decoder, clips
        self.frames, self.side, self.fps = frames, side, fps

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        s = self.side
        filters = (f"fps={self.fps},scale={s}:{s}:force_original_aspect_ratio=increase,"
                   f"crop={s}:{s},setsar=1")
        cmd = [self.ffmpeg, "-nostdin", "-v", "error", "-ss", str(clip.start_seconds),
               "-threads", "1", "-i", str(clip.video), "-map", "0:v:0", "-an", "-sn",
               "-filter_threads", "1", "-vf", filters, "-frames:v", str(self.frames),
               "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode or len(proc.stdout) != self.frames * s * s * 3:
            detail = proc.stderr.decode(errors="replace").strip() or "clip is too short"
            raise ValueError(f"{clip.video} at {clip.start_seconds}s: {detail}")
        # Keep host batches uint8 until transfer to the GPU.
        return torch.frombuffer(bytearray(proc.stdout), dtype=torch.uint8).view(
            self.frames, s, s, 3).permute(0, 3, 1, 2)


def visible_mask(batch: int, frames: int, device, *, endpoints_only=False):
    """Training always has an interior cue and a target. No cues at validation."""
    if frames < 4:
        raise ValueError("need at least four frames")
    mask = torch.zeros(batch, frames, dtype=torch.bool, device=device)
    mask[:, [0, -1]] = True
    if not endpoints_only:
        order = torch.rand(batch, frames - 2, device=device).argsort(dim=1)
        count = torch.randint(1, frames - 2, (batch, 1), device=device)
        mask[:, 1:-1] = order < count
    return mask
