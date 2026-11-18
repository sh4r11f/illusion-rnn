import shutil
import subprocess

import pytest
import torch

from illusion_rnn.video_data import Clip, VideoClips, check_splits, read_manifest, visible_mask
from illusion_rnn.video_models import VideoConfig, VideoConvGRU, VideoJEPA, interpolate_endpoints
from scripts.run_video_motion import parser


def test_source_split_and_manifest_paths(tmp_path):
    source = tmp_path / "shot, one.mkv"
    source.touch()
    alias = tmp_path / "alias.mkv"
    alias.symlink_to(source)
    manifest = tmp_path / "clips.csv"
    manifest.write_text('video,start_seconds\n"shot, one.mkv",1.25\n')
    train = read_manifest(manifest)
    assert train == [Clip(source, 1.25)]
    manifest.write_text("video,start_seconds\nalias.mkv,5\n")
    with pytest.raises(ValueError, match="share a source video"):
        check_splits(train, read_manifest(manifest))
    for start in ("-1", "inf", "nan", ""):
        manifest.write_text(f"video,start_seconds\nalias.mkv,{start}\n")
        with pytest.raises(ValueError, match=r"clips.csv:2"):
            read_manifest(manifest)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg CLI not installed")
def test_decode_rgb_window_and_reject_truncation(tmp_path):
    # Tiny lossless striped video, no network, GPU or model.
    path = tmp_path / "strip.mkv"
    frames = torch.zeros(8, 16, 32, 3, dtype=torch.uint8)
    frames[:, :, 8:24, 0] = torch.arange(8, dtype=torch.uint8)[:, None, None] * 30
    frames[:, :, 8:24, 2] = 211
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "32x16", "-r", "4", "-i", "pipe:0", "-c:v", "ffv1",
                    "-threads", "1", str(path)], input=frames.numpy().tobytes(), check=True)
    clips = VideoClips([Clip(path, .5), Clip(path, 1.75)], frames=4, side=16, fps=4)
    torch.testing.assert_close(clips[0], frames[2:6, :, 8:24].permute(0, 3, 1, 2))
    with pytest.raises(ValueError, match="too short"):
        clips[1]


@pytest.mark.parametrize("t", [4, 8, 17])
def test_masks_hold_out_endpoints_only_condition(t):
    train = visible_mask(64, t, "cpu")
    assert train.dtype == torch.bool and train.shape == (64, t)
    assert train[:, [0, -1]].all() and train[:, 1:-1].any(dim=1).all()
    assert (~train).any(dim=1).all()
    val = visible_mask(64, t, "cpu", endpoints_only=True)
    assert val[:, [0, -1]].all() and not val[:, 1:-1].any()


@pytest.mark.parametrize("shape", [(2, 5, 3, 7, 7), (2, 5, 4, 32)])
def test_crossfade_uses_only_endpoints(shape):
    values = torch.full(shape, 99.)
    values[:, 0], values[:, -1] = 0, 1
    pred = interpolate_endpoints(values)
    for t in range(5):
        assert (pred[:, t] == t / 4).all()


def test_cli_and_invalid_geometry():
    with pytest.raises(ValueError):
        VideoConfig(side=127)
    with pytest.raises(ValueError):
        VideoConfig(dim=128, heads=3)
    args = parser().parse_args(["train", "--model", "convgru", "--train-manifest", "train.csv",
                                "--val-manifest", "val.csv", "--checkpoint", "weights.pt"])
    assert not args.resume and args.frames == VideoConfig().frames


@pytest.mark.slow
@pytest.mark.parametrize("cls", [VideoConvGRU, VideoJEPA])
def test_opt_in_tiny_forward_masking_and_backward(cls):
    cfg = VideoConfig(frames=4, side=32, width=32, dim=24, heads=3, spatial_depth=1, temporal_depth=1)
    model = cls(cfg)
    clip = torch.rand(2, 4, 3, 32, 32)
    mask = visible_mask(2, 4, "cpu")
    other = clip.clone()
    other[~mask] = 1 - other[~mask]
    with torch.no_grad():
        pred = model(clip, mask)
        torch.testing.assert_close(model(other, mask), pred)
        ref_shape = (2, 4, 4, 24) if cls is VideoJEPA else clip.shape
        assert pred.shape == ref_shape
    model.objective(clip, mask).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if isinstance(model, VideoJEPA):
        assert all(p.grad is None for p in model.target.parameters())
