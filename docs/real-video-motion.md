# Masked imputation on real video (single GPU)

Train on continuous RGB clips with some frames hidden. At validation, give
the model only the first and last frame and score **all hidden frames** against
the real clip. These are new model implementations, not pretrained weights or
measured results; neither GPU experiment has been run.

| Model | Context and prediction | Validation target |
|---|---|---|
| `convgru` | Strided CNN plus two 256-channel ConvGRU layers. The last recurrent state and a learned time embedding decode every RGB frame. | RGB values in [0,1] |
| `jepa` | Six spatial transformer layers encode visible frames in 16×16 patches. Four temporal layers predict missing patch embeddings, using ordered slots and a learned mask token. Default dimension 384, six heads. | Frozen-gradient EMA teacher's patch embeddings |

The RNN gets an explicit visibility channel to distinguish a hidden frame
from a visible black frame. The JEPA student never encodes hidden RGB; the
teacher sees hidden RGB only to supply a training target or evaluation
reference. The teacher follows the student frame encoder with EMA decay 0.99.
A variance penalty across visible images at each spatial position discourages
constant embeddings; the penalty is excluded from validation.

## Data

Supply local videos you have permission to use, plus two CSV manifests. Each
row selects one clip starting at `start_seconds`; paths are relative to the CSV
(absolute paths also work). For example, in `train.csv`:

```csv
video,start_seconds
videos/scene-a.mp4,0
videos/scene-a.mp4,4.5
videos/scene-b.mkv,10
```

Use **different source videos** in `val.csv`, with the same columns. Training
rejects overlap by resolved source path. Keep duplicate exports, cuts from
the same recording and near-identical scenes in the same split yourself.
Prefer clips with continuous motion and no shot transitions. No downloader,
dataset credentials, bounding boxes, or object labels are required.

FFmpeg seeks to the requested start, resamples to constant FPS, scales the
short edge to the requested resolution, and center-crops the same region
throughout. Defaults: 8 frames at 8 fps, 128×128 RGB. Choose windows with
at least `frames / fps` seconds available. Short/corrupt clips are errors,
not silently repeated frames. Raw uint8 clips stay on the CPU until batched
transfer. Each manifest row is read once per epoch in shuffled order.

## Run later

Use Python ≥3.10, an NVIDIA GPU/driver, [CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/)
in the repo environment, and `ffmpeg` on `PATH`. From the repo root, examples
for a **future** training machine:

```bash
uv sync
# Check the resulting torch build without starting a model.
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

uv run python -m scripts.run_video_motion train --model convgru \
  --train-manifest train.csv --val-manifest val.csv --checkpoint runs/convgru.pt

uv run python -m scripts.run_video_motion train --model jepa \
  --train-manifest train.csv --val-manifest val.csv --checkpoint runs/jepa.pt

uv run python -m scripts.run_video_motion eval \
  --val-manifest val.csv --checkpoint runs/jepa.pt
```

Training is opt-in and refuses to fall back to CPU. Select a GPU with
`CUDA_VISIBLE_DEVICES`. `--precision auto` uses bf16 autocast when supported
and fp32 otherwise. Lower `--batch-size` first if memory is limited. `--side`
must be divisible by 16; `--width` scales the ConvGRU, `--dim` and `--heads`
scale the JEPA. No multi-GPU launcher is required or implemented.

The checkpoint is replaced atomically after each completed epoch. To continue,
repeat the training command with `--resume --epochs N` (desired total);
use the same manifests, sampling, model dimensions, batch size and seed.
Optimizer state, including its learning rate, resumes from the checkpoint.
Standalone evaluation loads model dimensions and FPS from the checkpoint;
use only trusted checkpoints and held-out source videos.

## Reading the metrics

Training always shows both endpoints, with a random nonempty subset of
interior frames visible and at least one interior frame hidden. Validation
never shows an interior frame. Each JSON line gives the epoch and training
loss, `hidden_mse` and `endpoint_crossfade_mse`. The latter uses only the
endpoints, interpolating their RGB or teacher embeddings at each hidden time.
MSE is weighted over hidden elements, not batches. **Only compare losses
within the same output space.** Low pixel MSE can reward static backgrounds;
low embedding MSE can reflect weak teacher variation. Neither metric alone
shows object-position tracking; that claim needs held-out position tracks
or a calibrated readout like the synthetic experiments.
