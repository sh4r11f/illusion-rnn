# Checkpoint manifest

All RNN checkpoints are `illusion_rnn.models.RNNNet` state dicts; load them
with `illusion_rnn.models.load_rnn(path)`. Label map (frozen):
fixation=0, left=1, middle=2, right=3, down=4, up=5.

| File | Model | Input | Trained on | Eval (horizontal, mean of 3 shapes x 100 trials, seed 0) |
|---|---|---|---|---|
| `rnn-pixel_h2048_tam-horiz.pt` | RNNNet 4096->2048->6, dt=50 | raw 64x64 frames, flattened | horizontal standard TAM, all shapes (2023) | standard mean 0.967 (square 1.0, circle 1.0, triangle 0.9), outline 0.100, basic 0.967 |
| `rnn-cnnfeat64_h1024_tam-horiz.pt` | RNNNet 64->1024->6, dt=50 | 64-d ShapesCNN features of 100x100 frames | horizontal standard TAM, all shapes (2023) | standard mean 0.767 (square 0.79, circle 0.70, triangle 0.81) |
| `rnn-cnnfeat64_h2048_tam-horiz.pt` | RNNNet 64->2048->6, dt=50 | 64-d ShapesCNN features of 100x100 frames | horizontal standard TAM, all shapes (2023) | standard mean 0.620 (square 0.79, circle 0.26 — below chance ~0.33, triangle 0.81) |
| `cnn-shapes_feat64_100px.pt` | ShapesCNN (100px, 64-d, 9 classes) | 100x100 grayscale shape images | 2D geometric shapes dataset (El Korchi & Ghanou, 2020) | n/a (feature extractor) |

The `rnn-cnnfeat64_*` models expect inputs produced by
`cnn_encoder(shapes_cnn, ...)` with the ShapesCNN checkpoint above and
**img_size=100** environments:

    cnn = ShapesCNN()
    cnn.load_state_dict(torch.load("checkpoints/cnn-shapes_feat64_100px.pt", weights_only=True))
    encoder = cnn_encoder(cnn)
    env = make_env("tam", img_size=100, ...)
    evaluate(load_rnn("checkpoints/rnn-cnnfeat64_h2048_tam-horiz.pt"), env, encoder=encoder)

## Legacy

`legacy/rnn_v13` (full pickled model) and `legacy/rnn_v13_state_dict`
(1024->128->4) are an early 32x32 / 4-action prototype kept only as the
companion of the archived notebooks; they do not fit the current envs.

## Provenance

Trained in 2022-2023 with the original `src/` code (see `notebooks/legacy/`).
Verified to load and evaluated under the ported neurogym 2.x environments
on 2026-08-13.
