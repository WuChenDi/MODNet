# Vendored MODNet source

The files under `src/` are vendored from the official
[ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) repository
(`src/`, Apache License 2.0) so that the ONNX export pipeline in
[`onnx_tools/`](../onnx_tools/README.md) can run without cloning the upstream project.

Only the backbone package required by `onnx_tools/modnet_onnx.py` is included:

```
src/models/backbones/
├── __init__.py       # SUPPORTED_BACKBONES
├── mobilenetv2.py    # MobileNetV2 implementation
└── wrapper.py        # MobileNetV2Backbone (enc_channels, forward, load_pretrained_ckpt)
```

The full model definition (`src/models/modnet.py`) and training code are **not**
vendored — `onnx_tools/modnet_onnx.py` provides an ONNX-friendly replacement of the model.

## Changes from upstream

Kept minimal so the code stays easy to diff against upstream:

- Removed unused imports (`json`, `functools.reduce`).
- `torch.load(...)` in the pretrained-backbone loaders now passes
  `map_location='cpu'` and `weights_only=True` (compatible with the PyTorch 2.6+
  default and CPU-only machines).
