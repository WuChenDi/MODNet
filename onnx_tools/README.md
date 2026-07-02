## MODNet ONNX Export & Inference Guide

Export a trained MODNet checkpoint to ONNX and run portrait matting with ONNX Runtime.

> **Note:** These scripts define an ONNX-friendly model (`modnet_onnx.py`) that outputs
> only the alpha matte. It uses the MobileNetV2 backbone vendored under
> [`src/`](../src/README.md) (from the official [MODNet](https://github.com/ZHKKKe/MODNet)
> repository), so the pipeline is self-contained — just run the commands below from the
> repository root.

### 1. Download the pre-trained model

Download the checkpoint and place it under `pretrained/`:

👉 [Download from Google Drive](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing)

Example filename: `modnet_photographic_portrait_matting.ckpt`

### 2. Install dependencies

Create and activate a virtual environment first:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Then install the requirements:

```bash
pip install -r onnx_tools/requirements.txt

# Or using a mirror
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r onnx_tools/requirements.txt --timeout 1000
```

### 3. Export the ONNX model

Runs on GPU when available, otherwise CPU.

```bash
python -m onnx_tools.export_onnx \
  --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
  --output-path=pretrained/modnet_photographic_portrait_matting.onnx
```

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--ckpt-path` | yes | — | Path to the `.ckpt` checkpoint to convert |
| `--output-path` | yes | — | Path for the exported `.onnx` model |
| `--opset-version` | no | `17` | ONNX opset version |

The exported model has a dynamic input shape `(batch_size, 3, height, width)` and
output shape `(batch_size, 1, height, width)`.

### 4. Run inference with the ONNX model

```bash
python -m onnx_tools.inference_onnx \
  --image-path=pretrained/logo.jpg \
  --output-path=pretrained/matte.png \
  --model-path=pretrained/modnet_photographic_portrait_matting.onnx
```

| Argument | Required | Description |
| --- | --- | --- |
| `--image-path` | yes | Input image (a file) |
| `--output-path` | yes | Path to save the predicted alpha matte |
| `--model-path` | yes | Path to the ONNX model |

## Notes for PyTorch 2.x export

The export path was originally written for PyTorch 1.11. Three things are needed
to make it work on PyTorch 2.x:

- **`onnxscript` is required.** The PyTorch 2.x ONNX exporter imports `onnxscript`
  internally, so it is listed in `requirements.txt`. PyTorch 1.11 did not need it.
- **The tools live in `onnx_tools/`, not `onnx/`.** A local package named `onnx`
  would shadow the installed `onnx` library on `sys.path`, breaking the exporter with
  `ModuleNotFoundError: No module named 'onnx.external_data_helper'`.
- **The exporter is pinned to `dynamo=False`** (legacy TorchScript exporter) in
  `export_onnx.py`. The new `torch.export`-based exporter (default since PyTorch 2.9)
  does not embed the weights here (it produces a ~0.6 MB file instead of ~25 MB) and
  fails the opset down-conversion for this model. The legacy exporter embeds the
  weights and honors `dynamic_axes`.

Two warnings during export are expected and harmless: a `DeprecationWarning` about the
legacy exporter, and a `UserWarning` that `instance_norm` is exported with `train=True`
(instance norm uses per-instance statistics at inference by design — the upstream MODNet
export emits the same warning).
