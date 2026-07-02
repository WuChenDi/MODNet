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

```bash
pip install -r onnx/requirements.txt

# Or using a mirror
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r onnx/requirements.txt --timeout 1000
```

### 3. Export the ONNX model

Runs on GPU when available, otherwise CPU.

```bash
python -m onnx.export_onnx \
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
python -m onnx.inference_onnx \
  --image-path=pretrained/logo.jpg \
  --output-path=pretrained/matte.png \
  --model-path=pretrained/modnet_photographic_portrait_matting.onnx
```

| Argument | Required | Description |
| --- | --- | --- |
| `--image-path` | yes | Input image (a file) |
| `--output-path` | yes | Path to save the predicted alpha matte |
| `--model-path` | yes | Path to the ONNX model |
