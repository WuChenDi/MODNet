# MODNet

> Trimap-Free Portrait Matting in Real Time

ONNX export/inference tooling and a Transformers.js example for the
[MODNet](https://github.com/ZHKKKe/MODNet) portrait matting model.

## Links

- [wuchendi/MODNet — Hugging Face](https://huggingface.co/wuchendi/MODNet)

## Contents

| Path | Description |
| --- | --- |
| [`onnx/`](onnx/README.md) | Export a MODNet checkpoint to ONNX and run inference with ONNX Runtime |
| [`examples/`](examples/README.md) | Portrait matting in JS/TS via `@huggingface/transformers` |
| [`pretrained/`](pretrained/README.md) | Location for downloaded pre-trained models |

## Quick start

- **Web / Node.js** — see [`examples/`](examples/README.md) to run matting with Transformers.js.
- **Python / ONNX** — see [`onnx/`](onnx/README.md) to export a checkpoint and run ONNX Runtime inference.
