## MODNet ONNX Export Guide

> 💡 **Note:** The ONNX export requires a PyTorch version higher than the one used in the official MODNet repository. It is recommended to use `torch==1.11.0`.

You can download the pre-trained **Image Matting Model** (ONNX format) from the following link:
👉 [Download from Google Drive](https://drive.google.com/drive/folders/1OUFBMSD0RwcfIDXd4mvv8eBJv-NdnzDW?usp=sharing)

---

### 🛠️ Steps to Export MODNet to ONNX

> Ensure you are in the root directory of the MODNet project.

#### 1. Download the Pre-trained Model

Download the model from the link above and place it in the following directory:

```
MODNet/pretrained/
```

Example filename:
`modnet_photographic_portrait_matting.ckpt`

---

#### 2. Install Dependencies

Install required dependencies:

```bash
pip install -r onnx/requirements.txt

# Or using Tsinghua mirror
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r onnx/requirements.txt --timeout 1000
```

---

#### 3. Export the ONNX Model

```bash
python -m onnx.export_onnx \
  --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
  --output-path=pretrained/modnet_photographic_portrait_matting.onnx
```

---

#### 4. Run Inference with ONNX Model

```bash
python -m onnx.inference_onnx \
  --image-path=pretrained/logo.jpg \
  --output-path=pretrained/matte.png \
  --model-path=pretrained/modnet_photographic_portrait_matting.onnx
```
