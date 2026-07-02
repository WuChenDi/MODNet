"""
Export ONNX model of MODNet with:
    input shape: (batch_size, 3, height, width)
    output shape: (batch_size, 1, height, width)

Arguments:
    --ckpt-path: path of the checkpoint that will be converted
    --output-path: path for saving the ONNX model
    --opset-version: ONNX opset version (default: 17)

Example:
    python -m onnx_tools.export_onnx \
        --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
        --output-path=pretrained/modnet_photographic_portrait_matting.onnx
"""

import os
import argparse

import torch

from . import modnet_onnx


def strip_module_prefix(state_dict):
    """Remove the 'module.' prefix left by nn.DataParallel checkpoints."""
    prefix = 'module.'
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the ONNX model')
    parser.add_argument('--opset-version', type=int, default=17, help='ONNX opset version')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find checkpoint path: {args.ckpt_path}')
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Define model & load checkpoint (strip DataParallel prefix so it runs on CPU too)
    modnet = modnet_onnx.MODNet(backbone_pretrained=False).to(device)
    state_dict = torch.load(args.ckpt_path, map_location=device, weights_only=True)
    modnet.load_state_dict(strip_module_prefix(state_dict))
    modnet.eval()

    # Export to ONNX model
    dummy_input = torch.randn(1, 3, 512, 512, device=device)
    torch.onnx.export(
        modnet, dummy_input, args.output_path, export_params=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        },
        opset_version=args.opset_version,
        dynamo=False,  # use the legacy TorchScript exporter: embeds weights and honors dynamic_axes
    )
    onnx_size = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f'ONNX model saved to: {args.output_path} ({onnx_size:.2f} MB)')
