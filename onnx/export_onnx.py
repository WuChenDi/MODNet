"""
Export ONNX model of MODNet with:
    input shape: (batch_size, 3, height, width)
    output shape: (batch_size, 1, height, width)  

Arguments:
    --ckpt-path: path of the checkpoint that will be converted
    --output-path: path for saving the ONNX model

Example:
    python export_onnx.py \
        --ckpt-path=modnet_photographic_portrait_matting.ckpt \
        --output-path=modnet_photographic_portrait_matting.onnx
"""

import os
import time
import argparse
import swanlab
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import modnet_onnx

def count_parameters(model):
    """Count total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    # Initialize SwanLab run
    start_time = time.time()
    swanlab.init(
        project="MODNet",
        workspace="wudi",
        config={
            "model": "MODNet",
            "ckpt_path": None,  # Will be updated after parsing args
            "output_path": None,  # Will be updated after parsing args
            "input_shape": [1, 3, 512, 512],  # batch_size, channels, height, width
            "opset_version": 11,  # ONNX opset version
            "step": "initializing"
        }
    )
    swanlab.log({"step": "initializing", "init_time": time.time() - start_time})

    # Step 1: Parse command-line arguments
    step_start_time = time.time()
    swanlab.log({"step": "parsing_arguments"})
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='path of the checkpoint that will be converted')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the ONNX model')
    args = parser.parse_args()
    swanlab.config.ckpt_path = args.ckpt_path
    swanlab.config.output_path = args.output_path
    swanlab.log({
        "step": "arguments_parsed",
        "ckpt_path": args.ckpt_path,
        "output_path": args.output_path,
        "parse_time": time.time() - step_start_time
    })

    # Step 2: Check input arguments
    step_start_time = time.time()
    swanlab.log({"step": "checking_input_path"})
    if not os.path.exists(args.ckpt_path):
        swanlab.log({"step": "input_path_error", "error": f"Cannot find checkpoint path: {args.ckpt_path}"})
        print(f'Cannot find checkpoint path: {args.ckpt_path}')
        exit()
    ckpt_size = os.path.getsize(args.ckpt_path) / (1024 * 1024)  # File size in MB
    swanlab.log({
        "step": "input_path_validated",
        "ckpt_size_mb": ckpt_size,
        "check_path_time": time.time() - step_start_time
    })

    # Step 3: Define model & load checkpoint
    step_start_time = time.time()
    swanlab.log({"step": "loading_model"})
    modnet = modnet_onnx.MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    state_dict = torch.load(args.ckpt_path)
    modnet.load_state_dict(state_dict)
    modnet.eval()
    param_count = count_parameters(modnet)
    swanlab.log({
        "step": "model_loaded",
        "param_count": param_count,
        "load_model_time": time.time() - step_start_time
    })

    # Step 4: Prepare dummy input
    step_start_time = time.time()
    swanlab.log({"step": "preparing_dummy_input"})
    batch_size = 1
    height = 512
    width = 512
    dummy_input = Variable(torch.randn(batch_size, 3, height, width)).cuda()
    swanlab.log({
        "step": "dummy_input_prepared",
        "input_shape": [batch_size, 3, height, width],
        "prepare_input_time": time.time() - step_start_time
    })

    # Step 5: Export to ONNX model
    step_start_time = time.time()
    swanlab.log({"step": "exporting_onnx"})
    torch.onnx.export(
        modnet.module, dummy_input, args.output_path, export_params=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0:'batch_size', 2:'height', 3:'width'}, 'output': {0:'batch_size', 2:'height', 3:'width'}},
        opset_version=11  # Added to resolve Upsample warning
    )
    onnx_size = os.path.getsize(args.output_path) / (1024 * 1024)  # File size in MB
    swanlab.log({
        "step": "onnx_exported",
        "output_file": args.output_path,
        "onnx_size_mb": onnx_size,
        "export_time": time.time() - step_start_time
    })

    # Step 6: Finish SwanLab run
    total_time = time.time() - start_time
    swanlab.log({
        "step": "completed",
        "total_time": total_time
    })
    swanlab.finish()
