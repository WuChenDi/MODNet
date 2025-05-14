"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

import os
import time
import argparse
import swanlab
import cv2
import numpy as np
from PIL import Image
import onnx
import onnxruntime

def get_scale_factor(im_h, im_w, ref_size):
    """Get x_scale_factor & y_scale_factor to resize image."""
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor

if __name__ == '__main__':
    # Initialize SwanLab run
    start_time = time.time()
    swanlab.init(
        project="MODNet",
        workspace="wudi",
        description="Inference of MODNet ONNX model for portrait matting. Processes input image, generates alpha matte, and logs execution time, file sizes, and image shapes using SwanLab.",
        config={
            "model": "MODNet",
            "image_path": None,  # Updated later
            "output_path": None,  # Updated later
            "model_path": None,  # Updated later
            "ref_size": 512,
            "current_step": "initializing",
            "step_index": 0
        }
    )
    swanlab.log({"init_time": time.time() - start_time, "step_index": 0})

    # Step 1: Parse command-line arguments
    step_start_time = time.time()
    swanlab.config.current_step = "parsing_arguments"
    swanlab.config.step_index = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, help='path of the input image (a file)')
    parser.add_argument('--output-path', type=str, help='path for saving the predicted alpha matte (a file)')
    parser.add_argument('--model-path', type=str, help='path of the ONNX model')
    args = parser.parse_args()
    swanlab.config.image_path = args.image_path
    swanlab.config.output_path = args.output_path
    swanlab.config.model_path = args.model_path
    swanlab.log({
        "parse_time": time.time() - step_start_time,
        "step_index": 1
    })

    # Step 2: Check input arguments
    step_start_time = time.time()
    swanlab.config.current_step = "checking_input_paths"
    swanlab.config.step_index = 2
    if not os.path.exists(args.image_path):
        swanlab.config.current_step = "image_path_error"
        swanlab.config.error = f"Cannot find the input image: {args.image_path}"
        print(f'Cannot find the input image: {args.image_path}')
        exit()
    if not os.path.exists(args.model_path):
        swanlab.config.current_step = "model_path_error"
        swanlab.config.error = f"Cannot find the ONNX model: {args.model_path}"
        print(f'Cannot find the ONNX model: {args.model_path}')
        exit()
    image_size = os.path.getsize(args.image_path) / (1024 * 1024)  # File size in MB
    model_size = os.path.getsize(args.model_path) / (1024 * 1024)  # File size in MB
    swanlab.log({
        "image_size_mb": image_size,
        "model_size_mb": model_size,
        "check_paths_time": time.time() - step_start_time,
        "step_index": 2
    })

    # Step 3: Read and preprocess image
    step_start_time = time.time()
    swanlab.config.current_step = "reading_image"
    swanlab.config.step_index = 3
    im = cv2.imread(args.image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    swanlab.log({
        "image_height": im.shape[0],
        "image_width": im.shape[1],
        "image_channels": im.shape[2],
        "step_index": 3
    })

    # Unify image channels to 3
    swanlab.config.current_step = "unifying_channels"
    swanlab.config.step_index = 4
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]
    swanlab.log({"step_index": 4})

    # Normalize values to scale it between -1 to 1
    swanlab.config.current_step = "normalizing_image"
    swanlab.config.step_index = 5
    im = (im - 127.5) / 127.5
    im_h, im_w, im_c = im.shape
    swanlab.log({
        "normalized_height": im_h,
        "normalized_width": im_w,
        "normalized_channels": im_c,
        "step_index": 5
    })

    # Get scale factors and resize image
    swanlab.config.current_step = "resizing_image"
    swanlab.config.step_index = 6
    x, y = get_scale_factor(im_h, im_w, ref_size=512)
    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)
    swanlab.log({
        "resized_height": im.shape[0],
        "resized_width": im.shape[1],
        "scale_factor_x": x,
        "scale_factor_y": y,
        "step_index": 6
    })

    # Prepare input shape
    swanlab.config.current_step = "preparing_input"
    swanlab.config.step_index = 7
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype('float32')
    swanlab.log({
        "input_batch_size": im.shape[0],
        "input_channels": im.shape[1],
        "input_height": im.shape[2],
        "input_width": im.shape[3],
        "preprocess_time": time.time() - step_start_time,
        "step_index": 7
    })

    # Step 4: Initialize session and get prediction
    step_start_time = time.time()
    swanlab.config.current_step = "initializing_session"
    swanlab.config.step_index = 8
    session = onnxruntime.InferenceSession(args.model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    swanlab.log({"step_index": 8})

    swanlab.config.current_step = "running_inference"
    swanlab.config.step_index = 9
    result = session.run([output_name], {input_name: im})
    swanlab.log({
        "output_batch_size": result[0].shape[0],
        "output_channels": result[0].shape[1],
        "output_height": result[0].shape[2],
        "output_width": result[0].shape[3],
        "inference_time": time.time() - step_start_time,
        "step_index": 9
    })

    # Step 5: Refine matte and save
    step_start_time = time.time()
    swanlab.config.current_step = "refining_matte"
    swanlab.config.step_index = 10
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)
    swanlab.log({
        "matte_height": matte.shape[0],
        "matte_width": matte.shape[1],
        "step_index": 10
    })

    swanlab.config.current_step = "saving_matte"
    swanlab.config.step_index = 11
    cv2.imwrite(args.output_path, matte)
    matte_size = os.path.getsize(args.output_path) / (1024 * 1024)  # File size in MB
    swanlab.log({
        "matte_size_mb": matte_size,
        "save_time": time.time() - step_start_time,
        "step_index": 11
    })

    # Step 6: Finish SwanLab run
    total_time = time.time() - start_time
    swanlab.config.current_step = "completed"
    swanlab.config.step_index = 12
    swanlab.log({
        "total_time": total_time,
        "step_index": 12
    })
    swanlab.finish()
