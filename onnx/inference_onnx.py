"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
    python -m onnx.inference_onnx \
        --image-path=logo.jpg --output-path=matte.png \
        --model-path=pretrained/modnet_photographic_portrait_matting.onnx
"""

import os
import argparse

import cv2
import numpy as np
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True, help='path of the input image (a file)')
    parser.add_argument('--output-path', type=str, required=True, help='path for saving the predicted alpha matte (a file)')
    parser.add_argument('--model-path', type=str, required=True, help='path of the ONNX model')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f'Cannot find the input image: {args.image_path}')
        exit()
    if not os.path.exists(args.model_path):
        print(f'Cannot find the ONNX model: {args.model_path}')
        exit()

    # Read image
    im = cv2.imread(args.image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # Normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5
    im_h, im_w, im_c = im.shape

    # Resize image to a size that the model accepts
    x, y = get_scale_factor(im_h, im_w, ref_size=512)
    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    # Prepare input shape: (H, W, C) -> (1, C, H, W)
    im = im.transpose(2, 0, 1)[np.newaxis, ...].astype('float32')

    # Initialize session and run inference
    session = onnxruntime.InferenceSession(args.model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # Refine matte and save
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.output_path, matte)
    print(f'Alpha matte saved to: {args.output_path}')
