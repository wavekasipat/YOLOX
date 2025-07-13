#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


# Define a wrapper class to handle input permutation
class InputPermuter(nn.Module):
    """
    A wrapper module that permutes the input tensor from channels-last (NHWC)
    to channels-first (NCHW) before passing it to the wrapped model.
    This makes the exported ONNX model expect NHWC input, while the internal
    PyTorch model (like YOLOX) can still process NCHW.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Input 'x' is expected to be [B, H, W, C] (channels-last) from the ONNX runtime.
        # Permute it to [B, C, H, W] (channels-first) for the original PyTorch model.
        x_channels_first = x.permute(0, 3, 1, 2)

        # Pass the channels-first tensor to the original YOLOX model
        output = self.model(x_channels_first)

        return output


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="whether the input shape should be dynamic or not",
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference", action="store_true", help="decode in inference or not"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    # --- FIX START: Explicitly set weights_only=False to allow loading ---
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # --- FIX END ---

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference
    print("args.decode_in_inference:", args.decode_in_inference)

    logger.info("loading checkpoint done.")

    # --- IMPORTANT CHANGE START ---
    # Wrap the original YOLOX model with the InputPermuter
    # model = InputPermuter(model)

    # The dummy input for ONNX export should now be in the *desired* channels-last format
    # because the InputPermuter will expect it this way.
    # The dimensions are (Batch, Height, Width, Channels)
    # dummy_input = torch.randn(args.batch_size, exp.test_size[0], exp.test_size[1], 3)
    # --- IMPORTANT CHANGE END ---

    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    print("Dummy input shape:", dummy_input.shape)

    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes=(
            {args.input: {0: "batch"}, args.output: {0: "batch"}}
            if args.dynamic
            else None
        ),
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        # use onnx-simplifier to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
