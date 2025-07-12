#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default="test_image.png",
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="demo_output",
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(",")))
    print("Input shape:", input_shape)
    origin_img = cv2.imread(args.image_path)
    print("Original image shape:", origin_img.shape)
    print("Original image sample:", origin_img[0, 0])
    img, ratio = preprocess(origin_img, input_shape, swap=(0, 1, 2))
    print("Input image shape:", img.shape)
    print("Input image sample:", img[0, 0])
    print("Input image ratio:", ratio)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    print("Output shape:", output[0].shape)
    print("Output sample:", output[0][0][0])
    predictions = demo_postprocess(output[0], input_shape)[0]
    print("Predictions shape:", predictions.shape)
    print("Predictions sample:", predictions[0])

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    print("predictions[:, 4:5]: ", predictions[:, 4:5])
    print("Boxes shape:", boxes.shape)
    print("Scores shape:", scores.shape)
    print("Scores sample:", scores[0])

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    print("Boxes shape:", boxes_xyxy.shape)
    print("Boxes sample:", boxes_xyxy[0])
    print(
        "Boxes sample with [3] > 130 and [2] > 102:",
        boxes_xyxy[(boxes_xyxy[:, 3] > 130) & (boxes_xyxy[:, 2] > 102)],
    )

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    print("Detections shape:", dets.shape)
    print("Detections sample:", dets[0] if dets is not None else "No detections")
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(
            origin_img,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=args.score_thr,
            class_names=COCO_CLASSES,
        )

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
