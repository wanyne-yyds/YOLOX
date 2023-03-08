#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import time
import numpy as np
from pathlib import Path

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
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
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


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    images_path = Path(args.image_path).rglob("*.*g")
    for imgpath in images_path:
        imgpath = str(imgpath)
        origin_img = cv2.imread(imgpath)
        img, ratio = preprocess(origin_img, input_shape)

        session = onnxruntime.InferenceSession(args.model)
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy[:, 0] /= ratio[1]
        boxes_xyxy[:, 1] /= ratio[0]
        boxes_xyxy[:, 2] /= ratio[1]
        boxes_xyxy[:, 3] /= ratio[0]

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        mkdir(args.output_dir)
        output_path = os.path.join(args.output_dir, os.path.basename(imgpath))
        output_txt = output_path.replace('jpg', 'txt').replace('png', 'txt')
        with open(output_txt, 'w') as f:
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                conf=args.score_thr, class_names=COCO_CLASSES)

                for index, box in enumerate(final_boxes):
                    scores_i = final_scores[index]
                    if scores_i < args.score_thr:
                        continue
                    xmin, ymin, xmax, ymax = box
                    xmin, ymin, xmax, ymax = map(lambda x: str(round(x + 0.5)), (xmin, ymin, xmax, ymax))
                    cls_ = COCO_CLASSES
                    scores_s = '%.2f'%scores_i
                    strxy = cls_+" "+scores_s+" "+xmin+" "+ymin+" "+xmax+" "+ymax
                    f.writelines(strxy+'\n')
        # cv2.imwrite(output_path, origin_img)
        # cv2.imwrite('./onnx.jpg', origin_img)
        # time.sleep(1)
