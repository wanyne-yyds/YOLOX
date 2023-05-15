#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import time
import numpy as np
from pathlib import Path
import shutil
import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

COCO_CLASSES = (
    "person",
    "pillar",
    "mirror",
)


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
        default="320,576",
        help="Specify an input shape for inference.",
    )
    return parser


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    images_path = Path(args.image_path).rglob("*.*g")
    for file_num, imgpath in enumerate(images_path):
        imgpath = str(imgpath)
        origin_img = cv2.imread(imgpath)
        img, ratio = preprocess(origin_img, input_shape)

        session = onnxruntime.InferenceSession(args.model)
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=True)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.3, score_thr=0.1)

        # ? 保存画 PR 曲线的 txt 文件
        output_path = imgpath.replace(args.image_path, args.output_dir).replace(
            'jpg', 'txt').replace('png', 'txt')
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        with open(output_path, 'w') as f:
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                 conf=args.score_thr, class_names=COCO_CLASSES)

                for index, box in enumerate(final_boxes):
                    scores_i = final_scores[index]
                    if scores_i < args.score_thr:
                        continue
                    xmin, ymin, xmax, ymax = box
                    xmin, ymin, xmax, ymax = map(lambda x: str(
                        round(x + 0.5)), (xmin, ymin, xmax, ymax))
                    cls_ = COCO_CLASSES[int(final_cls_inds[index])]
                    scores_s = '%.2f' % scores_i
                    strxy = cls_+" "+scores_s+" "+xmin+" "+ymin+" "+xmax+" "+ymax
                    f.writelines(strxy+'\n')

        # ? 保存 labelimg 格式的 txt 文件
        # height, width, _ = origin_img.shape
        # txtpath = imgpath.replace('jpg', 'txt').replace('png', 'txt')
        # lines = list()
        # if os.path.exists(txtpath):
        #     with open(txtpath, 'r') as f:
        #         lines = f.readlines()

        # output_path = imgpath.replace(args.image_path, args.output_dir).replace(
        #     'jpg', 'txt').replace('png', 'txt')
        # os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        # with open(output_path, 'w') as f:
        #     if dets is not None:
        #         final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

        #         for index, box in enumerate(final_boxes):
        #             scores_i = final_scores[index]
        #             if scores_i < args.score_thr:
        #                 continue
        #             if int(final_cls_inds[index]) != 2:
        #                 continue
        #             xmin, ymin, xmax, ymax = box

        #             cx, cy, wx, hy = convert((width, height), (float(
        #                 xmin), float(xmax), float(ymin), float(ymax)))
        #             cx, cy, wx, hy = map(lambda x: "%.6f"%x, (cx, cy, wx, hy))
        #             strxy = str(1) + " " + cx + " " + cy + " " + wx + " " + hy
        #             f.writelines(strxy+'\n')
        #             for content in lines:
        #                 f.writelines(content)
        # shutil.copy(imgpath, os.path.split(output_path)[0])


        #? 保存图片
        # if dets is not None:
        #     save_img = False
        #     final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        #     for index, box in enumerate(final_boxes):
        #         scores_i = final_scores[index]
        #         if scores_i < args.score_thr:
        #             continue
        #         xmin, ymin, xmax, ymax = box
        #         xmin, ymin, xmax, ymax = map(lambda x: int(
        #             round(x + 0.5)), (xmin, ymin, xmax, ymax))
        #         cls_ = COCO_CLASSES[int(final_cls_inds[index])]
        #         scores_s = '%.2f' % scores_i
        #         if cls_ != "mirror":
        #             continue
                
        #         save_img=True

        #         text = '{}:{:.1f}%'.format(cls_, float(scores_s) * 100)
        #         txt_color = (0, 255, 123)
        #         font = cv2.FONT_HERSHEY_SIMPLEX

        #         txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        #         cv2.rectangle(origin_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)

        #         txt_bk_color = [255, 255, 255]
        #         cv2.rectangle(
        #             origin_img,
        #             (xmin, ymin + 1),
        #             (xmin + txt_size[0] + 1, ymin + int(1.5*txt_size[1])),
        #             txt_bk_color,
        #             -1
        #         )
        #         cv2.putText(origin_img, text, (xmin, ymin + txt_size[1]), font, 0.4, txt_color, thickness=1)
            
        #     if save_img:
        #         save_path="/code/YOLOX/demo/ONNXRuntime/fisheye/0.5/onnx_%d.jpg"%file_num
        #         cv2.imwrite(save_path, origin_img)
