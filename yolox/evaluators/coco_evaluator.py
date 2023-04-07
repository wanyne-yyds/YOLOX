#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np

import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self, model, save_dir, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        pr_gt = []
        pr_dt = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, reg, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            gt_, dt_ = self.convert_to_pr_format(outputs, info_imgs, reg)
            pr_gt.extend(gt_)
            pr_dt.extend(dt_)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        pr_content = self.process_batch(pr_dt, pr_gt, save_dir)
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results, pr_content

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]
        
            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            cls = output[:, 6]
            if len(self.dataloader.dataset.class_ids) == 4:
                cls[cls==1] = 0
                
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            # if self.testdev:
            json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
            cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            # else:
            #     _, tmp = tempfile.mkstemp()
            #     json.dump(data_dict, open(tmp, "w"))
            #     cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info

    def convert_to_pr_format(self, outputs, info_imgs, targets):
        data_list = []
        gt_list = []
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)
        for index, (output, img_h, img_w) in enumerate(zip(
            outputs, info_imgs[0], info_imgs[1])
        ):
            if output is None:
                continue
            device = output.device
            output = output.cpu()

            bboxes = output[:, 0:4]
            num_gt = int(nlabel[index])
            scale = self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            bboxes[:, 0] /= scale[1]
            bboxes[:, 2] /= scale[1]
            bboxes[:, 1] /= scale[0]
            bboxes[:, 3] /= scale[0]    #! xyxy format
            # bboxes = xyxy2xywh(bboxes)
            if num_gt == 0:
                gts = targets.new_zeros((0, 5))
            else:
                gt_bboxes = targets[index, :num_gt, 0:4]
                gt_classes = targets[index, :num_gt, 4].reshape(-1, 1)
                gt_bboxes[:, 0] /= scale[1]
                gt_bboxes[:, 2] /= scale[1]
                gt_bboxes[:, 1] /= scale[0]
                gt_bboxes[:, 3] /= scale[0]    #! xyxy format
                gts = torch.cat((gt_classes, gt_bboxes), 1) #* cls,xyxy

            cls = output[:, 6].reshape(-1, 1)
            scores = (output[:, 4] * output[:, 5]).reshape(-1, 1)
            per = torch.cat((bboxes, scores, cls), 1)
            data_list.append(per.to(device))
            gt_list.append(gts.to(device))

        return gt_list, data_list

    def process_batch(self, detections, labels, save_dir):
        def box_iou(box1, box2):
            def box_area(box):
                # box = xyxy(4,n)
                return (box[2] - box[0]) * (box[3] - box[1])
            # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
            (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
            inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

            # IoU = inter / (area1 + area2 - inter)
            return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)

        device = detections[0].device
        jdict, stats, ap, ap_class = [], [], [], []
        iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        for id, pr in enumerate(detections):
            nl = len(labels[id])
            tcls = labels[id][:, 0].tolist() if nl else []  # target class
            if len(pr) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            if nl:
                correct = torch.zeros(pr.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
                iou = box_iou(labels[id][:, 1:], pr[:, :4])
                x = torch.where((iou >= iouv[0]) & (labels[id][:, 0:1] == pr[:, 5]))  # IoU above threshold and classes match
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    matches = torch.from_numpy(matches).to(iouv.device)
                    correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
            else:
                correct = torch.zeros(pr.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pr[:, 4].cpu(), pr[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        # Compute metrics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        tp, fp, p, r, f1, ap, ap_class = self.ap_per_class(*stats, plot=True, save_dir=save_dir)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        info = "\n metrics/precision: %f; metrics/recall: %f; metrics/mAP_0.5: %f; metrics/mAP_0.5:0.95: %f \n"%(mp, mr, map50, map)
        return info

    def ap_per_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir='.', eps=1e-16):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + eps)  # recall curve
                r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0:
                        py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + eps)
        names = {k: v for k, v in enumerate(COCO_CLASSES)}
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        names = {i: v for i, v in enumerate(names)}  # to dict
        if plot:
            self.plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
            self.plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
            self.plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
            self.plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

        i = f1.mean(0).argmax()  # max F1 index
        p, r, f1 = p[:, i], r[:, i], f1[:, i]
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def plot_pr_curve(self, px, py, ap, save_dir='pr_curve.png', names=()):
        # Precision-recall curve
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)

        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py.T):
                ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
        else:
            ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

        ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(Path(save_dir), dpi=250)
        plt.close()

    def plot_mc_curve(self, px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
        # Metric-confidence curve
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py):
                ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
        else:
            ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

        y = py.mean(0)
        ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(Path(save_dir), dpi=250)
        plt.close()