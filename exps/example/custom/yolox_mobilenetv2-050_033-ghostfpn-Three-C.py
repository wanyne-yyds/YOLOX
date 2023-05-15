#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import time
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        self.depth = 0.33
        self.width = 0.50
        self.act = 'LeakyReLU'              #* 激活函数
        self.num_classes = 3                #* 类别数
        self.backbone_net = "MobileNetV2"   #? 网络选择(MobileNetV2, CSPDarknet)
        self.out_indices = [6, 12, 16]
        self.mobilenet_invertedt = [
                # t, c, n, s
                [1, 16, 1, 1],
                [4, 24, 2, 2],
                [4, 32, 3, 2],
                [2, 64, 3, 2],
                [2, 96, 3, 1],
                [2, 160, 3, 2],
                [4, 320, 1, 1],
                                   ]
        self.conv_models_deploy = False
        self.depthwise = False
        self.strides = [8, 16, 32, 64]
        self.Head_in_channels = [32, 32, 32, 32] # ghostfpn 输出: in_c * width
        self.Head_out_channels = 32
        self.reg_iou_type = 'iou'

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 10          #* 工人数量
        self.input_size = (320, 576)        #* 输入尺寸(高, 宽)
        self.multiscale_range = 0           #* 0 关闭多尺度
        self.data_dir = "/code/data/YOLOX-Yolo2CocoFormat-BSD_Three_Classes-2023-05-15_11:12"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.output_dir = "./YOLOX_outputs/YOLOX-BSD-ghostfpn-Three_classes_hyh"

        # --------------- transform config ----------------- #
        self.mosaic_prob = -1               #* mosaic 概率
        self.mixup_prob = -1                #* mixup 概率
        self.hsv_prob = 1.0
        self.flip_prob = 0.5                
        self.degrees = 10.0                 #* 旋转角范围 (-2, 2)
        self.translate = 0.1                #* 翻转角范围 (-0.1, 0.1)
        self.mosaic_scale = (0.5, 1.5)      #* mosaic 尺度
        self.enable_mixup = False           #* 关闭 minup
        self.shear = 2.0                    #* 剪切角范围

        # --------------  training config --------------------- #
        self.warmup_epochs = 1              #* 热身
        self.max_epoch = 80                #* 最大 epoch
        self.basic_lr_per_img = 0.0001 / 64.0 #* LR (SGD: 0.01; AdamW: 0.004)
        self.no_aug_epochs = -1             #* 多少 epoch 关闭 mosaic 増强
        self.no_change_epochs = 60          #* 多少 epoch 更换训练数据扩展
        self.eval_interval = 10             #* 验证 epoch
        self.print_interval = 500           #* 打印间隔
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.weight_decay = 5e-4            #* SGD: 5e-4; AdamW: 0.05
        self.optim_type = "SGD"             #* SGD, AdamW
        self.scheduler = "yoloxwarmcos"     #* 默认：yoloxwarmcos
        # self.milestones = [50, 100, 160]    #* 学习率下降的 epoch

        self.train_transform_0 = [
            "PadToAspectRatio(w2h_ratio=1.8, border_mode=0)",
            "PadIfNeeded(min_height=320, min_width=576, border_mode=0, p=1)",
            "RandomCopyBbox(p=0.2, bbox_min_scale=0.008, scale=(0.7, 1.5))",
            "CropAndPad(percent=(-0.1, 0.1), keep_size=True, sample_independently=True, p=0.5)",
            "MyGridDropout(ratio_range=(0.1, 0.5), holes_number=(1, 7), random_offset=True, p=0.05)",
            "OneOf([\
                MyRotate(limit=40, border_mode=0, p=0.1, adjust_factor=0.35),\
                MyRotate(limit=70, border_mode=0, p=0.1, adjust_factor=0.35),\
                ], p=0.1)",
            "OneOf([\
                MyRandomSizedBBoxSafeCrop(height=320, width=576, erosion_rate=0.0, scale=(0.1, 1), ratio=(1.4, 2.2), p=0.3, alignment='random'),\
                MyRandomSizedBBoxSafeCrop(height=320, width=576, erosion_rate=0.0, scale=(0.1, 1), ratio=(1.7, 1.9), p=0.3, alignment='random'),\
                RandomEraseBbox(scale=(0.65, 0.8), p=0.3, p_erase_all=1),\
                ], p=0.2)",
            "OneOf([\
                RandomCropIfEmpty(height=320, width=576, scale=(0.001, 1), ratio=(0.01, 100.0), p=0.2),\
                RandomCropIfEmpty(height=320, width=576, scale=(0.25, 1), ratio=(1.0, 2.5), p=0.2),\
                ], p=0.3)",
            "RandomEraseBbox(scale=(0.6, 0.75), p=0.2)",
            "OneOf([\
                Resize(p=0.3, height=320, width=576, interpolation=cv2.INTER_NEAREST),\
                Resize(p=0.4, height=320, width=576, interpolation=cv2.INTER_LINEAR),\
                Resize(p=0.1, height=320, width=576, interpolation=cv2.INTER_CUBIC),\
                Resize(p=0.05, height=320, width=576, interpolation=cv2.INTER_LANCZOS4),\
                Resize(p=0.05, height=320, width=576, interpolation=cv2.INTER_AREA),\
            ], p=1)",
            "ToGray(p=0.1)",
            "OneOf([\
                RandomBrightnessContrast(p=0.1, brightness_limit=(-0.3, 0), contrast_limit=(-0.3, 0), brightness_by_max=False),\
                ColorJitter(p=0.2, brightness=0.3, contrast=0.5, saturation=0, hue=0),\
                ColorJitter(p=0.2, brightness=0.3, contrast=0.5, saturation=0.1, hue=0.1),\
                ], p=0.4)",
            "OneOf([\
                Downscale(scale_min=0.8, scale_max=0.97, p=0.2),\
                JpegCompression(quality_lower=40, quality_upper=95, p=0.8),\
                GaussNoise(var_limit=(10.0, 50.0), p=0.2),\
                ], p=0.02)",
            "OneOf([\
                MotionBlur(blur_limit=(3, 5), p=0.6),\
                GaussianBlur(blur_limit=(3, 5), sigma_limit=3, p=0.8),\
                MedianBlur(blur_limit=(3, 5), p=0.2),\
                Blur(blur_limit=(3, 5), p=0.2),\
                ], p=0.05)",
            "HorizontalFlip(p=0.5)",
        ]

        self.train_transform_1 = [
            "PadToAspectRatio(w2h_ratio=1.8, border_mode=0)",
            "PadIfNeeded(min_height=320, min_width=576, border_mode=0, p=1)",
            "CropAndPad(percent=(-0.1, 0.1), keep_size=True, sample_independently=True, p=0.3)",
            "OneOf([\
                MyRandomSizedBBoxSafeCrop(height=320, width=576, erosion_rate=0.0, scale=(0.1, 1), ratio=(1.4, 2.2), p=0.2, union_of_bboxes=True, alignment='random'),\
                MyRandomSizedBBoxSafeCrop(height=320, width=576, erosion_rate=0.0, scale=(0.1, 1), ratio=(1.7, 1.9), p=0.4, union_of_bboxes=True, alignment='random'),\
                ], p=0.1)",
            "OneOf([\
                Resize(p=0.2, height=320, width=576, interpolation=cv2.INTER_NEAREST),\
                Resize(p=0.4, height=320, width=576, interpolation=cv2.INTER_LINEAR),\
                Resize(p=0.1, height=320, width=576, interpolation=cv2.INTER_CUBIC),\
                Resize(p=0.05, height=320, width=576, interpolation=cv2.INTER_LANCZOS4),\
                Resize(p=0.05, height=320, width=576, interpolation=cv2.INTER_AREA),\
            ], p=1)",
            "OneOf([\
                MyRotate(limit=30, border_mode=0, p=0.1, adjust_factor=0.35),\
                ], p=0.1)",
            "OneOf([\
                ColorJitter(p=0.3, brightness=0.3, contrast=0.5, saturation=0, hue=0),\
                ColorJitter(p=0.2, brightness=0.3, contrast=0.5, saturation=0.1, hue=0.1),\
                ], p=0.4)",
            "HorizontalFlip(p=0.5)",
        ]

        self.val_transform = [
            "PadToAspectRatio(w2h_ratio=1.8, border_mode=0)",
            "Resize(p=1, height=320, width=576)",
        ]

        # -----------------  testing config ------------------ #
        self.test_size = (320, 576)         #* 测试尺寸

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOGhostPAN, YOLOXHeadFour_Fast

            if self.backbone_net == 'CSPDarknet':
                in_channels = [256, 512, 1024]          # C
                in_features=("dark3", "dark4", "dark5")
            elif self.backbone_net == 'MobileNetV2':
                in_channels = [32, 96, 320]             # M
                in_features = (0, 1, 2)

            backbone = YOLOGhostPAN(self.depth, self.width, in_features, in_channels=in_channels, act=self.act, \
                depthwise=self.depthwise, backbone=self.backbone_net, mobilenet_invertedt=self.mobilenet_invertedt, out_indices=self.out_indices)
            
            head = YOLOXHeadFour_Fast(self.num_classes, self.width, strides=self.strides, in_channels=self.Head_in_channels, out_c=self.Head_out_channels, \
                                 act=self.act, depthwise=True, iou_type=self.reg_iou_type, conv_models_deploy=self.conv_models_deploy)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model