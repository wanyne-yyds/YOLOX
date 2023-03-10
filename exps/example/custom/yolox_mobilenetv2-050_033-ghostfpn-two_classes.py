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
        self.num_classes = 1                #* 类别数
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

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 10          #* 工人数量
        self.input_size = (320, 576)        #* 输入尺寸(高, 宽)
        self.multiscale_range = 0           #* 0 关闭多尺度
        self.data_dir = "/code/data/YOLOX-CocoFormat-BSD_One_Classes-v0.0.1-2023-03-10_10:45:54"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.output_dir = "./YOLOX_outputs/YOLOX-BSD-ghostfpn-One_classes"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0              #* mosaic 概率
        self.mixup_prob = 1.0               #* mixup 概率
        self.hsv_prob = 1.0
        self.flip_prob = 0.5                
        self.degrees = 10.0                 #* 旋转角范围 (-2, 2)
        self.translate = 0.1                #* 翻转角范围 (-0.1, 0.1)
        self.mosaic_scale = (0.5, 1.5)      #* mosaic 尺度
        self.enable_mixup = False           #* 关闭 minup
        self.shear = 2.0                    #* 剪切角范围

        # --------------  training config --------------------- #
        self.warmup_epochs = 5              #* 热身
        self.max_epoch = 100                #* 最大 epoch
        self.basic_lr_per_img = 0.01 / 64.0 #* LR
        self.no_aug_epochs = 0              #* 多少 epoch 关闭 mosaic 増强
        self.eval_interval = 10             #* 验证 epoch
        self.print_interval = 100
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (320, 576)         #* 测试尺寸

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOGhostPAN, YOLOXHeadFour

            if self.backbone_net == 'CSPDarknet':
                in_channels = [256, 512, 1024]          # C
                in_features=("dark3", "dark4", "dark5")
            elif self.backbone_net == 'MobileNetV2':
                in_channels = [32, 96, 320]             # M
                in_features = (0, 1, 2)

            backbone = YOLOGhostPAN(self.depth, self.width, in_features, in_channels=in_channels, act=self.act, \
                depthwise=self.depthwise, backbone=self.backbone_net, mobilenet_invertedt=self.mobilenet_invertedt, out_indices=self.out_indices)
            
            in_channels = [32, 32, 32, 32] # ghostfpn 输出: in_c * width
            head = YOLOXHeadFour(self.num_classes, self.width, strides=self.strides, in_channels=in_channels, out_c=32, act=self.act, depthwise=self.depthwise, conv_models_deploy=self.conv_models_deploy)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model