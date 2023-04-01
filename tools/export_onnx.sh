#!/bin/bash
modelname="MobilenetV2" # MobilenetV2; GhostNet;
python3 tools/export_onnx.py \
    --input data \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-two_classes.py \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-One_classes/yolox-mobilenetv2-050_033_2023-03-22_152959/best_ckpt.pth \
    # --output bbox8,obj8,cls8,bbox16,obj16,cls16,bbox32,obj32,cls32 \
    # conv_models_deploy \
    # True

# --no-onnxsim \