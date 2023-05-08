#!/bin/bash
modelname="MobilenetV2" # MobilenetV2; GhostNet;
python3 tools/export_onnx.py \
    --input data \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-One-C.py \
    --output output \
    # conv_models_deploy \
    # True
    # --output seg_output,det_output \

# --no-onnxsim \