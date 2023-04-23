#!/bin/bash
modelname="MobilenetV2" # MobilenetV2; GhostNet;
python3 tools/export_onnx.py \
    --input data \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-Two-C.py \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Two_classes/yolox-mobilenetv2-050_033_2023-04-18_095933/best_ckpt.pth \
    --output bbox8,obj8,cls8,bbox16,obj16,cls16,bbox32,obj32,cls32,bbox64,obj64,cls64 \
    conv_models_deploy \
    True

# --no-onnxsim \