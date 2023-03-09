#!/bin/bash
modelname="MobilenetV2" # MobilenetV2; GhostNet;
python3 tools/export_onnx.py \
    --input data \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-two_classes.py \
    --output bbox8,obj8,cls8,bbox16,obj16,cls16,bbox32,obj32,cls32 \
    conv_models_deploy \
    True



# -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD/yolox-mobilenetv2-050_033_2023-02-22_163339/best_ckpt.pth \
# --no-onnxsim \