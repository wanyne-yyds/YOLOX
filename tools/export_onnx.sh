#!/bin/bash
modelname="MobilenetV2" # MobilenetV2; GhostNet;
python3 tools/export_onnx.py \
    --input data \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-One-C.py \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-One_classes_hyh/yolox-mobilenetv2-050_033_2023-05-12_092911/best_ckpt.pth \
    --output bbox8,obj8,cls8,bbox16,obj16,cls16,bbox32,obj32,cls32,bbox64,obj64,cls64 \
    conv_models_deploy \
    True
    # --output seg_output,det_output \

# --no-onnxsim \