#!/bin/bash
modelname="MobilenetV2" # MobilenetV2; MobileOne; GhostNet; LiteHrnet
python3 tools/export_onnx.py \
    --output-name /code/YOLOX/YOLOX_outputs/YOLOX-BSD/yolox-mobilenetv2-050_033_2023-02-22_163339/yolox-${modelname}_deploy-w050-bsd_416x416.onnx \
    --input data \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033_ckn.py \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD/yolox-mobilenetv2-050_033_2023-02-22_163339/best_ckpt.pth \
# --no-onnxsim \
# --output bbox8,obj8,cls8,bbox16,obj16,cls16,bbox32,obj32,cls32 \