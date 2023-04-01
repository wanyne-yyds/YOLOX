CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py \
    -expn Mobilenetv205033_320x576 \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-two_classes.py \
    -d 1 \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-One_classes/yolox-mobilenetv2-050_033_2023-03-22_152959/best_ckpt.pth \
    --conf 0.001