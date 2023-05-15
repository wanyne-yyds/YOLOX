CUDA_VISIBLE_DEVICES=2 python3 tools/eval.py \
    -expn Mobilenetv205033_320x576 \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-Two-C.py \
    -d 1 \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Two_classes_hyh/yolox-mobilenetv2-050_033_2023-05-13_010841/best_ckpt.pth \
    --conf 0.001