CUDA_VISIBLE_DEVICES=3 python3 tools/eval.py \
    -expn Mobilenetv205033_416x416 \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033.py \
    -d 1 \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD/yolox-mobilenetv2-050_033_2023-02-22_163339/best_ckpt.pth \
    --conf 0.001