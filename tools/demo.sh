python3 tools/demo.py \
    image \
    -expn YOLOX-BSD-Pre \
    --path /code/data/YOLOX-CocoFormat-BSD_Two_Classes-v0.0.1-2023-03-11_12:07:00/val2017 \
    --exp_file /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-Two-C.py \
    --ckpt /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Two_classes/yolox-mobilenetv2-050_033_2023-04-03_204224/best_ckpt.pth \
    --save_result \
    --device cpu \
    --conf 0.3 \
    --nms 0.3 \
    # --fp16 \
    # --fuse \