python3 tools/demo.py \
    image \
    -expn YOLOX-BSD-Pre \
    --path /code/data/s_BSD/ckn_bsd_cocoformat/JPEGImages/train/0318 \
    --exp_file /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-two_classes.py \
    --ckpt /code/YOLOX/YOLOX_outputs/YOLOX-BSD-Two_classes/yolox-mobilenetv2-050_033_2023-03-09_192315/best_ckpt.pth \
    --save_result \
    --device cpu \
    --conf 0.3 \
    --nms 0.3 \
    # --fp16 \
    # --fuse \