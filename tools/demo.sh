CUDA_VISIBLE_DEVICES=0 python3 tools/demo.py \
    video \
    -expn YOLOX-BSD-Pre \
    --path /code/YOLOX/pillar.mp4 \
    --exp_file /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-One-C.py \
    --ckpt /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-One_classes_hyh/yolox-mobilenetv2-050_033_2023-05-12_092911/best_ckpt.pth \
    --save_result \
    --device gpu \
    --conf 0.3 \
    --nms 0.45 \
    # --fp16 \
    # --fuse \