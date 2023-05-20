CUDA_VISIBLE_DEVICES=0 python3 tools/demo.py \
    image \
    -expn YOLOX-BSD-Pre \
    --path /code/data/s_BSD/BSD-Third-calibration-for-large-test-set/Universal-type/Front20220509/20220509Front \
    --exp_file /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-Three-C.py \
    --ckpt /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Three_classes_hyh/yolox-mobilenetv2-050_033_2023-05-17_195843/best_ckpt.pth \
    --save_result \
    --device gpu \
    --conf 0.3 \
    --nms 0.45 \
    # --fp16 \
    # --fuse \