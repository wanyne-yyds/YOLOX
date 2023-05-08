CUDA_VISIBLE_DEVICES=2 python3 tools/demo.py \
    image \
    -expn YOLOX-BSD-Pre \
    --path /code/data/s_BSD/BSD-Third-calibration-for-large-test-set/FishEye-type/FishEyeCameraIMG \
    --exp_file /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-Two-C.py \
    --ckpt /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Two_classes/yolox-mobilenetv2-050_033_2023-04-08_114437/best_ckpt.pth \
    --save_result \
    --device gpu \
    --conf 0.2 \
    --nms 0.45 \
    # --fp16 \
    # --fuse \