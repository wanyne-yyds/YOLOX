CUDA_VISIBLE_DEVICES=3 python3 ./tests/utils/test_wandb.py \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-two_classes.py \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-One_classes/yolox-mobilenetv2-050_033_2023-03-10_105541/best_ckpt.pth \
    -d 1 \
    -l wandb \
        wandb-project \
        YOLOX-BSD-Two-classes-temp \
        wandb-name \
        YOLOX-MobilenetV2-050_033-ckndatset-Changetheratio-320x576-$current_time \
        wandb-save_dir \
        YOLOX_outputs \