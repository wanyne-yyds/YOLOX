#!/bin/bash
jsonfile="/code/YOLOX/yolox_testdev_2017.json"
# arrayfile=""
if [ -f ${jsonfile} ];then
echo "delete ${jsonfile}"
rm -f ${jsonfile}
else
echo "${jsonfile} not exist"
fi
# if [ -f ${arrayfile} ];then
# echo "delete ${arrayfile}"
# rm -f ${arrayfile}
# else
# echo "${arrayfile} not exist"
# fi
current_time=$(date "+%Y-%m-%d_%H_%M_%S")
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train.py \
    -expn yolox-mobilenetv2-050_033 \
    -d 2 \
    -b 64 \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033.py \
    -o \
    --logger wandb \
        wandb-project YOLOX-BSD \
        wandb-name YOLOX-MobilenetV2-050_033-hyhdatset-$current_time \
        wandb-save_dir YOLOX_outputs \

# --fp16 \
# -c /home/ckn/Code/YOLOX-0.3.0-HOD/YOLOX_outputs/yolox_nano_customize_hod_RK_8/2022_12_31_09_46_38_MobileNetV2-0.5/best_ckpt.pth \
# --cache \