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
CUDA_VISIBLE_DEVICES=1,2,3 python3 tools/train.py \
    -expn yolox-mobilenetv2-050_033 \
    -d 3 \
    -b 64 \
    -f /code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-Three-C.py \
    -c /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Three_classes_hyh/yolox-mobilenetv2-050_033_2023-05-14_223317/best_ckpt.pth \
    -o \
    --logger wandb \
        wandb-project YOLOX-BSD-Hyh \
        wandb-name YOLOX-MobilenetV2-050_033-hyhdatset-multistep-SGD-Three-320x576-$current_time \
        wandb-save_dir YOLOX_outputs/YOLOX-BSD-ghostfpn-Three_classes_hyh \
