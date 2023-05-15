CUDA_VISIBLE_DEVICES=0 python3 ./onnx_inference.py \
    -m /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Three_classes_hyh/yolox-mobilenetv2-050_033_2023-05-14_223317/yolox.onnx \
    -i /code/data/s_BSD/hyh_bsd_yoloformat/train_640_0.004_221026_pillar_mirror_check_bbox \
    -o /code/data/s_BSD/hyh_bsd_yoloformat/ObjResult/yolox-mobilenetv2-050_033_2023-05-14_223317_pillar_mirror_check_bbox \
    -s 0.3 \
    --input_shape 320,576