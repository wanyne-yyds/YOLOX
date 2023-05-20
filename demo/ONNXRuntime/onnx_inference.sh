CUDA_VISIBLE_DEVICES=0 python3 ./onnx_inference.py \
    -m /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Three_classes_hyh/yolox-mobilenetv2-050_033_2023-05-14_223317/yolox.onnx \
    -i /code/YOLOX/YOLOX_outputs/temp/642test.jpg \
    -o /code/data/s_BSD/hyh_bsd_yoloformat/ObjResult/yolox-mobilenetv2-050_033_2023-05-14_223317_FishEye-type \
    -s 0.3 \
    --input_shape 320,576 \
    --save_result \
    # --save_pr_curve \
    # --save_labelimg_txt \
