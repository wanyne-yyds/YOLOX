CUDA_VISIBLE_DEVICES=0 python3 ./onnx_inference.py \
    -m /code/YOLOX/YOLOX_outputs/YOLOX-BSD/yolox-mobilenetv2-050_033_2023-02-22_163339/yolox-MobilenetV2_deploy-w050-bsd_416x416.onnx \
    -i /code/data/s_BSD/hyh_bsd_yoloformat/test_Normal_0.004 \
    -s 0.3 \
    -o /code/data/s_BSD/hyh_bsd_yoloformat/obj_detections_hyh_416x416_yolox \
    --input_shape 416,416