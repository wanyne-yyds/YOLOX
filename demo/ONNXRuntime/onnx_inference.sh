CUDA_VISIBLE_DEVICES=0 python3 ./onnx_inference.py \
    -m /code/YOLOX/YOLOX_outputs/val_model/YOLOX-BSD_val/yolox-mobilenetv2-050_033_2023-02-23_211534/yolox-MobilenetV2_deploy-w050-bsd_416x416.onnx \
    -i /code/data/s_BSD/ckn_bsd_cocoformat/JPEGImages/val \
    -s 0.3 \
    -o /code/data/s_BSD/ckn_bsd_cocoformat/obj_detections_hyhdataset_416x416_yolox \
    --input_shape 416,416