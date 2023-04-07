CUDA_VISIBLE_DEVICES=0 python3 ./onnx_inference.py \
    -m /code/YOLOX/YOLOX_outputs/YOLOX-BSD-ghostfpn-Two_classes/yolox-mobilenetv2-050_033_2023-04-03_204224/yolox.onnx \
    -i /code/data/s_BSD/hyh_bsd_yoloformat/test_Normal_0.004/images \
    -s 0.3 \
    -o /code/data/s_BSD/hyh_bsd_yoloformat/ObjResult/YOLOX-NewDataSet-320x576 \
    --input_shape 320,576