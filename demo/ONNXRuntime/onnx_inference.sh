CUDA_VISIBLE_DEVICES=0 python3 ./onnx_inference.py \
    -m /code/YOLOX/demo/ONNXRuntime/yolox_2023-04-28_145229.onnx \
    -i /code/data/s_BSD/ckn_bsd_cocoformat_1/JPEGImages/train \
    -s 0.3 \
    -o /code/data/s_BSD/ckn_bsd_cocoformat_1/ObjResult/obj_pre_One_train_dataset_2023-04-28_193642 \
    --input_shape 320,576