python3 tools/demo.py \
    image \
    -expn YOLOX-BSD-Pre \
    --path /code/data/s_BSD/ckn_bsd_cocoformat/JPEGImages/train/0318/13000000000_ch02_CeMang_1646902886229.jpg \
    --exp_file exps/example/custom/yolox_mobilenetv2-050_033.py \
    --ckpt YOLOX_outputs/YOLOX-BSD/yolox-mobilenetv2-050_033_2023-02-20_174102/best_ckpt.pth \
    --save_result \
    --device cpu \
    --conf 0.3 \
    --nms 0.3 \
    --tsize 416 \
    # --fp16 \
    # --fuse \