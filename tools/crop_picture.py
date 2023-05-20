import os
import cv2
import time
import os.path as osp
from pathlib import Path

if __name__ == '__main__':

    dataset = "/code/data/s_BSD/hyh_bsd_yoloformat/train_640_0.004_221026_pillar_mirror_check_bbox"
    out = "/code/YOLOX/YOLOX_outputs/temp"

    imgfile = Path(dataset).rglob('*.*g')
    for i, file in enumerate(imgfile):
        img_name = file.name
        file = str(file)
        img = cv2.imread(file)
        label = file.replace('jpg', 'txt').replace('png', 'txt')
        if not osp.exists(label):
            continue
        img_name = img_name.split('.')[0]
        img_name = img_name + '_pillar_crop.jpg'

        img_name = osp.join(out, img_name)
        txt_name = img_name.replace('jpg', 'txt')   
        image_height, image_width, _ = img.shape

        with open(label, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line.split() for line in lines]
            lines = [[int(float(x)) if i == 0 else float(x) for i, x in enumerate(line)] for line in lines]

            for index, (label, x, y, width, height) in enumerate(lines):
                img_name_ = img_name.replace('.jpg', f'_{index}.jpg')
                txt_name_ = txt_name.replace('.txt', f'_{index}.txt')
                if label == 1:
                    # 原始图像坐标
                    x0 = int((x - width / 2) * image_width)
                    y0 = int((y - height / 2) * image_height)
                    x1 = int((x + width / 2) * image_width)
                    y1 = int((y + height / 2) * image_height)

                    # 计算裁剪后的宽度和高度
                    target_width = x1 - x0
                    target_height = y1 - y0

                    # 检查是否需要调整裁剪后的尺寸以适应16:9的范围
                    if target_width / target_height > 16 / 9:
                        target_height = int(target_width / 16 * 9)
                    else:
                        target_width = int(target_height / 9 * 16)

                    # 根据裁剪后的尺寸调整坐标
                    x_min = max(0, x0 - int((target_width - (x1 - x0)) / 2))
                    y_min = max(0, y0 - int((target_height - (y1 - y0)) / 2))
                    x_max = min(image_width, x0 + target_width)
                    y_max = min(image_height, y0 + target_height)

                    # 裁剪图像
                    cropped_image = img[y_min:y_max, x_min:x_max]
                    # 保存裁剪后的图像
                    cv2.imwrite(img_name_, cropped_image)

                    cropped_x_min = max(0, x0 - x_min)
                    cropped_y_min = max(0, y0 - y_min)
                    cropped_x_max = min(target_width, x1 - x_min)
                    cropped_y_max = min(target_height, y1 - y_min)

                    cropped_center_x = (cropped_x_min + (cropped_x_max - cropped_x_min) * 0.5) / target_width
                    cropped_center_y = (cropped_y_min + (cropped_y_max - cropped_y_min) * 0.5) / target_height
                    cropped_width = (cropped_x_max - cropped_x_min) / target_width
                    cropped_height = (cropped_y_max - cropped_y_min) / target_height

                    yolo_x = cropped_center_x
                    yolo_y = cropped_center_y
                    yolo_width = cropped_width
                    yolo_height = cropped_height

                    with open(txt_name_, 'w') as f:
                        yolo_string = f"{label} {yolo_x} {yolo_y} {yolo_width} {yolo_height}"
                        f.write(yolo_string)

    class_path = osp.join(out, 'classes.txt')
    with open(class_path, 'w') as f:
        for name in ['person', 'pillar', 'mirror']:
            f.write(name + '\n')