import cv2
import time
import torch
import argparse
import numpy as np
from yolox.exp import Exp, get_exp

def make_parser():
    parser = argparse.ArgumentParser("YOLOX browse dataset")

    parser.add_argument(
        "-f",
        "--exp_file",
        default="/code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033.py",
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        '--phase',
        '-p',
        default='train',
        type=str,
        choices=['train', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')

    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    
    swap = (1, 2, 0)
    classes_name = ["person", "other"]
    # classes_name = ["person"]
    if args.phase == 'train':
        loader = exp.get_data_loader(batch_size=1, is_distributed=False)
    elif args.phase == 'val':
        loader = exp.get_eval_loader(batch_size=1, is_distributed=False)

    for img, target, img_info, img_id in loader:
        image = torch.squeeze(img)
        image = image.numpy().astype(np.uint8).copy()
        target = torch.squeeze(target)
        images = image.transpose(swap).astype(np.uint8).copy()
        imgname = "./readimg.jpg"
        if len(target.shape) == 1:
            target = target.numpy()[np.newaxis, :]

        for i in range(len(target)):
            if args.phase == "train":
                box = target[i][1:]
                cls = target[i][0]
                x0 = int((box[0] - box[2] * 0.5))
                y0 = int((box[1] - box[3] * 0.5))
                x1 = int((box[0] + box[2] * 0.5))
                y1 = int((box[1] + box[3] * 0.5))
            else:
                box = target[i][:4]
                cls = target[i][4]
                x0 = int((box[0]))
                y0 = int((box[1]))
                x1 = int((box[2]))
                y1 = int((box[3]))

            if np.sum([x0,x1,y0,y1]) == 0:
                continue
            cv2.rectangle(images, (x0, y0), (x1, y1), (0,0,255), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = '{}'.format(classes_name[int(cls)])
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            txt_bk_color = (np.array([0.000, 0.447, 0.741]) * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                images,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            txt_color = (0, 0, 0)
            cv2.putText(images, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        time.sleep(2)
        cv2.imwrite(imgname, images)