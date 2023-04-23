import json
import numpy as np
import cv2
import os
import uuid
import math
import numpy as np
import PIL.Image
import PIL.ImageDraw
from pathlib import Path
from matplotlib import pyplot as plt 
from typing import List, Tuple
import time

def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    points_list_show = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        points_list_show.append(points)
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins, points_list_show

def labelme_shapes_to_label(img_shape, shapes):
    label_name_to_value = {"_background_": 0}
    newshape = list()
    for shape in shapes:
        label_name = shape["label"]
        if not label_name.endswith("_seg"):
            continue
        newshape.append(shape)
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    lbl, _, points_list = shapes_to_label(img_shape, newshape, label_name_to_value)
    return lbl, label_name_to_value, points_list

def reduce_polygon(polygon: np.array, angle_th: int = 0, distance_th: int = 0) -> np.array(List[int]):
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = list()
        for i in range(0, len(polygon)-2, 2):
            v01 = polygon[i-1] - polygon[i]
            v12 = polygon[i] - polygon[i+1]
            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)
            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i)
                continue
                angle = np.arccos(np.sum(v01*v12) / (d01 * d12))
                if angle < angle_th_rad:
                    points_removed.append(i)
        polygon = np.delete(polygon, points_removed, axis=0)
    return polygon
    
def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)
   
def show_result_reducing(polygon: List[List[int]]) -> List[Tuple[int, int]]:
    polygon_list = []
    for i in range(len(polygon)):
        original_polygon = np.array([[x, y] for x, y in polygon[i]])
        tic = time.time()
        reduced_polygon = reduce_polygon(original_polygon, angle_th=1, distance_th=20)
        toc = time.time()

        fig = plt.figure(figsize=(16,5))
        axes = fig.subplots(nrows=1, ncols=2)
        axes[0].scatter(original_polygon[:, 0], original_polygon[:, 1], label=f"{len(original_polygon)}", c='b', marker='x', s=2)
        axes[1].scatter(reduced_polygon[:, 0], reduced_polygon[:, 1], label=f"{len(reduced_polygon)}", c='b', marker='x', s=2)
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()
    
        axes[0].set_title("Original polygon")
        axes[1].set_title("Reduced polygon")
        axes[0].legend()
        axes[1].legend()
        
        # plt.show()
        plt.savefig("reduced_polygon.png")

        print("\n\n", f'[bold black] Original_polygon length[/bold black]: {len(original_polygon)}\n', 
            f'[bold black] Reduced_polygon length[/bold black]: {len(reduced_polygon)}\n'
            f'[bold black]Running time[/bold black]: {round(toc - tic, 4)} seconds')
        
        polygon_list.append(reduced_polygon.tolist())
    return polygon_list

def get_finished_json(root_dir):
    json_filter_path = Path(root_dir).rglob('*.json')
    return json_filter_path

def get_dict(json_list):
    dict_all = {}
    for json_path in json_list:
        dir, file = os.path.split(json_path)
        dir = dir.replace("Labelme", "JPEGImages")
        file_name = file.replace(".json", ".jpg")
        if not os.path.exists(os.path.join(dir, file_name)):
            file_name = file.replace(".json", ".png")
            if not os.path.exists(os.path.join(dir, file_name)):
                raise Exception("image not exist")
        image_path = os.path.join(dir, file_name)
        dict_all[image_path] = json_path
    return dict_all

def show_image_mask(img: np.array, points_list: list, alpha: float = 0.7, save_name="mask.png"):
    #? https://medium.com/mlearning-ai/yolov5-for-segmentation-fab39c3487f6
    # Create zero array for mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    overlay = img.copy()
    image = img.copy()
    for i, polygon in enumerate(points_list):
        polygon = np.array(polygon).astype(np.int32)
        # Draw polygon on the image and mask
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
        cv2.fillPoly(img, pts=[polygon], color=(255, 0, 0))
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, image)
    
    # Plot image with mask
    fig = plt.figure(figsize=(16,9))
    axes = fig.subplots(nrows=1, ncols=2)
    axes[0].imshow(image)
    axes[1].imshow(mask, cmap="Greys_r")
    axes[0].set_title("Original image with mask")
    axes[1].set_title("Mask")
    
    plt.savefig(save_name)

def process(dict_, show_mask=False):
    for image_path in dict_:
        mask = []
        class_id = []
        key_ = []
        image = cv2.imread(image_path)
        plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_path = dict_[image_path]
        data = json.load(open(json_path))
        lbl, lbl_names, points_list = labelme_shapes_to_label(image.shape, data['shapes'])
        for i in range(1, len(lbl_names)):  # * 跳过第一个class（因为0默认为背景,跳过不取！）
            key = [k for k, v in lbl_names.items() if v == i][0]
            # * 举例：当解析出像素值为1，此时对应第一个mask 为0、1组成的（0为背景，1为对象）
            mask.append((lbl == i).astype(np.uint8))
            class_id.append(i)  # * mask与class_id 对应记录保存
            key_.append(key)

        if len(mask) == 0:
            image_name = os.path.basename(image_path).split('.')[0]
            dir_ = os.path.dirname(image_path)
            image_name_ = "{}.png".format(image_name)
            dir_path = dir_.replace("JPEGImages", "p_seg_annotations")
            checkpath(dir_path)
            image_path_ = os.path.join(dir_path, image_name_)
            height, width, _ = image.shape
            im_at_fixed = np.ones((int(height),int(width)),dtype=np.uint8)
            cv2.imwrite(image_path_, im_at_fixed)
        else:
            mask = np.asarray(mask, np.uint8)
            mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])
            image_name = os.path.basename(image_path).split('.')[0]
            dir_ = os.path.dirname(image_path)
            for i in range(0, len(class_id)):
                image_name_ = "{}.png".format(image_name)
                dir_path = dir_.replace("JPEGImages", "p_seg_annotations")
                checkpath(dir_path)
                image_path_ = os.path.join(dir_path, image_name_)
                retval, im_at_fixed = cv2.threshold(mask[:, :, i], 0, 255, cv2.THRESH_BINARY)
                cv2.imwrite(image_path_, im_at_fixed)
            
        if show_mask and len(mask) != 0 :
            show_image_mask(plt_image, points_list)
            # new_points_list = show_result_reducing(points_list)
            # show_image_mask(plt_image, new_points_list, save_name="reduced_mask.png")
        elif len(mask) == 0:
            print("no mask")
        break
if __name__ == "__main__":
    root_dir = '/code/data/s_BSD/ckn_bsd_cocoformat_1/Labelme/train/pos'
    json_file = get_finished_json(root_dir)
    image_json = get_dict(json_file)
    show_mask = True
    process(image_json, show_mask=show_mask)