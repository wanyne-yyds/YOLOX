#coding:utf-8
import os
import cv2
import time
import json
import shutil
import numpy as np
from pathlib import Path 
from prettytable import PrettyTable

path2 = "/code/data/YOLOX-Yolo2CocoFormat-BSD_One_Classes-%s/"%(time.strftime("%Y-%m-%d_%H:%M", time.localtime())) # 输出文件夹
classes = ["person"]
train_txt_dir = "/code/data/s_BSD/hyh_bsd_yoloformat/train_640_0.004_221026/" # train xml文件
test_txt_dir = "/code/data/s_BSD/hyh_bsd_yoloformat/test_Normal_0.004/" # train xml文件

# train_ratio = 1.0 # 训练集的比例

START_BOUNDING_BOX_ID = 1

def get(root, name):
    return root.findall(name)

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars
  
def convert(jpg_list, json_file, txt_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    new_categories = {v-1:k for k,v in categories.items()}
    for index, line in enumerate(jpg_list):
        jpg_f = line
        if txt_file == "Train":
            filename = jpg_f.replace(train_txt_dir, '')
        else:
            filename = jpg_f.replace(test_txt_dir, '')
        img = cv2.imread(jpg_f)
        height, width, _ = img.shape
        txt_f = jpg_f[:-3] + 'txt'
        image_id = 20190000001 + index
        image = {'file_name': filename, 'height': height, 'width': width, 'id':image_id}
        json_dict['images'].append(image)
        if os.path.exists(txt_f):
            f = open(txt_f, 'r', encoding='unicode_escape')
            content = f.readlines()
            for bbox in content:
                category_num, x, y, w, h = bbox.split(' ')
                category_num, x, y, w, h = int(category_num), float(x) * width, float(y) * height, float(w) * width, float(h) * height
                category = new_categories[category_num]    
                if category in all_categories:
                    all_categories[category] += 1
                else:
                    all_categories[category] = 1
                    
                # if category not in categories:
                #     if only_care_pre_define_categories:
                #         continue
                #     new_id = len(categories) + 1
                #     print("[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(category, pre_define_categories, new_id))
                #     categories[category] = new_id

                xmin = int((x - w * 0.5))
                ymin = int((y - h * 0.5))
                xmax = int((x + w * 0.5))
                ymax = int((y + h * 0.5))

                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 1)

                category_id = categories[category]
                assert(xmax > xmin), "xmax <= xmin, {}".format(line)
                assert(ymax > ymin), "ymax <= ymin, {}".format(line)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                assert(o_width*o_height > 100), "width*height <= 0, {}".format(line)
                ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                        image_id, 'bbox':[xmin, ymin, o_width, o_height],
                        'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                        'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1
 
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories), all_categories.keys(), len(pre_define_categories), pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())
    f = open('%s %sClassessQuantity.txt'%(path2, txt_file), 'w')
    table = PrettyTable([ '类别', '数量'])
    for keys, values in all_categories.items():
        table.add_row([keys, values])
    print(table)
    print("Dataset Quantity: %d"%(index))
    f.write(str(table))
    f.write("\nDataset Quantity: %d"%(index))
    f.close() 

if __name__ == '__main__':

    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    only_care_pre_define_categories = True
    # only_care_pre_define_categories = False

    if os.path.exists(path2 + "/annotations"):
        shutil.rmtree(path2 + "/annotations")
    os.makedirs(path2 + "/annotations")
    if os.path.exists(path2 + "/train2017"):
        shutil.rmtree(path2 + "/train2017")
    os.makedirs(path2 + "/train2017", exist_ok=True)
    if os.path.exists(path2 + "/val2017"):
        shutil.rmtree(path2 +"/val2017")
    os.makedirs(path2 + "/val2017", exist_ok=True)
    
    save_json_train = path2 + 'annotations/instances_train2017.json'
    save_json_val = path2 + 'annotations/instances_val2017.json'

    train_jpg_list = list(str(i) for i in Path(train_txt_dir).rglob("*.*g"))
    train_jpg_list = np.sort(train_jpg_list)
    np.random.seed(100)
    np.random.shuffle(train_jpg_list)

    test_jpg_list = list(str(i) for i in Path(test_txt_dir).rglob("*.*g"))
    test_jpg_list = np.sort(test_jpg_list)
    np.random.seed(100)
    np.random.shuffle(test_jpg_list)

    convert(train_jpg_list, save_json_train, "Train")
    convert(test_jpg_list, save_json_val, "Val")

    f1 = open(path2 + "train.txt", "w")
    for jpg in train_jpg_list:
        name = jpg.replace(train_txt_dir, "")
        f1.write(name[:-4] + "\n")
        train_image_pathname = path2 + "train2017/" + name
        if os.path.exists(os.path.dirname(train_image_pathname)) == False:
            os.makedirs(os.path.dirname(train_image_pathname))
        shutil.copyfile(jpg, train_image_pathname)

    f2 = open(path2 + "test.txt", "w")
    for jpg in test_jpg_list:
        name = jpg.replace(test_txt_dir, "")
        f2.write(name[:-4] + "\n")
        val_image_pathname = path2 + "/val2017/" + name
        if os.path.exists(os.path.dirname(val_image_pathname)) == False:
            os.makedirs(os.path.dirname(val_image_pathname))
        shutil.copyfile(jpg, val_image_pathname)

    f1.close()
    f2.close()
    print("-------------------------------")
    print("train number:", len(train_jpg_list))
    print("val number:", len(test_jpg_list))


