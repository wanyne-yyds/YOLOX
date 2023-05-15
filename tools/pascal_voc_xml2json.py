#coding:utf-8
import os
import cv2
import time
import json
import shutil
import numpy as np
import os.path as osp
from pathlib import Path
from collections import Counter
from prettytable import PrettyTable
from xml.etree import ElementTree as ET

path2 = "/code/data/YOLOX-CocoFormat-BSD_Two_Classes-New-WeightLoss-%s/"%(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) # 输出文件夹
classes = [ "person", "personD", "other", "ignore"]
# train_xml_dir = "/Annotations/train/"         # xml文件
# val_xml_dir   = "/Annotations/val/"           # xml文件
train_img_dir = "/code/data/s_BSD/ckn_bsd_cocoformat_1/JPEGImages/train/pos/Ordinary_camera/front_blind/2/"           # 图片
val_img_dir   = "/code/data/s_BSD/ckn_bsd_cocoformat_1/JPEGImages/val"             # 图片

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

def convert(img_list, json_file, txt_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = []
    for index, line in enumerate(img_list):
        # print("Processing %s"%(line))
        img_f = line
        xml_f = str(img_f).replace('JPEGImages', 'Annotations')[:-4] + ".xml"
        filename = osp.basename(img_f)
        image_id = 20190000001 + index
        if not osp.exists(xml_f) and img_f.find('/neg/') != -1:
            height, width, _ = cv2.imread(str(img_f)).shape
            image = {'file_name': filename, 'height': height, 'width': width, 'id':image_id}
            json_dict['images'].append(image)          
            ann = {'area': 0, 'iscrowd': 0, 'image_id':
                image_id, 'bbox':[],
                'category_id': 0, 'id': bnd_id, 'ignore': 0,
                'segmentation': []}
            
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        else:
            tree = ET.parse(xml_f)
            root = tree.getroot()
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
            image = {'file_name': filename, 'height': height, 'width': width, 'id':image_id}
            json_dict['images'].append(image)

            ## Cruuently we do not support segmentation
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category == "rearview mirror":
                    category = "other"
                all_categories.append(category)
                if category not in categories:
                    if only_care_pre_define_categories:
                        continue
                    new_id = len(categories) + 1
                    print("[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(category, pre_define_categories, new_id))
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
                ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
                xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
                ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
                assert(xmax > xmin), "xmax <= xmin, {}".format(line)
                assert(ymax > ymin), "ymax <= ymin, {}".format(line)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
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
    # print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories), all_categories.keys(), len(pre_define_categories), pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())
    f = open('%s %sClassessQuantity.txt'%(path2, txt_file), 'w')
    table = PrettyTable([ '类别', '数量'])
    count = Counter(all_categories)
    for keys, values in count.items():
        table.add_row([keys, values])
    print(table)
    print("Dataset Quantity: %d"%(index+1))
    f.write(str(table))
    f.write("\nDataset Quantity: %d"%(index+1))
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

    train_img_list = list(str(i) for i in Path(train_img_dir).rglob("**/*.*g"))
    train_img_list = np.sort(train_img_list)
    np.random.seed(100)
    np.random.shuffle(train_img_list)

    val_img_list = list(str(i) for i in Path(val_img_dir).rglob("**/*.*g"))
    val_img_list = np.sort(val_img_list)
    np.random.seed(100)
    np.random.shuffle(val_img_list)

    convert(train_img_list, save_json_train, "Train")
    convert(val_img_list, save_json_val, "Val")
    
    f1 = open(path2 + "train.txt", "w")
    for imgfile in train_img_list:
        img_name = osp.basename(imgfile)[:-4] + "\n"
        f1.write(img_name)
        train_image_pathname = path2 + "/train2017/" + osp.basename(imgfile)
        shutil.copyfile(imgfile, train_image_pathname)

    f2 = open(path2 + "test.txt", "w")
    for imgfile in val_img_list:
        img_name = osp.basename(imgfile)[:-4] + "\n"
        f2.write(img_name)
        val_image_pathname = path2 + "/val2017/" + os.path.basename(imgfile)
        shutil.copyfile(imgfile, val_image_pathname)
    f1.close()
    f2.close()
    print("-------------------------------")
    print("train number:", len(train_img_list))
    print("val number:", len(val_img_list))