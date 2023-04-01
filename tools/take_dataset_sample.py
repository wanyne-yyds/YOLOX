import os
import shutil
import os.path as osp
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from prettytable import PrettyTable

from tools.pascal_voc_xml2json import get, get_and_check

if __name__ == '__main__':
    dataset = '/code/data/s_BSD/ckn_bsd_cocoformat_1'
    outpath = '/code/data/s_BSD/ckn_bsd_cocoformat_small_Troubleshooting_rearview_mirror'

    labelfile = Path(dataset).rglob('*.xml')

    sum_table = PrettyTable(['总数据集的类别', '总数据集的类别数量'])
    img_table = PrettyTable(['类别', '图片数量'])

    classes = list()

    person = 0
    other = 0
    person_and_other = 0
    not_exist = 0
    for i, file in enumerate(labelfile):
        p=False
        o=False
        classes_in_image = False

        file = str(file)
        tree = ET.parse(file)
        root = tree.getroot()
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            classes.append(category)

            if category == "person":
                p = True
            if category == "rearview mirror":
                o = True
                classes_in_image=True
        if True:
        # if classes_in_image:
        # if i < 40000:
            if p and not o:
                person+=1
            if not p and o:
                other+=1
            if p and o:
                person_and_other += 1
            if not p and not o:
                not_exist+=1

            # imgfile = file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg')
            # out_label_path = file.replace(dataset, outpath)
            # out_image_path = out_label_path.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg')
            # if not osp.exists(imgfile):
            #     imgfile = imgfile.replace('.jpg', '.png')
            #     out_image_path = out_image_path.replace('.jpg', '.png')
            # os.makedirs(osp.split(out_label_path)[0], exist_ok=True)
            # os.makedirs(osp.split(out_image_path)[0], exist_ok=True)
            # shutil.copy(imgfile, out_image_path)
            # shutil.copy(file, out_label_path)

    count = Counter(classes)
    for key, value in count.items():
        sum_table.add_row([key, value])
    for name, quantity in zip(["Person", "Other", "Person And Other", "Not Exist"],
                              [person, other, person_and_other, not_exist]):
        img_table.add_row([name, quantity])

    print(sum_table)
    print(img_table)