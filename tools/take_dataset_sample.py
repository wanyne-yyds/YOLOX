import os
import shutil
import os.path as osp
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from prettytable import PrettyTable

from tools.pascal_voc_xml2json import get, get_and_check


if __name__ == '__main__':
    dataset = '/code/data/s_BSD/ckn_bsd_cocoformat'
    outpath = '/code/data/s_BSD/ckn_bsd_cocoformat_v0.0.1'

    labelfile = Path(dataset).rglob('*.xml')

    table = PrettyTable([ '类别', '数量'])

    classes = list()

    for file in labelfile:
        file = str(file)
        tree = ET.parse(file)
        root = tree.getroot()
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            classes.append(category)
            if category == "rearview mirror":
                imgfile = file.replace('Annotations', 'JPEGImages').replace('xml', 'jpg').replace('xml', 'png')               
                out_label_path = file.replace(dataset, outpath)
                out_image_path = out_label_path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg').replace('xml', 'png')
                os.makedirs(osp.split(out_label_path)[0], exist_ok=True)
                os.makedirs(osp.split(out_image_path)[0], exist_ok=True)
                shutil.move(imgfile, out_image_path)
                shutil.move(file, out_label_path)
                break
            
    count = Counter(classes)
    for key, value in count.items():
        table.add_row([key, value])

    print(table)