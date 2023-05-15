import os
import cv2
import copy
from tqdm import tqdm
import pprint
import imagesize
import traceback
import numpy as np
from easydict import EasyDict

from yolox.utils import image_util
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.bmp')

class BaseDataset(Dataset):

    def __init__(self, label_map={}, read_image=True, transform=None, data_cache=None, ann_file_ext=None, use_opencv=True) -> None:
        super().__init__()
        if label_map is None:
            self.label_map = {}
        else:
            self.label_map = label_map
        print(f'BaseDataset label_map: {pprint.pformat(dict(label_map))}')
        self.dataset_list = []
        self.read_image = read_image
        self.transform = transform
        self.data_cache = data_cache
        self.file_num = 0
        self.ann_file_ext = ann_file_ext
        self.use_opencv = use_opencv

    def __getitem__(self, index):
        target = {}
        im = None
        if len(self.dataset_list) > 0:
            dataset = self.dataset_list[0]
            data_root = dataset['data_root']
            name_list = dataset['name_list']
            ann_list = dataset['ann_list']
            name = name_list[index]
            name = name.strip(os.path.sep)
            image_file = os.path.join(data_root, name)
            if not os.path.exists(image_file):
                return im, target
            if self.read_image:
                im = self.get_image(image_file)
                if im is None:
                    print('imread error, image_file: ', image_file)
                else:
                    height, width, _ = im.shape
            else:
                width, height = imagesize.get(image_file)

            target = copy.deepcopy(ann_list[index])
            target.update({'file_name': image_file})
            target.update({'data_root': data_root})

            bboxes = target.get('bboxes', [])
            class_labels = target.get('class_labels', [])
            keypoints = target.get('keypoints', [])
            try:
                if self.transform is not None:
                    transformed = self.transform(image=im, bboxes=bboxes, keypoints=keypoints,
                                                 class_labels=class_labels)
                    im = transformed['image']
                    transformed.pop('image')
                    target.update(transformed)
            except:
                traceback.print_exc()
                print(f'error, image_file: {image_file}')
                print(f'bboxes: {bboxes}')

        return im, target

    def __len__(self):
        if len(self.dataset_list) > 0:
            dataset = self.dataset_list[0]
            num = len(dataset['name_list'])
            return num
        else:
            return 0

    def load(self, data_root=None, label_path=None, step=1, skip_empty_label=False, min_scale=None,
             skip_no_label_file=False, **kwargs):
        dataset = {'data_root': data_root,
                   'label_path': label_path,
                   }
        name_list = []
        ann_list = []
        if self.ann_file_ext is None:
            raise ValueError(f'ann_file_ext is {self.ann_file_ext}!')

        image_dir = data_root
        count = 0

        bbox_num = 0
        empty_label_num = 0
        no_label_num = 0
        bbox_num_max = 0
        class_count = {}
        labels = None
        for root, dirs, files in sorted(os.walk(image_dir)):
            label_file_name = 'classes.txt'
            if label_file_name in files:
                labels = self.parse_classes_file(os.path.join(root, label_file_name))
                # print(f'parse_classes_file labels: {labels}')
                if len(labels) == 0:
                    print(f'NO {label_file_name} in {root}')
                    continue

            if label_path is None:
                _label_dir = root
            else:
                _label_dir = root.replace(data_root, label_path)
            types = IMG_EXTENSIONS
            for file_name in files:
                (file_base, ext) = os.path.splitext(file_name)
                if ext in types:
                    ann_file = file_base + self.ann_file_ext
                    ann_file = os.path.join(_label_dir, ann_file)
                    if not os.path.exists(ann_file):
                        ann_file = ann_file.replace('images', 'labels')

                    image_file = os.path.join(root, file_name)
                    if os.path.exists(image_file):
                        count = count + 1
                        if step != 0 and (count % step) != 0:
                            continue

                        width, height = imagesize.get(image_file)
                        anns_out = {}
                        if os.path.exists(ann_file):
                            anns_out = self.parse_annotation_file(ann_file, skip_empty_label=skip_empty_label,
                                                                  bbox_min_scale=min_scale, labels=labels,
                                                                  image_size=(width, height))
                            if skip_empty_label:
                                if 'bboxes' in anns_out:
                                    num = len(anns_out.get('bboxes', []))
                                    if num == 0:
                                        continue
                                else:
                                    continue
                        else:
                            if skip_no_label_file:
                                continue
                            no_label_num += 1
                        if len(image_dir) < len(root):
                            relative_path = root[len(image_dir):]
                        else:
                            relative_path = ''
                        name_list.append(os.path.join(relative_path, file_name))
                        ann_list.append(anns_out)
                        num = 0
                        if 'bboxes' in anns_out:
                            num = len(anns_out.get('bboxes', []))
                            bbox_num += num
                            if num > bbox_num_max:
                                bbox_num_max = num
                        if num == 0 and os.path.exists(ann_file):
                            empty_label_num += 1
                        if 'class_labels' in anns_out:
                            class_labels = anns_out.get('class_labels', [])
                            for key in class_labels:
                                class_count[key] = class_count.get(key, 0) + 1

        dataset.update({'name_list': name_list})
        dataset.update({'ann_list': ann_list})
        print('Dataset load name_list len: ', len(name_list))
        print(f'Dataset load ann_list len: {len(ann_list)}, bbox_num: {bbox_num}, empty_label_num: {empty_label_num}, '
              f'no_label_file_num: {no_label_num}, bbox_num_max: {bbox_num_max}')
        print(f'class_count: {pprint.pformat(class_count)}')
        self.dataset_list.append(dataset)

    def parse_annotation_file(self, filename, skip_empty_label=False):
        pass

    def visit(self, show=False, dump_ann=False, save_dir=None, format_converter=None, bbox_checker=None, transform=None,
              relative_path=True, min_area_ratio=None):
        # pbar = tqdm(total=len(self), mininterval=1, position=0, leave=True)
        for im, target in tqdm(self):
            # pbar.update(1)
            image_file = target['file_name']
            if relative_path and 'data_root' in target:
                data_root = target['data_root']
                full_path = os.path.dirname(image_file)
                rel_path = full_path[len(data_root):]
                rel_path = rel_path.strip(os.path.sep)
            else:
                rel_path = ''
            self.file_num += 1
            bboxes = target.get('bboxes', [])
            class_labels = target.get('class_labels', [])
            keypoints = target.get('keypoints', [])
            found = False

            if bbox_checker is not None:
                im, target = bbox_checker(im, target)
                bboxes = target.get('bboxes', [])
                class_labels = target.get('class_labels', [])

            area_ratio_max = 0
            if len(bboxes) > 0:
                bboxes = np.array(bboxes)
                h, w, _ = im.shape
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
                if min_area_ratio is not None:
                    area = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
                    area = area / (h * w)
                    area_ratio_max = np.max(area)

            if min_area_ratio is not None:
                if area_ratio_max < min_area_ratio:
                    continue

            if transform is not None:
                try:
                    # label = np.zeros(bboxes.shape[0]).reshape(-1, 1)
                    # bboxes = np.append(bboxes, label, axis=1)
                    transformed = transform(image=im, bboxes=bboxes, keypoints=keypoints,
                                                 class_labels=class_labels)
                    im = transformed['image']
                    transformed.pop('image')
                    target.update(transformed)
                    bboxes = target.get('bboxes', [])
                    class_labels = target.get('class_labels', [])
                except:
                    print(f'error, visit file_name: {image_file}')
                    print(f'bboxes: {bboxes}')
                    traceback.print_exc()

            if format_converter is not None:
                format_converter(im, target, save_dir)

            im_out = im
            if dump_ann:
                for bbox, class_id in zip(bboxes, class_labels):
                    found = True
                    left, top, right, bottom = map(int, bbox[:4])
                    if dump_ann and im_out is not None:
                        color = image_util.get_color(class_id)
                        cv2.rectangle(im_out, (left, top), (right, bottom), color, 2)
                        # h, w = im_out.shape[:2]
                        # class_id = (right - left) * (bottom - top) / (w * h)
                        cv2.putText(im_out, f'{class_id}', (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

            if show and found:
                cv2.imshow('output', im_out)
                cv2.waitKey(100)
            if dump_ann and save_dir is not None and os.path.exists(save_dir):
                # image_file = f'{self.file_num}.jpg'
                path = os.path.join(save_dir, rel_path, os.path.basename(image_file))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, im_out)

    def parse_classes_file(self, label_file):
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                labels.append(line.rstrip())

        for label in labels:
            if label is not None and label not in self.label_map:
                index = len(self.label_map)
                self.label_map.update({label: index})

        return labels

    def get_label_id(self, label):
        index = 0
        if label is not None:
            if label not in self.label_map:
                index = len(self.label_map)
                self.label_map.update({label: index})
            else:
                index = self.label_map[label]
        return index

    def _read_image(self, path, method=1):
        return image_util.read_image(path, method)

    # @debug_util.print_function_time
    def get_image(self, path):
        # key = path
        # to_gray = False
        # if self.data_cache is None:
        #     im = self._read_image(path)
        # else:
        #     try:
        #         im = self.data_cache.get(key)
        #         if im is None:
        #             im = self._read_image(path)
        #             if not self.data_cache.is_full:
        #                 if to_gray:
        #                     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #                     self.data_cache.set(key, gray)
        #                 else:
        #                     self.data_cache.set(key, im)
        #         else:
        #             if to_gray:
        #                 im = cv2.merge([im, im, im])
        #     except Exception as e:
        #         # print("Exception: %s" % e)
        #         im = self._read_image(path)
        if self.use_opencv:
            method = 1
        else:
            method = 2
        im = self._read_image(path, method=method)
        return im

    def get_property(self, index):
        target = {}
        if len(self.dataset_list) > 0:
            dataset = self.dataset_list[0]
            data_root = dataset['data_root']
            name_list = dataset['name_list']
            ann_list = dataset['ann_list']
            name = name_list[index]
            name = name.strip(os.path.sep)
            image_file = os.path.join(data_root, name)
            width, height = imagesize.get(image_file)
            target = copy.deepcopy(ann_list[index])
            target.update({'file_name': image_file})
            target.update({'data_root': data_root})
            target.update({'width': width})
            target.update({'height': height})
        return target

class YoloDataset(BaseDataset):
    """
    label file: txt
    """

    def __init__(self, label_map={}, read_image=True, transform=None, data_cache=None, ann_file_ext='.txt', use_opencv=True) -> None:
        super().__init__(label_map, read_image, transform, data_cache, ann_file_ext, use_opencv=use_opencv)

    def parse_annotation_file(self, filename, skip_empty_label=False, bbox_min_scale=None, labels=None, image_size=None):
        ret = {}
        bboxes = []
        class_labels = []

        ann_file = filename
        lines = []
        with open(ann_file, 'r', encoding='utf8') as f:
                try:
                    for line in f:
                        lines.append(line.rstrip())
                except:
                    print(f'parse_annotation_file error, file: {ann_file}')
        if len(lines) == 0:
            # print(f'Warning, empty file! {ann_file}')
            if skip_empty_label:
                return ret
        else:
            nrow = len(lines)
            anns = np.loadtxt(lines).reshape(nrow, -1)
            for ann in anns:
                [_id, c1, c2, w, h] = ann.tolist()
                scale = w * h
                if bbox_min_scale is not None:
                    if scale < bbox_min_scale:
                        # print(f'Warning, scale[{scale}] < min_scale[{min_scale}]')
                        continue

                width, height = image_size
                left = int((c1 - w / 2) * width)
                top = int((c2 - h / 2) * height)
                right = int((c1 + w / 2) * width)
                bottom = int((c2 + h / 2) * height)
                if self.label_map is not None and len(self.label_map) > 0 and labels is not None:
                    name = labels[int(_id)]
                    if name in self.label_map:
                        label_id = self.label_map[name]
                    else:
                        continue
                else:
                    label_id = int(_id)

                bboxes.append((left, top, right, bottom))
                class_labels.append(label_id)

        ret = {
            'bboxes': bboxes,
            'class_labels': class_labels,
        }
        return ret

class BaseObject(object):
    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

class FormatConverter(BaseObject):
    def __init__(self, save_dir, label_save_dir=None, skip_empty_label=False, relative_path=True) -> None:
        super().__init__()
        self.save_dir = save_dir
        if label_save_dir is None:
            label_save_dir = save_dir
        self.label_save_dir = label_save_dir
        self.skip_empty_label = skip_empty_label
        self.relative_path = relative_path

    def convert(self, dataset):
        output_dir = self.save_dir
        if not os.path.exists(output_dir):
            return

        file_num = 0
        pbar = tqdm(total=len(dataset), mininterval=1, position=0, leave=True)
        for im, target in dataset:
            pbar.update(1)
            file_num += 1
            self.forward(im, target)

    def forward(self, im, target, label_map=None):
        pass

class YoloFormatConverter(FormatConverter):

    def forward(self, im, target, label_map=None, rel_path='', file_name=None):
        save_dir = self.save_dir
        label_save_dir = self.label_save_dir
        file_num = 0
        ann_file_ext = '.txt'
        if im is not None and target is not None:
            image_file = target.get('file_name')
            height, width, _ = im.shape
            if self.relative_path and 'data_root' in target:
                data_root = target['data_root']
                full_path = os.path.dirname(image_file)
                rel_path = full_path[len(data_root):]
                rel_path = rel_path.strip(os.path.sep)
            file_num += 1
            # bboxes = target['bboxes']
            # class_labels = target['class_labels']
            bboxes = target.get('bboxes', [])
            class_labels = target.get('class_labels', [])
            if self.skip_empty_label:
                if len(class_labels) == 0:
                    return
            # id, cx, cy, w, h
            if file_name is None:
                image_name = os.path.basename(image_file)
            else:
                image_name = file_name
            name = os.path.splitext(image_name)[0]
            ann_file_name = name + ann_file_ext
            ann_file_out = os.path.join(label_save_dir, rel_path, ann_file_name)
            os.makedirs(os.path.dirname(ann_file_out), exist_ok=True)
            with open(ann_file_out, 'w+') as ann_fd:
                for bbox, class_id in zip(bboxes, class_labels):
                    found = True
                    left, top, right, bottom = map(int, bbox)
                    w = abs(right - left) / width
                    h = abs(bottom - top) / height
                    cx = (left + right) / 2
                    cx /= width
                    cy = (top + bottom) / 2
                    cy /= height
                    ann_fd.write(f'{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

            if save_dir is not None and os.path.exists(save_dir):
                path = os.path.join(save_dir, rel_path, image_name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                h, w, c = im.shape
                if h >= 1000 or w >= 1000:
                    im = cv2.resize(im, (w//2, h//2))
                cv2.imwrite(path, im)

class BboxChecker(BaseObject):

    def __init__(self, min_scale=None, min_length=5, iou_threshold=0.2, classes_keep=[], classes_remove=[], w2h_ratio=None) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.min_length = min_length
        self.w2h_ratio = w2h_ratio
        self.iou_threshold = iou_threshold
        self.classes_keep = classes_keep
        self.classes_remove = classes_remove

    def forward(self, im, target):
        min_scale = self.min_scale
        min_length = self.min_length
        if im is not None and target is not None:
            image_file = target['file_name']
            height, width, _ = im.shape
            bboxes = target.get('bboxes', [])
            class_labels = target.get('class_labels', [])
            keypoints = target.get('keypoints', [])
            bbox_keep = []
            bbox_del = []
            for bbox, class_id in zip(bboxes, class_labels):
                if self.classes_keep is not None and class_id not in self.classes_keep and class_id not in self.classes_remove:
                    continue
                left, top, right, bottom = map(int, bbox)
                w = abs(right - left)
                h = abs(bottom - top)

                wrong_scale = False
                scale = (w * h) / (width * height)
                if min_scale is not None:
                    if scale < min_scale:
                        wrong_scale = True
                if self.w2h_ratio is not None:
                    w2h = w / h
                    if not (self.w2h_ratio[0] < w2h < self.w2h_ratio[1]):
                        wrong_scale = True

                bbox_info = EasyDict()
                bbox_info.bbox = bbox
                bbox_info.class_id = class_id
                if class_id in self.classes_remove:
                    bbox_del.append(bbox_info)
                else:
                    if w < min_length or h < min_length or wrong_scale:
                        bbox_del.append(bbox_info)
                        # print(f'delete_anns, w = {w}, h = {h}, img_name = {image_file}, scale = {scale:.5f}')
                    else:
                        bbox_keep.append(bbox_info)

            bbox_keep, bbox_del = self.check_bboxes(bbox_keep, bbox_del, image_file, iou_threshold=self.iou_threshold)
            for delete_ann in bbox_del:
                rect = tuple(map(lambda x: int(x), delete_ann['bbox']))
                (left, top, right, bottom) = rect
                rect = (left, top, (right - left), (bottom - top))
                image_util.fill_rect(im, rect, fill_value=0)

            bboxes = []
            class_labels = []
            for bbox_info in bbox_keep:
                bboxes.append(bbox_info.bbox)
                class_labels.append(bbox_info.class_id)
            target['bboxes'] = bboxes
            target['class_labels'] = class_labels
        return im, target

    def check_bboxes(self, bbox_keep, bbox_del, img_name, iou_threshold=0.1):
        for good_ann in bbox_keep:
            for delete_ann in bbox_del:
                iou = image_util.calc_iou(good_ann.bbox, delete_ann.bbox, mode=1)
                if iou > iou_threshold:
                    # print(f"check_bboxes img_name: {img_name}, iou: {iou}")
                    bbox_del.append(good_ann)
                    bbox_keep.remove(good_ann)
                    return self.check_bboxes(bbox_keep, bbox_del, img_name)
                else:
                    pass

        return bbox_keep, bbox_del
    
if __name__ == "__main__":

    data_root = r'/code/data/s_BSD/hyh_bsd_yoloformat/train_640_0.004_221026_pillar_mirror_check_bbox'
    output_dir = r'/code/data/s_BSD/hyh_bsd_yoloformat/xxx'

    label_map = {'person': 0, 'pillar': 1, 'mirror': 2}
    # label_map = {'person': 0, 'personD': 1, 'pillar': 2, 'mirror': 3}
    dataset = YoloDataset(label_map=label_map)
    dataset.load(data_root=data_root, step=1, skip_empty_label=False, skip_no_label_file=False)
    # dataset.load(data_root=data_root, step=1, skip_empty_label=True, skip_no_label_file=True)
    # time = datetime.now().strftime('%m%d_%H%M%S')
    # output_dir = os.path.join(output_dir, time)
    os.makedirs(output_dir, exist_ok=True)

    # bbox_checker = None
    bbox_checker = BboxChecker(min_scale=0.004, classes_keep=[0, 1, 2], classes_remove=[], iou_threshold=0.15)
    # format_converter = None
    format_converter = YoloFormatConverter(save_dir=output_dir, skip_empty_label=False)
    # transform = get_transform2()
    transform = None
    dataset.visit(show=False, dump_ann=False, save_dir=output_dir, bbox_checker=bbox_checker,
                  format_converter=format_converter, transform=transform)