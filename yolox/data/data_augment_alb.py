from albumentations import *
import cv2
import math
import numpy as np
import random
import traceback
import copy
from typing import *
from yolox.utils import image_util
from albumentations.core.bbox_utils import union_of_bboxes, \
    convert_bbox_to_albumentations, convert_bbox_from_albumentations, \
    convert_bboxes_from_albumentations, convert_bboxes_to_albumentations


class PadToAspectRatio(PadIfNeeded):
    def __init__(self, w2h_ratio, min_height: Optional[int] = 1024, min_width: Optional[int] = 1024,
                 pad_height_divisor: Optional[int] = None, pad_width_divisor: Optional[int] = None,
                 position: Union[PadIfNeeded.PositionType, str] = PadIfNeeded.PositionType.CENTER, border_mode=cv2.BORDER_REFLECT_101,
                 value=None, mask_value=None, always_apply=False, p=1.0, threshold=0.3):
        super().__init__(min_height, min_width, pad_height_divisor, pad_width_divisor, position, border_mode, value,
                         mask_value, always_apply, p)
        self.w2h_ratio = w2h_ratio
        self.threshold = threshold

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        dst_w = max(img_w, int(img_h * self.w2h_ratio))
        dst_h = max(img_h, int(img_w / self.w2h_ratio))
        self.min_width = dst_w
        self.min_height = dst_h
        return {}

    def __call__(self, *args, force_apply=False, **kwargs):
        params = kwargs
        img_h, img_w = params["image"].shape[:2]
        img_w2h = img_w / img_h
        if img_w2h - self.threshold < self.w2h_ratio < img_w2h + self.threshold:
            return kwargs
        else:
            return super().__call__(*args, force_apply=force_apply, **kwargs)


class CommonDualTransform(DualTransform):

    def __init__(self, apply_bbox_target=None, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.apply_bbox_target = apply_bbox_target

    def __call__(self, *args, force_apply=False, **kwargs):
        params = kwargs
        if self.apply_bbox_target is None:
            return super().__call__(*args, force_apply=force_apply, **kwargs)
        else:
            has_bbox = False
            to_apply = False
            if "bboxes" in params and params["bboxes"] is not None and len(params["bboxes"]) > 0:
                has_bbox = True
            if self.apply_bbox_target and has_bbox:
                to_apply = True
            elif (not self.apply_bbox_target) and (not has_bbox):
                to_apply = True
            if to_apply:
                return super().__call__(*args, force_apply=force_apply, **kwargs)
            else:
                return kwargs

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_params_dependent_on_targets(self, params):
        return {}


class RandomEraseBbox(CommonDualTransform):
    """
    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale=(0.6, 0.75),
        h_start=(0, 0.05),
        always_apply=False,
        p=1.0,
        p_erase_all=0.5,
        keep_bbox=False,
        bbox_min_scale=None,
        bbox_min_edge_scale=None,
        apply_bbox_target=None
    ):

        super(RandomEraseBbox, self).__init__(
            apply_bbox_target=apply_bbox_target,
            always_apply=always_apply, p=p
        )
        self.scale_ratio = scale
        self.h_start = h_start
        self.p_erase_all = p_erase_all
        self.keep_bbox = keep_bbox
        self.bbox_min_scale = bbox_min_scale
        self.bbox_min_edge_scale = bbox_min_edge_scale

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        self.update_params(params, **kwargs)
        im = kwargs.get('image', [])
        bboxes = kwargs.get('bboxes', [])
        class_labels = kwargs.get('class_labels', [])
        img_h, img_w = im.shape[:2]
        if len(bboxes) == 0 or len(class_labels) == 0:
            return kwargs

        bboxes_out = []
        class_labels_out = []
        box_id = -1
        if random.random() < self.p_erase_all:
            force = True
        else:
            force = False

        for box, class_id in zip(bboxes, class_labels):
            box_id += 1
            box_orig = copy.deepcopy(box)
            if self.bbox_min_scale is not None:
                area_size = image_util.get_box_size(box)
                if area_size < self.bbox_min_scale:
                    bboxes_out.append(box_orig)
                    class_labels_out.append(class_id)
                    continue
            box = convert_bbox_from_albumentations(box, 'pascal_voc', img_h, img_w)[:4]
            if self.bbox_min_edge_scale is not None:
                edge_ratio = image_util.get_bboxes_edge_ratio(im, [box])
                if edge_ratio < self.bbox_min_edge_scale:
                    bboxes_out.append(box_orig)
                    class_labels_out.append(class_id)
                    continue
            skip = False
            for _id, box_2 in enumerate(bboxes):
                box_2 = convert_bbox_from_albumentations(box_2, 'pascal_voc', img_h, img_w)
                if _id != box_id:
                    iou = image_util.calc_iou(box, box_2, mode=1)
                    if iou > 0.001:
                        skip = True
                        break

            if self.keep_bbox and skip:
                bboxes_out.append(box_orig)
                class_labels_out.append(class_id)
                continue
            if (not self.always_apply) and (not force) and (skip or random.random() >= 0.7):
                bboxes_out.append(box_orig)
                class_labels_out.append(class_id)
                continue

            # fill_value = random.randint(0, 255)
            fill_value = [random.randint(0, 255) for _ in range(3)]
            if self.keep_bbox:
                bbox_crop = image_util.crop_sub_bbox_from_bbox(
                    im, self.scale_ratio, bbox=box, h_start=self.h_start)
                x1, y1, x2, y2 = bbox_crop[:4]
                img_crop = copy.deepcopy(im[y1:y2, x1:x2])
                image_util.fill_bbox(im, bbox=box, fill_value=fill_value)
                im[y1:y2, x1:x2] = img_crop
                bbox_new = convert_bbox_to_albumentations(bbox_crop, 'pascal_voc', img_h, img_w)
                bboxes_out.append((*bbox_new, *box_orig[4:]))
                class_labels_out.append(class_id)
            else:
                image_util.random_fill_rect(im, self.scale_ratio, bbox=box,
                                            h_start=self.h_start, fill_value=fill_value)
        kwargs.update({
            "image": im,
            "bboxes": bboxes_out,
            "class_labels": class_labels_out,
        })
        return kwargs

    def get_transform_init_args_names(self):
        return {"scale_ratio": self.scale_ratio, "h_start": self.h_start}


class MyRotate(Rotate):

    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None,
                 mask_value=None, rotate_method="largest_box", always_apply=False, p=0.5, adjust_factor=0.35):
        super().__init__(limit, interpolation, border_mode, value, mask_value, rotate_method, always_apply, p)
        self.adjust_factor = adjust_factor

    def apply_to_bbox(self, bbox, angle=0, **params):
        rows = params["rows"]
        cols = params["cols"]
        x_min, y_min, x_max, y_max = bbox[:4]
        scale = cols / float(rows)
        x = np.array([x_min, x_max, x_max, x_min]) - 0.5
        y = np.array([y_min, y_min, y_max, y_max]) - 0.5
        angle = np.deg2rad(angle)
        x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
        y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
        x_t = x_t + 0.5
        y_t = y_t + 0.5

        if self.adjust_factor is None:
            x_min, x_max = min(x_t), max(x_t)
            y_min, y_max = min(y_t), max(y_t)
        else:
            x_t = np.sort(x_t)
            y_t = np.sort(y_t)
            x_min = x_t[0] + self.adjust_factor * (x_t[1] - x_t[0])
            x_max = x_t[3] - self.adjust_factor * (x_t[3] - x_t[2])
            y_min = y_t[0] + self.adjust_factor * (y_t[1] - y_t[0])
            y_max = y_t[3] - self.adjust_factor * (y_t[3] - y_t[2])
        return x_min, y_min, x_max, y_max


class RandomResizedCropIfEmpty(RandomResizedCrop):

    def __call__(self, *args, force_apply=False, **kwargs):
        params = kwargs
        if "bboxes" in params and params["bboxes"] is not None and len(params["bboxes"]) > 0:
            return kwargs
        else:
            return super().__call__(*args, force_apply=force_apply, **kwargs)

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]


class RandomCropIfEmpty(RandomResizedCropIfEmpty):

    def __init__(self, height, width, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),
                 interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super().__init__(height, width, scale, ratio, interpolation, always_apply, p)
        self.to_resize = False

    def apply_with_params(self, params, **kwargs):
        if not self.to_resize:
            self.width = params['crop_width']
            self.height = params['crop_height']
        return super().apply_with_params(params, **kwargs)


class MyRandomSizedBBoxSafeCrop(RandomResizedCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height,
        width,
        erosion_rate=0.0,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.33),
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        union_of_bboxes=False,
        crop_bbox_num=None,
        p=1.0,
        to_resize=False,
        bbox_weight_power=0.5,
        alignment='center'
    ):

        super(MyRandomSizedBBoxSafeCrop, self).__init__(
            height=height, width=width, interpolation=interpolation, scale=scale, ratio=ratio,
            always_apply=always_apply, p=p
        )
        self.erosion_rate = erosion_rate
        self.union_of_bboxes = union_of_bboxes
        self.crop_bbox_num = crop_bbox_num
        self.to_resize = to_resize
        self.bbox_weight_power = bbox_weight_power
        self.alignment = alignment

    def get_crop_bboxes(self, params):
        img_h, img_w = params["image"].shape[:2]
        bboxes = params["bboxes"]
        is_union = self.union_of_bboxes
        if is_union:
            # get union of all bboxes
            x, y, x2, y2 = union_of_bboxes(
                width=img_w, height=img_h, bboxes=params["bboxes"], erosion_rate=self.erosion_rate
            )
        else:
            bbox_weight = []
            bbox_num = len(bboxes)
            if bbox_num > 0:
                for bbox in bboxes:
                    area_size = image_util.get_box_size(bbox)
                    bbox_weight.append(area_size)
                bbox_weight = np.power(bbox_weight, self.bbox_weight_power)
                if self.crop_bbox_num is None:
                    num_list = np.ones(len(bboxes))
                    # num_list[0] = len(bboxes) + 2
                    num_weight = num_list / num_list.sum()
                    num = np.random.choice(np.arange(1, len(bboxes) + 1),
                                           1, replace=False, p=num_weight)
                else:
                    num = min(self.crop_bbox_num, bbox_num)
                bbox_weight = bbox_weight / bbox_weight.sum()
                dst_index = np.random.choice(np.arange(0, len(bboxes)),
                                             num, replace=False, p=bbox_weight)
                dst_bboxes = []
                for i in dst_index:
                    dst_bboxes.append(bboxes[i])
                    # area_size = image_util.get_box_size(bboxes[i])
                    # if area_size < 0.01:
                    #     too_small = True
                # print(f'dst_index: {dst_index}, dst_bboxes: {dst_bboxes}')
            else:
                dst_bboxes = bboxes
            x, y, x2, y2 = union_of_bboxes(
                width=img_w, height=img_h, bboxes=dst_bboxes, erosion_rate=self.erosion_rate
            )
        return x, y, x2, y2

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        area = img.shape[0] * img.shape[1]

        img_h, img_w = params["image"].shape[:2]
        scale_min, scale_max = self.scale
        x, y, x2, y2 = self.get_crop_bboxes(params)
        if (x2 == 0 and y2 == 0) or (x2 == img_w and y2 == img_h) or (x2 == 1 and y2 == 1):
            return {
                "crop_height": img_h,
                "crop_width": img_w,
                "h_start": 0,
                "w_start": 0,
            }
        x_min, y_min, x_max, y_max = int(
            x * img_w), int(y * img_h), int(x2 * img_w), int(y2 * img_h)
        bbox = (x_min, y_min, x_max, y_max)
        bbox = image_util.zoom_box(bbox, 0.1, img)
        (x_min, y_min, x_max, y_max) = bbox
        target_area_min = image_util.get_box_size(bbox) / image_util.get_image_size(img)
        target_aspect_ratio = (x_max - x_min) / (y_max - y_min)
        scale_min = max(scale_min, target_area_min)
        # print(f'scale_min: {scale_min}, scale_max: {scale_max}, target_area_min: {target_area_min}')
        for _attempt in range(10):
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            if target_aspect_ratio == 0 or aspect_ratio == 0:
                continue
            _target_area_min = target_area_min * \
                max(aspect_ratio / target_aspect_ratio, target_aspect_ratio / aspect_ratio)
            _scale_min = max(scale_min, _target_area_min)
            _scale_max = max(scale_max, _scale_min)
            # print(f'_scale_min: {_scale_min}, _scale_max: {_scale_max}, _target_area_min: {_target_area_min}')
            target_area = random.uniform(_scale_min, _scale_max) * area
            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                crop_box = image_util.get_crop_coords_from_bbox(
                    bbox, img, (w, h), alignment=self.alignment)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": crop_box[1] * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": crop_box[0] * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        if j > x_min or i > y_min or y_max > (i + h) or x_max > (j + w):
            i = 0
            j = 0
            w = img.shape[1]
            h = img.shape[0]
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return "height", "width", "erosion_rate", "scale", "ratio", "interpolation"

    def apply_with_params(self, params, **kwargs):
        if not self.to_resize:
            self.width = params['crop_width']
            self.height = params['crop_height']
        return super().apply_with_params(params, **kwargs)


class DetectedBBoxSafeCrop(MyRandomSizedBBoxSafeCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, erosion_rate=0.0, scale=(0.08, 1.0), ratio=(0.75, 1.33),
                 interpolation=cv2.INTER_LINEAR, always_apply=False, union_of_bboxes=False, crop_bbox_num=1, p=1.0,
                 apply_if_empty=True, alignment='center', score_weight_power=0, detector=None):
        super().__init__(height, width, erosion_rate, scale, ratio, interpolation, always_apply, union_of_bboxes,
                         crop_bbox_num, p, alignment=alignment)
        self.apply_if_empty = apply_if_empty
        if isinstance(detector, str):
            self.detector = eval(detector)
        else:
            self.detector = detector
        self.score_weight_power = score_weight_power

    def __call__(self, *args, force_apply=False, **kwargs):
        params = kwargs
        if self.apply_if_empty and "bboxes" in params and params["bboxes"] is not None and len(params["bboxes"]) > 0:
            # print(f'len(params["bboxes"]): {len(params["bboxes"])}')
            return kwargs
        else:
            return super().__call__(*args, force_apply=force_apply, **kwargs)

    def get_crop_bboxes(self, params):
        img = params["image"]
        img_h, img_w = img.shape[:2]
        bboxes = params["bboxes"]
        bbox_num = len(bboxes)
        if self.apply_if_empty and bbox_num > 0:
            return 0, 0, 1, 1

        if self.detector is not None:
            result = self.detector.detect_image(
                img, image_size=None, threshold=0.02, keep_ratio=False)
            bboxes = result.get('bboxes', [])
            bbox_num = len(bboxes)
            if bbox_num > 0:
                bboxes = convert_bboxes_to_albumentations(bboxes, 'pascal_voc', img_h, img_w)
                scores = result['scores']
                scores_sorted = np.sort(scores)[::-1]
                dst_num = min(8, np.size(scores_sorted))
                score_threshold = scores_sorted[dst_num - 1]
                scores[np.where(scores < score_threshold)] = 0
                # print(f'bbox_num: {bbox_num}, score_threshold: {score_threshold}, scores: {scores}')

        is_union = self.union_of_bboxes
        if is_union:
            # get union of all bboxes
            x, y, x2, y2 = union_of_bboxes(
                width=img_w, height=img_h, bboxes=bboxes, erosion_rate=self.erosion_rate
            )
        else:
            if bbox_num > 0:
                if self.crop_bbox_num is None:
                    num_list = np.ones(len(bboxes))
                    # num_list[0] = len(bboxes) + 2
                    num_weight = num_list / num_list.sum()
                    num = np.random.choice(np.arange(1, len(bboxes) + 1),
                                           1, replace=False, p=num_weight)
                else:
                    num = min(self.crop_bbox_num, bbox_num)
                if self.score_weight_power != 0:
                    scores = np.power(scores, self.score_weight_power)
                bbox_weight = scores / scores.sum()
                dst_index = np.random.choice(np.arange(0, len(bboxes)),
                                             num, replace=True, p=bbox_weight)
                dst_bboxes = []
                for i in dst_index:
                    dst_bboxes.append(bboxes[i])
            else:
                dst_bboxes = bboxes
            x, y, x2, y2 = union_of_bboxes(
                width=img_w, height=img_h, bboxes=dst_bboxes, erosion_rate=self.erosion_rate
            )
        return x, y, x2, y2


class RandomCropAndPaste(DualTransform):
    """Torchvision's variant of crop a bbox of the input and rescale it to some size,
        and then paste it into random location.

    Args:
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, scale=None,
                 ratio=None,
                 interpolation=cv2.INTER_LINEAR,
                 bbox_min_scale=None,
                 always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.bbox_min_scale = bbox_min_scale

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        self.update_params(params, **kwargs)
        im = kwargs.get('image', [])
        bboxes = kwargs.get('bboxes', [])
        class_labels = kwargs.get('class_labels', [])
        img_h, img_w = im.shape[:2]
        bbox_num = len(bboxes)
        if bbox_num == 0 or len(class_labels) == 0:
            return kwargs
        if bbox_num >= 5:
            return kwargs

        bboxes_out = copy.copy(bboxes)
        class_labels_out = copy.deepcopy(class_labels)
        # bbox_id_list = [*range(bbox_num)]
        bbox_area = []
        for bbox in bboxes:
            area_size = image_util.get_box_size(bbox)
            bbox_area.append(area_size)
        bbox_weight = np.power(bbox_area, 0.5)
        bbox_weight = bbox_weight / bbox_weight.sum()
        bbox_id_list = np.random.choice(
            np.arange(0, bbox_num), bbox_num, replace=False, p=bbox_weight)
        # random.shuffle(bbox_id_list)
        dst_box = None
        for box_id in bbox_id_list:
            if self.bbox_min_scale is not None:
                if bbox_area[box_id] < self.bbox_min_scale:
                    continue
            box = bboxes[box_id]
            box = convert_bbox_from_albumentations(box, 'pascal_voc', img_h, img_w)

            skip = False
            bboxes_abs = convert_bboxes_from_albumentations(bboxes, 'pascal_voc', img_h, img_w)
            for _id, box_2 in enumerate(bboxes_abs):
                if _id != box_id:
                    iou = image_util.calc_iou(box_2, box, mode=1)
                    if iou > 0.00001:
                        skip = True
                        break

            if not skip:
                box = tuple(map(int, box[:4]))
                x1, y1, x2, y2 = box
                for i in range(10):
                    dst_w = abs(x2 - x1)
                    dst_h = abs(y2 - y1)
                    if self.scale is not None:
                        scale = random.uniform(self.scale[0], self.scale[1])
                        scale = min(scale, img_w / dst_w, img_h / dst_h)
                        dst_w = int(dst_w * scale)
                        dst_h = int(dst_h * scale)
                    got = True
                    dst_left = random.uniform(0, img_w - dst_w)
                    dst_top = random.uniform(0, img_h - dst_h)
                    dst_box = (dst_left, dst_top, dst_left + dst_w, dst_top + dst_h)
                    for _id, box_2 in enumerate(bboxes_abs):
                        if _id != box_id:
                            iou = image_util.calc_iou(box_2, dst_box, mode=1)
                            if iou > 0.00001:
                                got = False
                                break
                    if got:
                        break
                if got:
                    img_target = copy.deepcopy(im[y1:y2, x1:x2])
                    if self.scale is not None:
                        img_target = cv2.resize(img_target, (dst_w, dst_h))
                    # fill_value = random.randint(0, 255)
                    fill_value = [random.randint(0, 255) for _ in range(3)]
                    image_util.fill_bbox(im, box, fill_value=fill_value)
                    x1, y1, x2, y2 = tuple(map(int, dst_box[:4]))
                    im[y1:y2, x1:x2] = img_target
                    break
                else:
                    dst_box = None

        if dst_box is not None:
            bbox_new = convert_bbox_to_albumentations(dst_box, 'pascal_voc', img_h, img_w)
            bboxes_out[box_id] = (*bbox_new, *bboxes_out[box_id][4:])
        kwargs.update({
            "image": im,
            "bboxes": bboxes_out,
            "class_labels": class_labels_out,
        })
        return kwargs


class RandomCopyBbox(RandomCropAndPaste):
    """Torchvision's variant of copy a bbox of the input and rescale it to some size,
        and then paste it into random location.

    Args:
        scale ((float, float)): range of size of the bbox resized
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, scale=(0.8, 1.5), ratio=None, interpolation=cv2.INTER_LINEAR, bbox_min_scale=None,
                 always_apply=False, p=0.5):
        super().__init__(scale, ratio, interpolation, bbox_min_scale, always_apply, p)

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        self.update_params(params, **kwargs)
        im = kwargs.get('image', [])
        bboxes = kwargs.get('bboxes', [])
        class_labels = kwargs.get('class_labels', [])
        img_h, img_w = im.shape[:2]
        bbox_num = len(bboxes)
        if bbox_num == 0 or len(class_labels) == 0:
            return kwargs
        if bbox_num >= 5:
            return kwargs

        bboxes_out = copy.copy(bboxes)
        class_labels_out = copy.deepcopy(class_labels)
        bbox_area = []
        for bbox in bboxes:
            area_size = image_util.get_box_size(bbox)
            bbox_area.append(area_size)
        bbox_weight = np.power(bbox_area, 0.5)
        bbox_weight = bbox_weight / bbox_weight.sum()
        bbox_id_list = np.random.choice(
            np.arange(0, bbox_num), bbox_num, replace=False, p=bbox_weight)
        dst_box = None
        for box_id in bbox_id_list:
            if self.bbox_min_scale is not None:
                if bbox_area[box_id] < self.bbox_min_scale:
                    continue
            box = bboxes[box_id]
            box = convert_bbox_from_albumentations(box, 'pascal_voc', img_h, img_w)

            skip = False
            bboxes_abs = convert_bboxes_from_albumentations(bboxes, 'pascal_voc', img_h, img_w)
            for _id, box_2 in enumerate(bboxes_abs):
                if _id != box_id:
                    iou = image_util.calc_iou(box, box_2, mode=1)
                    if iou > 0.00001:
                        skip = True
                        break

            if not skip:
                box = tuple(map(int, box[:4]))
                x1, y1, x2, y2 = box
                for i in range(10):
                    dst_w = abs(x2 - x1)
                    dst_h = abs(y2 - y1)
                    if self.scale is not None:
                        scale = random.uniform(self.scale[0], self.scale[1])
                        scale = min(scale, img_w / dst_w, img_h / dst_h)
                        dst_w = int(dst_w * scale)
                        dst_h = int(dst_h * scale)
                    got = True
                    dst_left = random.uniform(0, img_w - dst_w)
                    dst_top = random.uniform(0, img_h - dst_h)
                    dst_box = (dst_left, dst_top, dst_left + dst_w, dst_top + dst_h)
                    for _id, box_2 in enumerate(bboxes_abs):
                        iou = image_util.calc_iou(box_2, dst_box, mode=1)
                        if iou > 0.00001:
                            got = False
                            break
                    if got:
                        break
                if got:
                    img_target = copy.deepcopy(im[y1:y2, x1:x2])
                    if self.scale is not None:
                        img_target = cv2.resize(img_target, (dst_w, dst_h))
                    bbox = tuple(map(int, dst_box[:4]))
                    if random.random() < 0.5:
                        x1, y1, x2, y2 = bbox
                        im[y1:y2, x1:x2] = img_target
                    else:
                        im_out = image_util.seamless_clone(img_target, im, bbox)
                        if im_out is None:
                            dst_box = None
                        else:
                            im = im_out
                    break
                else:
                    dst_box = None

        if dst_box is not None:
            bbox_new = convert_bbox_to_albumentations(dst_box, 'pascal_voc', img_h, img_w)
            bboxes_out.append((*bbox_new, *bboxes_out[box_id][4:]))
            class_labels_out.append(*bboxes_out[box_id][4:])
        kwargs.update({
            "image": im,
            "bboxes": bboxes_out,
            "class_labels": class_labels_out,
        })
        return kwargs


class MyGridDropout(GridDropout):
    def __init__(self, ratio_range=(0.15, 0.3), holes_number=(2, 10), ratio: float = 0.5, unit_size_min: int = None, unit_size_max: int = None,
                 holes_number_x: int = None, holes_number_y: int = None, shift_x: int = 0, shift_y: int = 0,
                 random_offset: bool = False, fill_value: int = 0, mask_fill_value: int = None,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(ratio, unit_size_min, unit_size_max, holes_number_x, holes_number_y, shift_x, shift_y,
                         random_offset, fill_value, mask_fill_value, always_apply, p)
        self.ratio_range = ratio_range
        self.holes_number = holes_number
        self.bboxes = []

    def apply(self, image, holes=(), **params):
        if len(self.bboxes) > 0:
            img_h, img_w = image.shape[:2]
            img_orig = image.copy()
            img_cutout = augmentations.dropout.functional.cutout(image, holes, self.fill_value)
            bboxes = convert_bboxes_from_albumentations(self.bboxes, 'pascal_voc', img_h, img_w)
            for bbox in bboxes:
                x1, y1, x2, y2 = tuple(map(int, bbox[:4]))
                img_cutout[y1:y2, x1:x2] = img_orig[y1:y2, x1:x2]
            return img_cutout
        else:
            return super().apply(image, holes, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_params(self):
        # self.fill_value = random.randint(0, 255)
        self.fill_value = [random.randint(0, 255) for _ in range(3)]
        if type(self.ratio_range) is tuple and len(self.ratio_range) == 2:
            self.ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        if type(self.holes_number) is tuple and len(self.holes_number) == 2:
            self.holes_number_x = random.randint(self.holes_number[0], self.holes_number[1])
            self.holes_number_y = random.randint(self.holes_number[0], self.holes_number[1])
        return super().get_params()

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_params_dependent_on_targets(self, params):
        if 'bboxes' in params and len(params['bboxes']) > 0:
            self.bboxes = params['bboxes']
        else:
            self.bboxes = []
        return super().get_params_dependent_on_targets(params)


class RandomCopyFpBbox(RandomCopyBbox):
    """Torchvision's variant of copy a bbox of the input and rescale it to some size,
        and then paste it into random location.

    Args:
        scale ((float, float)): range of size of the bbox resized
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, detector, scale=(0.8, 1.5), ratio=None, interpolation=cv2.INTER_LINEAR, bbox_min_scale=None,
                 always_apply=False, p=0.5, score_weight_power=0, copy_bbox_num=None, apply_times=1,
                 bbox_zoom_scale=None, rotate_limit=None):
        super().__init__(scale, ratio, interpolation, bbox_min_scale, always_apply, p)
        if isinstance(detector, str):
            self.detector = eval(detector)
        else:
            self.detector = detector
        self.score_weight_power = score_weight_power
        self.copy_bbox_num = copy_bbox_num
        self.apply_times = apply_times
        self.bbox_zoom_scale = bbox_zoom_scale
        self.rotate_limit = to_tuple(rotate_limit)

    def get_target_bboxes(self, params):
        img = params["image"]
        img_h, img_w = img.shape[:2]

        dst_bboxes = []
        if self.detector is not None:
            result = self.detector.detect_image(
                img, image_size=None, threshold=0.1, keep_ratio=False)
            bboxes = result.get('bboxes', [])
            bbox_num = len(bboxes)
            if bbox_num > 0:
                bboxes = convert_bboxes_to_albumentations(bboxes, 'pascal_voc', img_h, img_w)
                scores = result['scores']
                scores_sorted = np.sort(scores)[::-1]
                dst_num = min(4, np.size(scores_sorted))
                score_threshold = scores_sorted[dst_num - 1]
                scores[np.where(scores < score_threshold)] = 0
                # print(f'bbox_num: {bbox_num}, score_threshold: {score_threshold}, scores: {scores}')

                if self.copy_bbox_num is None:
                    num_list = np.ones(len(bboxes))
                    num_weight = num_list / num_list.sum()
                    num = np.random.choice(np.arange(1, len(bboxes) + 1),
                                           1, replace=False, p=num_weight)
                else:
                    num = min(self.copy_bbox_num, bbox_num)
                if self.score_weight_power != 0:
                    scores = np.power(scores, self.score_weight_power)
                bbox_weight = scores / scores.sum()
                dst_index = np.random.choice(np.arange(0, len(bboxes)),
                                             num, replace=True, p=bbox_weight)
                # print(f'dst_index:{dst_index}, num: {num}, bbox_weight: {bbox_weight}, len: {np.arange(0, len(bboxes))}')
                for i in dst_index:
                    dst_bboxes.append(bboxes[i])
        return dst_bboxes

    # @debug_util.print_function_time
    def apply_with_params(self, params, force_apply=False, **kwargs):
        if params is None:
            return kwargs
        self.update_params(params, **kwargs)
        im = kwargs.get('image', [])
        bboxes = kwargs.get('bboxes', [])
        img_h, img_w = im.shape[:2]
        bbox_num = len(bboxes)
        if bbox_num > 0:
            return kwargs

        dst_boxes = self.get_target_bboxes(kwargs)
        bbox_area = []
        for bbox in dst_boxes:
            area_size = image_util.get_box_size(bbox)
            bbox_area.append(area_size)
        for (box_id, box) in enumerate(dst_boxes):
            if self.bbox_min_scale is not None:
                if bbox_area[box_id] < self.bbox_min_scale:
                    continue
            box = convert_bbox_from_albumentations(box, 'pascal_voc', img_h, img_w)
            bboxes_abs = convert_bboxes_from_albumentations(bboxes, 'pascal_voc', img_h, img_w)
            box = box[:4]
            for i in range(self.apply_times):
                if self.bbox_zoom_scale is not None:
                    zoom_scale = random.uniform(self.bbox_zoom_scale[0], self.bbox_zoom_scale[1])
                    box_2 = image_util.zoom_box(box, zoom_scale, im)
                    x1, y1, x2, y2 = box_2
                else:
                    x1, y1, x2, y2 = box
                for i in range(10):
                    dst_w = abs(x2 - x1)
                    dst_h = abs(y2 - y1)
                    if self.scale is not None:
                        scale = random.uniform(self.scale[0], self.scale[1])
                        scale = min(scale, img_w / dst_w, img_h / dst_h)
                        dst_w = int(dst_w * scale)
                        dst_h = int(dst_h * scale)
                    got = True
                    dst_left = random.uniform(0, img_w - dst_w)
                    dst_top = random.uniform(0, img_h - dst_h)
                    dst_box = (dst_left, dst_top, dst_left + dst_w, dst_top + dst_h)
                    for _id, box_2 in enumerate(bboxes_abs):
                        iou = image_util.calc_iou(box_2, dst_box, mode=1)
                        if iou > 0.00001:
                            got = False
                            break
                    if got:
                        break
                if got:
                    x1, y1, x2, y2 = tuple(map(int, (x1, y1, x2, y2)))
                    if x2 <= x1 or y2 <= y1 or dst_w == 0 or dst_h == 0:
                        continue
                    img_target = copy.deepcopy(im[y1:y2, x1:x2])
                    try:
                        if self.scale is not None and img_target is not None:
                            if self.rotate_limit is not None and random.random() < 0.2:
                                rotate_limit = random.uniform(
                                    self.rotate_limit[0], self.rotate_limit[1])
                                img_target = image_util.rotate_image(
                                    img_target, rotate_limit, keep_size=False)
                            img_target = cv2.resize(img_target, (dst_w, dst_h))
                    except:
                        traceback.print_exc()
                    bbox = tuple(map(int, dst_box[:4]))
                    if random.random() < 0.6:
                        x1, y1, x2, y2 = bbox
                        im[y1:y2, x1:x2] = img_target
                    else:
                        flags_list = [cv2.NORMAL_CLONE, cv2.MIXED_CLONE, cv2.MONOCHROME_TRANSFER]
                        flags = random.choice(flags_list)
                        im_out = image_util.seamless_clone(img_target, im, bbox, flags=flags)
                        if im_out is not None:
                            im = im_out
        kwargs.update({
            "image": im,
        })
        return kwargs

    @property
    def targets_as_params(self):
        return ["image"]


class ToColor(ImageOnlyTransform):
    """Convert the input RGB image to specified color.

    Args:
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, color_format='bgr', always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.color_format = color_format

    def apply(self, img, **params):
        img = image_util.bgr_to_color(img, self.color_format, out_channel=3)
        return img

    def get_transform_init_args_names(self):
        return ("color_format",)
    

# if __name__ == "__main__":
#     import time
#     import cv2
#     import numpy as np
#     import albumentations as A
#     from yolox.exp.yolox_base import Exp

#     def yolo2voc(x, y, w, h, img_w, img_h):
#         center_x = round(float(x) * img_w)
#         center_y = round(float(y) * img_h)
#         bbox_width = round(float(w) * img_w)
#         bbox_height = round(float(h) * img_h)

#         xmin = str(int(center_x - bbox_width / 2))
#         ymin = str(int(center_y - bbox_height / 2))
#         xmax = str(int(center_x + bbox_width / 2))
#         ymax = str(int(center_y + bbox_height / 2))

#         return xmin, ymin, xmax, ymax

#     train_transform = [
#         "PadToAspectRatio(w2h_ratio=1.8, border_mode=0)",
#         "PadIfNeeded(min_height=320, min_width=576, border_mode=0, p=1)",
#         "RandomCopyBbox(p=1, bbox_min_scale=0.008, scale=(0.7, 1.5))",
#         "CropAndPad(percent=(-0.1, 0.1), keep_size=True, sample_independently=True, p=0.5)",
#         "MyGridDropout(ratio_range=(0.1, 0.5), holes_number=(1, 7), random_offset=True, p=0.05)",
#         "OneOf([\
#             MyRotate(limit=40, border_mode=0, p=0.1, adjust_factor=0.35),\
#             MyRotate(limit=70, border_mode=0, p=0.1, adjust_factor=0.35),\
#             ], p=0.1)",
#         "OneOf([\
#             MyRandomSizedBBoxSafeCrop(height=320, width=576, erosion_rate=0.0, scale=(0.1, 1), ratio=(1.4, 2.2), p=0.3, alignment='random'),\
#             MyRandomSizedBBoxSafeCrop(height=320, width=576, erosion_rate=0.0, scale=(0.1, 1), ratio=(1.7, 1.9), p=0.3, alignment='random'),\
#             RandomEraseBbox(scale=(0.65, 0.8), p=0.3, p_erase_all=1),\
#             ], p=0.2)",
#         "OneOf([\
#             RandomCropIfEmpty(height=320, width=576, scale=(0.001, 1), ratio=(0.01, 100.0), p=0.2),\
#             RandomCropIfEmpty(height=320, width=576, scale=(0.25, 1), ratio=(1.0, 2.5), p=0.2),\
#             ], p=0.3)",
#         "RandomEraseBbox(scale=(0.6, 0.75), p=0.2)",
#         "OneOf([\
#             Resize(p=0.3, height=320, width=576, interpolation=cv2.INTER_NEAREST),\
#             Resize(p=0.4, height=320, width=576, interpolation=cv2.INTER_LINEAR),\
#             Resize(p=0.1, height=320, width=576, interpolation=cv2.INTER_CUBIC),\
#             Resize(p=0.05, height=320, width=576, interpolation=cv2.INTER_LANCZOS4),\
#             Resize(p=0.05, height=320, width=576, interpolation=cv2.INTER_AREA),\
#         ], p=1)",
#         "ToGray(p=0.1)",
#         "OneOf([\
#             RandomBrightnessContrast(p=0.1, brightness_limit=(-0.3, 0), contrast_limit=(-0.3, 0), brightness_by_max=False),\
#             ColorJitter(p=0.2, brightness=0.3, contrast=0.5, saturation=0, hue=0),\
#             ColorJitter(p=0.2, brightness=0.3, contrast=0.5, saturation=0.1, hue=0.1),\
#             ], p=0.4)",
#         "OneOf([\
#             Downscale(scale_min=0.8, scale_max=0.97, p=0.2),\
#             JpegCompression(quality_lower=40, quality_upper=95, p=0.8),\
#             GaussNoise(var_limit=(10.0, 50.0), p=0.2),\
#             ], p=0.02)",
#         "OneOf([\
#             MotionBlur(blur_limit=(3, 5), p=0.6),\
#             GaussianBlur(blur_limit=(3, 5), sigma_limit=3, p=0.8),\
#             MedianBlur(blur_limit=(3, 5), p=0.2),\
#             Blur(blur_limit=(3, 5), p=0.2),\
#             ], p=0.05)",
#         "HorizontalFlip(p=0.5)",
#     ]

#     exp = Exp()
#     exp.train_transform = train_transform
#     transform = exp.get_transform()

#     imgpath = r"/code/data/s_BSD/hyh_bsd_yoloformat/train_640_0.004_221026_mirror/front/220916/ZQ490/train_clean_0.004_640/pos/00000000013305327467_pos_cze/00000000013305327467_02_03_6703_00_1564775872249073664_35.jpg"
#     img_size = [320, 576]
#     swap = (1, 2, 0)
#     image = cv2.imread(imgpath)
#     print("image: ", image.shape)
#     r = min(img_size[0] / image.shape[0], img_size[1] / image.shape[1])

#     resized_img = cv2.resize(
#         image,
#         (int(image.shape[1] * r), int(image.shape[0] * r)),
#         interpolation=cv2.INTER_LINEAR,
#     ).astype(np.uint8)
#     print("resized_img: ", resized_img.shape)
#     bboxes = [
#         [0.162646, 0.662119, 0.233249, 0.223281],
#         [0.745313, 0.227778, 0.087500, 0.227778]
#     ]
#     class_labels = ['person', 'mirror']

#     index = 0
#     while True:
#         # Augment an image
#         transformed = transform(image=resized_img, bboxes=bboxes, class_labels=class_labels)
#         transformed_image = transformed["image"]
#         transformed_bboxes = transformed['bboxes']
#         transformed_class_labels = transformed['class_labels']
#         height, width, _ = transformed_image.shape
#         print("transformed_image: ", transformed_image.shape)

#         # 画框
#         for i in range(len(transformed_bboxes)):
#             x, y, w, h = transformed_bboxes[i]
#             x1, y1, x2, y2 = yolo2voc(x, y, w, h, width, height)
#             cv2.rectangle(transformed_image, (int(x1), int(y1)),
#                             (int(x2), int(y2)), color=[255, 0, 0], thickness=2)
#         # for i in range(len(bboxes)):
#         #     x, y, w, h = bboxes[i]
#         #     x1, y1, x2, y2 = yolo2voc(x, y, w, h, width, height)
#         #     cv2.rectangle(transformed_image, (int(x1), int(y1)),
#         #                     (int(x2), int(y2)), color=[0, 0, 255], thickness=2)

#         cv2.imwrite('./transformed.png', transformed_image)
#         time.sleep(2)
#         index += 1
#         if index == 10:
#             break