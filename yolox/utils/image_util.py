import copy
import logging
import os
import pprint
import random
import shutil
import sys
import traceback
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from skimage import transform
from PIL import Image

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
import utils.file_util as file_util
import utils.debug_util as debug_util
# from detect import detector_factory
from utils.common_util import to_tuple

color_list = ['blue', 'green', 'purple', 'orange', 'red', 'black', 'yellow']
# color_list = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]


def bgr2gray(dataset_dir, output_dir):
    count = 0
    save_dir = output_dir
    print('dataset_dir: ', dataset_dir)
    print('output_dir: ', output_dir)

    for root, dirs, files in sorted(os.walk(dataset_dir)):
        for file_name in files:
            list_name = file_name.split('.')
            if len(list_name) < 2:
                continue
            if list_name[1] == 'jpg':
                count = count + 1
                if len(dataset_dir) < len(root):
                    relative_path = root[len(dataset_dir):]
                    file_dir = save_dir + relative_path
                else:
                    file_dir = save_dir
                file_util.mkdirs(file_dir)
                image_out_base = file_dir + os.sep + file_name
                image_src = root + os.sep + file_name

                img = cv2.imread(image_src)
                if img is None:
                    print('error, image_src: ', image_src)
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.merge([gray, gray, gray])
                cv2.imwrite(image_out_base, img)

                xml_src = file_util.replace_ext(image_src, '.xml')
                xml_out = file_util.replace_ext(image_out_base, '.xml')
                if os.path.exists(xml_src):
                    shutil.copyfile(xml_src, xml_out)

    print('bgr2gray, count = ', count)
    return count


def get_face_area(landmark, image, factor):
    out_img = None
    feature_info = landmark.detect_face(image)
    if feature_info and feature_info.__contains__('box_dict'):
        box_dict = feature_info['box_dict']
        if box_dict.get('face'):
            box = box_dict['face']
            border = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
            face_enlarged = zoom_box(box, factor, border=border)
            face_enlarged = map(int, face_enlarged)
            (left, top, right, bottom) = face_enlarged

            w = abs(right - left)
            h = abs(bottom - top)
            if w > 100 and h > 100:
                out_img = image[top:bottom, left:right]
            else:
                pass

    return out_img


def get_object_list(xml_file):
    object_list = []
    tree = ET.parse(xml_file)
    xml_root = tree.getroot()
    for member in xml_root.findall('object'):
        name = member.find('name').text
        for num in member.findall('bndbox'):
            xmin = int(num.find('xmin').text)
            ymin = int(num.find('ymin').text)
            xmax = int(num.find('xmax').text)
            ymax = int(num.find('ymax').text)
            object_list.append((name, xmin, ymin, xmax, ymax))
    return object_list


def zoom_box(box, factor, border=None):
    (left, top, right, bottom) = box
    w = abs(right - left)
    h = abs(bottom - top)

    if type(factor) is float or type(factor) is int:
        factor = (factor, factor, factor, factor)

    left = left - w * factor[0]
    top = top - h * factor[1]
    right = right + w * factor[2]
    bottom = bottom + h * factor[3]

    box_out = [left, top, right, bottom]
    if border is not None:
        box_out = fix_box_border(box_out, border)

    return tuple(box_out)


def fix_box_border(box, border):
    image_box = get_image_box(border)
    if image_box:
        border = image_box

    if type(border) is tuple or type(border) is list:
        box = list(box)
        box[0] = max(box[0], border[0])
        box[1] = max(box[1], border[1])
        box[2] = min(box[2], border[2])
        box[3] = min(box[3], border[3])
    else:
        print('error, fix_box_border type(border): ', type(border))

    return box


def get_box_aspect_ratio(box):
    result = 0
    (left, top, right, bottom) = box
    w = abs(right - left)
    h = abs(bottom - top)
    if w > 0:
        result = h / w
    return result


def square_box(box):
    # convert box to square
    box = list(box)
    h = box[3] - box[1]
    w = box[2] - box[0]
    l = max(w, h)
    box[0] = box[0] + w * 0.5 - l * 0.5
    box[1] = box[1] + h * 0.5 - l * 0.5
    box[2] = box[2] + l * 0.5 - w * 0.5
    box[3] = box[3] + l * 0.5 - h * 0.5

    result = [int(i) for i in box]
    return tuple(result)


def get_box_width(box):
    (left, top, right, bottom) = box
    w = abs(right - left)
    return w


def get_box_height(box):
    (left, top, right, bottom) = box
    h = abs(bottom - top)
    return h


def get_box_size(box):
    (left, top, right, bottom) = box[:4]
    w = abs(right - left)
    h = abs(bottom - top)
    return w * h


def get_bbox_area(box):
    (left, top, right, bottom) = box[:4]
    w = abs(right - left)
    h = abs(bottom - top)
    return w * h


def crop_box_image(box, image):
    b = convert_type(box, int)
    box_image = image[b[1]:b[3], b[0]:b[2]]
    return box_image


def letter_box(box, dst_size):
    output_size = dst_size
    (left, top, right, bottom) = box
    (target_w, target_h) = dst_size
    width = right - left
    height = bottom - top
    if width > target_w or height > target_h:
        scale_w = width / target_w
        scale_h = height / target_h
        scale = max(scale_w, scale_h)
        output_size = (int(target_w * scale), int(target_h * scale))
    return output_size


def bbox_inverse(offset_x, offset_y, bboxes, scale=1):
    if bboxes is None:
        return bboxes

    bboxes_orig = copy.deepcopy(bboxes)
    if isinstance(bboxes, (tuple, list)):
        bboxes = np.asarray(bboxes)
        bboxes = bboxes[np.newaxis, :]

    if isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
        if isinstance(scale, (tuple, list)):
            bboxes[:, 0::2] /= scale[0]
            bboxes[:, 1::2] /= scale[1]
        else:
            bboxes /= scale
        bboxes[:, 0::2] += offset_x
        bboxes[:, 1::2] += offset_y

    if isinstance(bboxes_orig, (tuple, list)):
        bboxes = bboxes.tolist()
    return bboxes


def points_inverse(offset_x, offset_y, points, scale=1):
    if points is None:
        return points

    input_type = type(points)
    if isinstance(points, (tuple, list)):
        points = np.asarray(points)

    if isinstance(points, np.ndarray):
        if isinstance(scale, (tuple, list)):
            points[:, 0] /= scale[0]
            points[:, 1] /= scale[1]
        else:
            points /= scale
        points[:, 0] += offset_x
        points[:, 1] += offset_y

    if isinstance(input_type, (tuple, list)):
        points = points.tolist()
    return points


def get_crop_coords_from_bbox(bbox, image, crop_size, alignment='center'):
    letter_size = letter_box(bbox, crop_size)
    dst_w, dst_h = letter_size
    height, width, _ = image.shape
    (left, top, right, bottom) = bbox
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    dst_box = ((cx - dst_w / 2), (cy - dst_h / 2), (cx + dst_w / 2), (cy + dst_h / 2))
    (dst_left, dst_top, dst_right, dst_bottom) = dst_box

    delta_x = -min(0, dst_left) - max(0, (dst_right - width))
    delta_y = -min(0, dst_top) - max(0, (dst_bottom - height))

    dst_box = (dst_left + delta_x, dst_top + delta_y, dst_right + delta_x, dst_bottom + delta_y)
    if alignment != 'center':
        (x1, y1, x2, y2) = dst_box
        dst_w = abs(x2 - x1)
        dst_h = abs(y2 - y1)
        dst_left = random.uniform(max(0, (right - dst_w)), min(left, width - dst_w))
        dst_top = random.uniform(max(0, (bottom - dst_h)), min(top, height - dst_h))
        dst_box = (dst_left, dst_top, dst_left + dst_w, dst_top + dst_h)
    dst_box = fix_box_border(dst_box, image)
    dst_box = convert_type(dst_box, int)
    return dst_box


def convert_type(box, new_type):
    result = [new_type(i) for i in box]
    return tuple(result)


def convert_round(p):
    result = [round(i) for i in p]
    return tuple(result)


def get_image_size(image):
    is_image = hasattr(image, 'shape')
    if is_image:
        height, width, _ = image.shape
        return height * width
    else:
        return 0


def get_image_shape(image):
    h, w = image.shape[:2]
    return h, w


def get_image_box(image):
    box = None
    is_image = hasattr(image, 'shape')
    if is_image:
        height, width, _ = image.shape
        box = (0, 0, (width - 1), (height - 1))
    return box


def get_box_center(box):
    w = get_box_width(box)
    h = get_box_height(box)
    x = box[0] + w / 2
    y = box[1] + h / 2
    return (x, y)


def union_of_bboxes(bboxes):
    """Calculate union of bounding boxes.
    Args:
        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    first_time = True
    for bbox in bboxes:
        if first_time:
            x1, y1, x2, y2 = bbox[:4]
            first_time = False
            continue
        x_min, y_min, x_max, y_max = bbox[:4]
        x1, y1 = np.min([x1, x_min]), np.min([y1, y_min])
        x2, y2 = np.max([x2, x_max]), np.max([y2, y_max])
    return x1, y1, x2, y2


def get_bboxes_edge_ratio(border, bboxes):
    is_image = hasattr(border, 'shape')
    if is_image:
        height, width, _ = border.shape
        border = (0, 0, width, height)
    bbox = union_of_bboxes(bboxes)
    x, y, x2, y2 = bbox
    w = get_box_width(border)
    h = get_box_height(border)
    ratio = min(x / w, (w - x2) / w, y / h, (h - y2) / h)
    return ratio


def get_face_part_box(image, detector, which, square_box=False):
    box = None
    feature_info = detector.detect_face(image)
    if feature_info and feature_info.get('box_dict'):
        box = detector.get_feature_box(feature_info, which, square_box)
    return box


def calc_iou(box1, box2, mode=0):
    """

    Args:
        box1: left, top, right, bottom
        box2: left, top, right, bottom
        mode: 0: standard iou, 1: iou of box1
    Returns:

    """
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    if inter == 0:
        iou = 0
    else:
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        if mode == 1:
            union = min(area1, area2)
        else:
            union = area1 + area2 - inter
        iou = inter / union
    return iou


def rotate_image(image, range=(-5, 5), center=None, keep_size=False):
    range = to_tuple(range)
    box = get_image_box(image)
    if center is None:
        center = get_box_center(box)
    degree = random.uniform(range[0], range[1])
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    if keep_size is False:
        rotated = cv2.warpAffine(image, M, (get_box_height(box), get_box_width(box)))
    else:
        (h, w) = image.shape[:2]
        (cX, cY) = center
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotated = cv2.warpAffine(image, M, (nW, nH))
        # perform the actual rotation and return the image
    return rotated


# @debug_util.print_function_time
def fill_rect(img, rect, fill_value=(0, 0, 0)):
    """
    fill_rect with fill_value
    Args:
        img: image
        rect: (x, y, w, h)
        fill_value: color, if None, fill random color
    """

    (x, y, w, h) = convert_type(rect, int)
    if type(fill_value) is int:
        color = (fill_value, fill_value, fill_value)
    else:
        color = fill_value

    # very very slow, abandon it.
    # for i in range(int(h)):
    #     for j in range(int(w)):
    #         img[int(y) + i, int(x) + j] = color
    img[y:(y + h), x:(x + w)] = color


def fill_bbox(img, bbox, fill_value=(0, 0, 0)):
    """
    fill_bbox with fill_value
    Args:
        img: image
        rect: (x, y, w, h)
        fill_value: color, if None, fill random color
    """

    (x, y, x2, y2) = convert_type(bbox, int)
    if type(fill_value) is int:
        color = (fill_value, fill_value, fill_value)
    else:
        color = fill_value
    img[y:y2, x:x2] = color


def random_erasing(img, rect, fill_value=(0, 0, 0), p=0.2):
    if random.random() <= p:
        fill_rect(img, rect, fill_value=fill_value)
        return True
    else:
        return False


# @debug_util.print_function_time
def crop_sub_bbox_from_bbox(img, scale_ratio, bbox=None, w_start=None, h_start=None):
    x = y = 0
    if bbox is not None:
        (left, top, right, bottom) = bbox
        (x, y, w, h) = (left, top, (right - left), (bottom - top))
    else:
        h, w = get_image_shape(img)
    area = w * h
    if isinstance(scale_ratio, (tuple, list)):
        scale = random.uniform(*scale_ratio)

    if isinstance(w_start, (tuple, list)):
        w_start_min = w_start[0] * w
        w_start_max = w_start[1] * w
    else:
        w_start_min = 0
        w_start_max = w

    if isinstance(h_start, (tuple, list)):
        h_start_min = h_start[0] * h
        h_start_max = h_start[1] * h
    else:
        h_start_min = 0
        h_start_max = h

    target_area = scale * area
    dst_w = int(random.uniform(scale * w, (w - w_start_min)))
    dst_w = max(dst_w, 1)
    dst_h = int(target_area / dst_w)
    dst_h = min(h, dst_h)
    h_start = random.uniform(max(0, h_start_min), min((h - dst_h), h_start_max))

    w_start = random.uniform(max(0, w_start_min), min((w - dst_w), w_start_max))
    dst_w = min(dst_w, (w - w_start))

    rect = (x + w_start, y + h_start, dst_w, dst_h)
    bbox_out = xywh2xyxy(rect)
    bbox_out = convert_round(bbox_out)
    return bbox_out


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def random_fill_rect(img, scale_ratio, bbox=None, w_start=None, h_start=None, fill_value=(0, 0, 0)):
    bbox_new = crop_sub_bbox_from_bbox(img, scale_ratio, bbox, w_start, h_start)
    return fill_bbox(img, bbox_new, fill_value=fill_value)


def _bgr_to_nv12(bgr):
    h, w = bgr.shape[:2]
    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    yuv_nv12 = copy.deepcopy(yuv_i420)
    yuv_nv12[h::, 0::2] = yuv_i420[h:(h + h // 4):, 0::].reshape((h // 2, w // 2))
    yuv_nv12[h::, 1::2] = yuv_i420[(h + h // 4)::, 0::].reshape((h // 2, w // 2))
    return yuv_nv12


def _nv12_to_bgr(yuv):
    # return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def bgr_to_nv12(bgr, out_channel=1):
    yuv = _bgr_to_nv12(bgr)
    if out_channel == 3:
        h, w = bgr.shape[:2]
        yuv = np.vstack((yuv, yuv))
        yuv = np.reshape(yuv, (3, h, w))
        yuv = yuv.transpose(1, 2, 0)
    return yuv


def nv12_to_bgr(yuv):
    if len(yuv.shape) == 3:
        yuv_0 = yuv[:, :, 0]
        yuv_1 = yuv[:, :, 1]
        h, w = yuv_1.shape
        yuv_1 = yuv_1[:h//2, :]
        yuv = np.vstack((yuv_0, yuv_1))

    img = _nv12_to_bgr(yuv)
    return img


def bgr_to_i420(bgr, out_channel=1):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    if out_channel == 3:
        h, w = bgr.shape[:2]
        yuv = np.vstack((yuv, yuv))
        yuv = np.reshape(yuv, (3, h, w))
        yuv = yuv.transpose(1, 2, 0)
    return yuv


def i420_to_bgr(yuv):
    if len(yuv.shape) == 3:
        yuv_0 = yuv[:, :, 0]
        yuv_1 = yuv[:, :, 1]
        h, w = yuv_1.shape
        yuv_1 = yuv_1[:h//2, :]
        yuv = np.vstack((yuv_0, yuv_1))

    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return img


def bgr_to_gray(bgr, out_channel=1):
    img = bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if out_channel == 3:
        img = cv2.merge([img, img, img])
    return img


def gray_to_bgr(gray):
    img = gray
    if len(gray.shape) == 2:
        img = cv2.merge([gray, gray, gray])
    return img


def bgr_to_color(bgr, color_format, out_channel=1):
    img = bgr
    if color_format == 'nv12':
        img = bgr_to_nv12(img, out_channel=out_channel)
    elif color_format == 'gray':
        img = bgr_to_gray(img, out_channel=out_channel)
    elif color_format == 'I420':
        img = bgr_to_i420(img, out_channel=out_channel)
    elif color_format == 'bgr':
        img = bgr
    else:
        print("bgr_to_color Invalid color format: {color_format}")
        img = None
    return img


def color_to_bgr(img, color_format):
    if color_format == 'nv12':
        img = nv12_to_bgr(img)
    elif color_format == 'gray':
        img = gray_to_bgr(img)
    elif color_format == 'I420':
        img = i420_to_bgr(img)
    elif color_format == 'bgr':
        pass
    else:
        print("color_to_bgr Invalid color format: {color_format}")
        img = None
    return img


def show_bbox(img, bboxes, scores=None, labels=None, class_names=None, font_scale=0.5, show_center_point=False,
              color=None, text_position='up', show_text_rect=True):
    if labels is not None:
        label_len = len(labels)
    for idx, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = map(int, bbox)
        if scores is not None and len(scores) > 0:
            score = scores[idx]
        else:
            score = None

        # show text
        label = 0
        if labels is not None:
            if isinstance(labels, (tuple, list, np.ndarray)) and idx < label_len:
                label = labels[idx]
            elif isinstance(labels, int):
                label = labels

        if class_names is not None and len(class_names) > 0:
            title = class_names[label]
        else:
            title = f'{label}'
        if score is not None:
            score_txt = f': {(score * 100):.1f}'
        else:
            score_txt = ''
        text = title + score_txt
        if color is None:
            color = get_color(label)
        txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if show_text_rect:
            txt_size = cv2.getTextSize(text, font, font_scale, 2)[0]
            cv2.rectangle(
                img,
                (x0, y0 - txt_size[1] - 5),
                (x0 + txt_size[0] + txt_size[1], y0 - 2),
                color,
                -1,
            )
        if text_position == 'down':
            position = (x0, y1 + txt_size[1] + 2)
        else:
            position = (x0, y0 - 5)

        cv2.putText(img, text, position, font, font_scale, txt_color, thickness=1)

        # show bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        if show_center_point:
            # draw center Point
            left, top, right, bottom = x0, y0, x1, y1
            width = right - left
            height = bottom - top
            center_x = left + width // 2
            center_y = top + height // 2
            cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), -1)

    return img


def draw_bbox(im, bboxes, scores, font_scale=1, draw_center_point=False, color=(0, 255, 0), text_position='up'):
    for bbox, score in zip(bboxes, scores):
        left, top, right, bottom = map(int, bbox)
        cv2.rectangle(im, (left, top), (right, bottom), color, 2)
        text = f'{score:.2f}'
        if text_position == 'down':
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, font_scale, 2)[0]
            position = (left, bottom + txt_size[1] + 2)
        else:
            position = (left, top - 2)
        cv2.putText(im, text, position, cv2.FONT_HERSHEY_DUPLEX, font_scale, color)

        if draw_center_point:
            width = right - left
            height = bottom - top
            center_x = left + width // 2
            center_y = top + height // 2
            cv2.circle(im, (center_x, center_y), 2, color, -1)


_COLORS = (
    np.array(
        [
            0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188,
            0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
            1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
            0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
            0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500,
            0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
            0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500,
            1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
            0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
            0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
            1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000,
            0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
            0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167,
            0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
            0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571,
            0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def similarity_transform(img, src, dst, output_size=(96, 112)):
    """

    Args:
        img: input image
        src: input image key points
        dst: template's points
        output_size:

    Returns: output image

    """
    src, dst = np.array(src), np.array(dst)
    tform = transform.SimilarityTransform()
    # 程序直接估算出转换矩阵M
    tform.estimate(src, dst)
    # M, _ = cv2.estimateAffinePartial2D(src, dst)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, output_size, borderValue=0.0)
    return warped


def read_image(path, method=None):
    if method is None:
        if path.isascii():
            method = 1
        else:
            method = 2

    if method == 1:
        img = cv2.imread(path)
    elif method == 2:
        from PIL import Image
        try:
            img = Image.open(path)
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except:
            traceback.print_exc()
    else:
        import imageio
        img = imageio.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def write_image(filename, img, method=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if method is None:
        if filename.isascii():
            method = 1
        else:
            method = 2

    if method == 1:
        cv2.imwrite(filename, img)
    else:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.save(filename, quality=95)


# @debug_util.print_function_time
def get_quality(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # resLap = cv2.Laplacian(gray, cv2.CV_64F)
    resLap = cv2.Laplacian(gray, cv2.CV_16S)
    score = resLap.var()
    return score


def get_color(idx):
    if idx < len(_COLORS):
        color = (_COLORS[idx] * 255).astype(np.uint8).tolist()
    else:
        color = None
    return color


def get_color2(idx):
    if idx < len(color_list):
        color = color_list[idx]
    else:
        color = get_color(idx)
        if color is not None:
            color = tuple(map(lambda x: x / 255, color))
    return color


def draw_points(points, img, color=(255, 0, 0), radius=1, do_round=False):
    for (x, y) in points:
        if do_round:
            (x, y) = (round(x), round(y))
        cv2.circle(img, (x, y), radius, color)


@debug_util.print_function_time
def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    fimg = 255 * fimg
    fimg = fimg.astype(np.uint8)
    return fimg


def seamless_clone(src, dst, bbox, mask=None, flags=cv2.NORMAL_CLONE):
    if mask is None:
        mask = 255 * np.ones(src.shape, src.dtype)
    if bbox is None:
        h, w = dst.shape[:2]
        x1, y1, x2, y2 = (0, 0, w, h)
    else:
        x1, y1, x2, y2 = bbox[:4]
    # Location of the center of the source image in the destination image.
    center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
    try:
        dst = cv2.seamlessClone(src, dst, mask, center, flags)
    except:
        traceback.print_exc()
        return None
    return dst

def get_box_aspect_ratio(box):
    result = 0
    (left, top, right, bottom) = box
    w = abs(right - left)
    h = abs(bottom - top)
    if w > 0:
        result = h / w
    return result

def checkXmlClass():
    from tqdm import tqdm
    from imutils.paths import list_files
    # xml_dir = r'C:\Users\RJBSJ\Desktop\s_ADAS\Annotations\train'
    xml_dir = '/data_ssd/user/cyh_ssd/dataset/bsd/bsd_ckn/Annotations'
    nameS = set()
    for xml_file in tqdm(list(list_files(xml_dir))):
        try:
            object_list = get_object_list(xml_file)#((name, xmin, ymin, xmax, ymax))
            for object in object_list:
                nameS.add(object[0])
        except:
            # print(xml_file)
            pass

    print(nameS, len(nameS))

    nameDict = {'person':0, 'rearview mirror':1, 'personD':2}
    i = 3
    for name in nameS:
        if nameDict.get(name, None) != None:
            continue
        nameDict[name] = i
        i += 1
    for name in nameDict.keys():
        print(name)
    print(nameDict)

def main():
    checkXmlClass()

if __name__ == '__main__':
    main()
