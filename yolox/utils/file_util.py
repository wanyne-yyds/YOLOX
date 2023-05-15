import glob
import gzip
import os
import shutil
import traceback
from pathlib import Path
import random
import pickle

import cv2
# import debug_util

IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.bmp')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.flv', '.webm')


def replace_ext(filename, new_ext):
    (root, ext) = os.path.splitext(filename)
    return root + new_ext


def get_file_ext(filename):
    (root, ext) = os.path.splitext(filename)
    return ext


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def copy_files_to_single_dir(src_dirs, out_dir):
    dirs_list = []

    print('src_dirs: ', src_dirs)
    print('out_dir: ', out_dir)

    if list == type(src_dirs):
        dirs_list = src_dirs
    elif str == type(src_dirs) and os.path.exists(src_dirs):
        dirs_list.append(src_dirs)
    else:
        print('error, src_dirs not support this type!')
        return 0

    mkdirs(out_dir)
    index = 0
    for dir in dirs_list:
        if not os.path.exists(dir):
            print('ERROR, dir not exists: ', dir)
        for image_file in sorted(glob.glob(dir + '/**/*.jpg', recursive=True)):
            index += 1
            out_file = out_dir + os.sep + str(index) + '.jpg'
            shutil.copy(image_file, out_file)

    for dir in dirs_list:
        for image_file in sorted(glob.glob(dir + '/**/*.png', recursive=True)):
            index += 1
            img = cv2.imread(image_file)
            out_file = out_dir + os.sep + str(index) + '.jpg'
            cv2.imwrite(out_file, img)

    print('num = ', index)
    return index


def get_file_names(file_pattern, shuffle=False):
    """Parse list of file names from pattern, optionally shuffled.

    Args:
      file_pattern: File glob pattern, or list of glob patterns.
      shuffle: Whether to shuffle the order of file names.

    Returns:
      List of file names matching `file_pattern`.

    Raises:
      ValueError: If `file_pattern` is empty, or pattern matches no files.
    """
    if isinstance(file_pattern, list):
        if not file_pattern:
            raise ValueError("File pattern is empty.")
        file_names = []
        for entry in file_pattern:
            file_names.extend(glob.glob(entry))
    else:
        file_names = list(glob.glob(file_pattern))

    if not file_names:
        raise ValueError("No files match %s." % file_pattern)

    # Sort files so it will be deterministic for unit tests.
    if not shuffle:
        file_names = sorted(file_names)
    return file_names


def get_image_file_list(root, extensions=IMG_EXTENSIONS, recursive=True, full_path=True):
    if isinstance(extensions, str):
        extensions = [extensions]

    file_list = []
    if not isinstance(root, Path):
        root = Path(root)
    if recursive:
        file_iter = root.rglob('*.*')
    else:
        file_iter = root.glob('*.*')

    if full_path:
        file_list.extend([str(f) for f in file_iter if f.is_file() and f.suffix.lower() in extensions])
    else:
        file_list.extend([f.name for f in file_iter if f.is_file() and f.suffix.lower() in extensions])

    return file_list


def get_file_list(root, extensions='', recursive=True):
    return get_image_file_list(root, extensions, recursive)


def get_video_file_list(root, extensions=VIDEO_EXTENSIONS, recursive=True):
    return get_image_file_list(root, extensions, recursive)


def random_sample_from_dirs(input_dir, output_dir, is_moved=False, min_num=1, percent=None, percent_fn=None):
    cwd = Path(input_dir)
    dir_list = []
    for f in cwd.iterdir():
        if f.is_dir():
            dir_list.append(f)

    for d in dir_list:
        dir_name = d.stem
        file_list = get_file_list(d, recursive=False)
        num = len(file_list)
        if num == 0:
            continue
        if percent is not None:
            sample_num = int(percent * num)
        elif percent_fn is not None:
            sample_num = int(percent_fn(num))
        else:
            raise ValueError("percent and percent_fn should not be None both.")
        sample_num = max(sample_num, min_num)
        sample_list = random.sample(file_list, sample_num)
        for f in sample_list:
            save_dir = Path(output_dir).joinpath(dir_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            if is_moved:
                shutil.move(str(f), str(save_dir))
            else:
                shutil.copy(str(f), str(save_dir))


def get_relative_path(file_path, root_dir):
    full_path = os.path.dirname(file_path)
    rel_path = full_path[len(root_dir):]
    rel_path = rel_path.strip(os.path.sep)
    return rel_path


def dump_as_zip(content, file_name, compresslevel=9):
    with gzip.open(file_name, 'wb', compresslevel=compresslevel) as f:
        pickle.dump(content, f)


def load_from_zip(filename):
    with gzip.open(filename, 'rb') as f:
        content = pickle.load(f)
    return content


# @debug_util.print_function_time
def dump(content, file_name):
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(content, f)
    except:
        traceback.print_exc()


# @debug_util.print_function_time
def load(filename):
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content


def main():
    # import math
    # input_dir = r'C:\Users\admin\Desktop\temp\test\orig'
    # output_dir = r'C:\Users\admin\Desktop\temp\test\sample'
    # input_dir = r'H:\dataset\face_id\test\identify\FaceRecognition\test'
    # output_dir = r'H:\dataset\face_id\test\identify\FaceRecognition\sample'
    # random_sample_from_dirs(input_dir, output_dir, is_moved=True, percent_fn=math.log)
    pass


if __name__ == '__main__':
    main()

