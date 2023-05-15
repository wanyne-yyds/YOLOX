#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import platform
# from timebudget import timebudget

cfg_debug_enable = True
# cfg_debug_enable = False


def get_time_ms():
    if platform.system() == 'Linux':
        t = time.perf_counter()
    else:
        t = time.clock()  # waring, DO NOT use time.clock() in linux, or you will get wrong result.
    return t


# def html_tags(tag_name):
#     def wrapper_(func):
#         def wrapper(*args, **kwargs):
#             content = func(*args, **kwargs)
#             return "<{tag}>{content}</{tag}>".format(tag=tag_name, content=content)
#         return wrapper
#     return wrapper_
#
# @html_tags('b')
# def hello(name='Toby'):
#     return 'Hello {}!'.format(name)


def print_time(tag=''):
    def wrapper_(fn):
        def _wrapper(*args, **kwargs):
            begin = get_time_ms()
            result = fn(*args, **kwargs)
            if cfg_debug_enable:
                print(f"{tag} {fn.__name__} took {1000 * (get_time_ms() - begin):.3f} ms")
            return result
        return _wrapper
    return wrapper_


def print_function_time(fn):
    def _wrapper(*args, **kwargs):
        begin = get_time_ms()
        result = fn(*args, **kwargs)
        if cfg_debug_enable:
            print("{} took {:.3f} ms".format(fn.__name__, 1000 * (get_time_ms() - begin)))
        return result

    return _wrapper


def log_init(log_file, level=logging.INFO):
    # LOG_FORMAT = "%(asctime)s %(message)s"
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger()  # 不加名称设置root logger

    # 使用FileHandler输出测试结果到文件
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info("log_init ready, log_file: " + log_file)
    return logger


class Logger(object):
    def __init__(self, logger_name=None, filename=None, *args, **kwargs):
        '''
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        '''

        # 创建一个logger
        if filename is None:
            file = 'log.txt'
        else:
            file = filename
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        # formatter = logging.Formatter('[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
        formatter = logging.Formatter("%(asctime)s %(message)s")

        if file:
            hdlr = logging.FileHandler(file, 'a', encoding='utf-8')
            hdlr.setLevel(logging.INFO)
            hdlr.setFormatter(formatter)
            self.logger.addHandler(hdlr)

        strhdlr = logging.StreamHandler()
        strhdlr.setLevel(logging.INFO)
        strhdlr.setFormatter(formatter)
        self.logger.addHandler(strhdlr)

        if file: hdlr.close()
        strhdlr.close()

    def getlogger(self):
        return self.logger

    @staticmethod
    def attach_output(logger=None, log_file=None, level=logging.INFO):
        LOG_FORMAT = "%(asctime)s %(message)s"
        # LOG_FORMAT = "%(message)s"
        logging.basicConfig(level=level, format=LOG_FORMAT)

        if logger is None:
            logger = logging.getLogger()  # 不加名称设置root logger
        logger.propagate = False
        if logger.hasHandlers():
            logger.handlers.clear()

        if log_file:
            # 使用FileHandler输出测试结果到文件
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            formatter = logging.Formatter(LOG_FORMAT)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            fh.close()

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
        console.close()

        return logger


def init_log_in_dir(work_path: object, script_filename: object) -> object:
    if os.path.exists(work_path):
        script = os.path.basename(script_filename)
        (script, _) = os.path.splitext(script)
        name = "{}.txt".format(script)
        log_file = os.path.join(work_path, name)
        log_init(log_file)

