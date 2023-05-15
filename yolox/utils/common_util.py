from __future__ import print_function

import os
import math
import traceback
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import numpy as np
from typing import Sequence
try:
    from reprlib import repr
except ImportError:
    pass


def dict_format(d: dict, precision=4):
    for k, n in d.items():
        if isinstance(n, (float, np.float32, np.float64, np.float16)):
            d[k] = round(n, precision)
        elif isinstance(n, dict):
            d[k] = dict_format(n, precision)
    return d


def sequence_format(seq, precision=4):
    seq_type = (tuple, list)
    seq_out = seq
    if isinstance(seq, seq_type):
        seq_out = []
        for n in seq:
            if isinstance(n, float):
                seq_out.append(round(n, precision))
            elif isinstance(n, seq_type):
                seq_out.append(sequence_format(n, precision))
            else:
                seq_out.append(n)
    return seq_out


def assign_id(d: dict, name, start=1):
    _id = d.get(name)
    if _id is None:
        _id = len(d) + start
        d.update({name: _id})
    return _id


def increase_count(stats: dict, name, n=1):
    count = stats.get(name, 0)
    count += n
    stats.update({name: count})


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


    # print('masks size: ', sys.getsizeof(masks))
    # print('masks total_size: ', common_util.total_size(masks))


def radian2degree(x):
    return x * 180 / math.pi


def degree2radian(x):
    return x * math.pi / 180


def getpid():
    pid = os.getpid()
    return pid


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


def np_stats(arr, name=''):
    try:
        d = {
            'min': arr.min(),
            'max': arr.max(),
            'mean': arr.mean(),
            'len': len(arr),
            'sum': sum(arr),
        }
        if len(name) > 0:
            d = ({name: d})
        d = dict_format(d, 4)
    except:
        d = {}
        print(arr)
        traceback.print_exc()
    return d


def dict_stats(d):
    result = {}
    for k, v in d.items():
        array = np.array(v)
        array = array.flatten()
        if array.size > 0:
            out = np_stats(array, k)
            result.update(out)
    return result


def dict_flatten(d):
    for k, v in d.items():
        if isinstance(v, (tuple, list)) and isinstance(v[0], (tuple, list)):
            d[k] = sum(v, [])
    return d


def histogram(input_list, bins=10, min=None, max=None, density=False):
    if len(input_list) == 0:
        return [], []
    a = np.array(input_list)
    if min is None:
        min = a.min()
    if max is None:
        max = a.max()
    a = np.clip(a, min, max)
    min = int(min * 2) / 2
    hist, bins = np.histogram(a, bins=bins, range=(min, max), density=density)
    return hist, bins


