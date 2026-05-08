import os
import sqlite3
import numpy as np
from typing import Union, Iterable
import cv2

__mul = np.array([[128, 64, 32, 16, 8, 4, 2, 1]], dtype=np.uint8)
__mask = __mul.reshape((1, 8, 1))

def bytes2bev(sqlbytes: Union[bytes, Iterable[bytes]], shape: Iterable[int], blen: int) -> np.ndarray:
    if type(sqlbytes) == bytes:
        return bytes2bev(tuple([sqlbytes]), shape, blen)[0]
    data = np.frombuffer(np.array(sqlbytes, dtype=np.bytes_).reshape(-1), dtype=np.uint8)
    im = np.bool_(data.reshape((-1, 1, blen)) & __mask)
    return im.reshape(-1, blen * 8)[:, :shape[0] * shape[1]].reshape(-1, shape[1], shape[0]).transpose(0, 2, 1)

def pre_process(image):
    return (image / 255.0 - 0.5) * 2.0

print("Ready.")
