import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Union



__mul = np.array([[128, 64, 32, 16, 8, 4, 2, 1]], dtype = np.uint8) #8位的伪码
def bev2bytes(bev: np.ndarray) -> bytes: 
    char_len = (len(bev) * len(bev[0]) - 1) // 8 + 1
    bins = np.array(bev, dtype = np.bool_)
    bins.resize((char_len, 8), refcheck = False)
    hex: NDArray[np.uint8] = (bins * __mul).sum(axis = 1, dtype = np.uint8)
    return hex.tobytes()

__mask = __mul.reshape((1, 8, 1))
def bytes2bev(sqlbytes: Union[bytes, Iterable[bytes]], shape: Iterable[int], blen: int) -> NDArray[np.bool_]: 
    """由二进制序列复原BEV

    Args:
        sqlbytes (Union[bytes, Iterable[bytes]]): 一条或多条二进制序列
        shape (Iterable[int]): BEV形状
        blen (int): 序列有效位数

    Returns:
        NDArray[np.bool_]: 二值BEV
    """
    if type(sqlbytes) == bytes: 
        return bytes2bev(tuple([sqlbytes]), shape, blen)[0]
    data = np.frombuffer(np.array(sqlbytes, dtype = np.bytes_).reshape(-1), dtype = np.uint8)
    im = np.bool_(data.reshape((-1, 1, blen)) & __mask)
    return im.reshape(-1, blen * 8)[:, :shape[0] * shape[1]].reshape(-1, shape[1], shape[0]).transpose(0, 2, 1)