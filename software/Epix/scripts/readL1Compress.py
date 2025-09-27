import numpy as np
import struct
try:
    from pyrogue.utilities.fileio import FileReader
except Exception:
    from pyrogue.utilities.filio import FileReader

def iter_l1bm_with_hdr(filename, chan=0x4, ssi_prepend=8):
    """
    读取 L1BitmaskCompressor 输出（文件中每帧前置 ssi_prepend=8 字节的 SSI 头）。
    返回: (orig32, ny, nx, thr, mask2d, vals)
    """
    fr = FileReader(filename)
    while True:
        fd = fr.next
        if fd is None:
            break
        if fd.channel != chan:
            continue

        p = fd.data                   # FileReader 已去掉文件层 8B(size/flags/err/chan)
        base = ssi_prepend            # 这里默认 8B SSI 头
        if len(p) < base + 32 + 24:
            continue

        orig32 = p[base : base+32]    # 原始 32B 相机头
        e1off  = base + 32            # E1BM 起始处（40）
        magic, ny, nx, thr, _rsv, count, mbytes = struct.unpack_from('<4sHHHHII', p, e1off)
        if magic != b'E1BM':
            continue

        total = base + 32 + 24 + mbytes + 2*count
        if len(p) < total:
            continue

        mask_off = e1off + 24
        mask = np.unpackbits(
            np.frombuffer(p, dtype=np.uint8, offset=mask_off, count=mbytes),
            bitorder='big'
        ).astype(bool)[:ny*nx].reshape(ny, nx)

        vals = np.frombuffer(p, dtype='<u2', offset=mask_off + mbytes, count=count)
        yield orig32, ny, nx, thr, mask, vals

def reconstruct(ny, nx, mask2d, vals, fill=0):
    img = np.full((ny, nx), fill, dtype=np.uint16)
    if vals.size:
        img[mask2d] = vals
    return img
