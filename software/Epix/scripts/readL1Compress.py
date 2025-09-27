import numpy as np
import struct
try:
    from pyrogue.utilities.fileio import FileReader
except Exception:
    from pyrogue.utilities.filio import FileReader

def iter_l1bm_with_hdr(filename, chan=0x4):
    fr = FileReader(filename)
    while True:
        fd = fr.next
        if fd is None: break
        if fd.channel != chan: continue
        p = fd.data
        if len(p) < 32 + 24: continue
        orig32 = p[:32]  # 原始相机头
        magic, ny, nx, thr, _rsv, count, mbytes = struct.unpack_from('<4sHHHHII', p, 32)
        if magic != b'E1BM': continue
        if len(p) != 32 + 24 + mbytes + 2*count: continue
        mask = np.unpackbits(
            np.frombuffer(p, dtype=np.uint8, offset=32+24, count=mbytes),
            bitorder='big'
        ).astype(bool)[:ny*nx].reshape(ny,nx)
        vals = np.frombuffer(p, dtype='<u2', offset=32+24+mbytes, count=count)
        yield orig32, ny, nx, thr, mask, vals

def reconstruct(ny, nx, mask2d, vals, fill=0):
    img = np.full((ny,nx), fill, dtype=np.uint16)
    if vals.size:
        img[mask2d] = vals
    return img
