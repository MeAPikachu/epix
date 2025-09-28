import struct
import numpy as np

E1_FMT = "<4sHHHHII"
E1_SZ  = struct.calcsize(E1_FMT)  # 20B
# This script is used to read the L1 Bitmap Compress files; 

def iter_l1bm_frames(path):
    """
    Read the compressed frames
    """
    with open(path, "rb") as f:
        while True:
            ssi = f.read(8)
            if not ssi: break
            if len(ssi) < 8: break

            orig32 = f.read(32)
            if len(orig32) < 32: break

            e1hdr = f.read(E1_SZ)                 # Read the head 
            if len(e1hdr) < E1_SZ: break
            magic, ny, nx, thr, _rsv, count, mbytes = struct.unpack(E1_FMT, e1hdr)
            if magic != b"E1BM": break

            _gap4 = f.read(4)                     # Discard the useless information
            if len(_gap4) < 4: break

            mask_bytes = f.read(mbytes)
            if len(mask_bytes) < mbytes: break

            vals_bytes = f.read(count * 2)
            if len(vals_bytes) < count * 2: break

            mask = np.unpackbits(np.frombuffer(mask_bytes, np.uint8), bitorder="big")
            mask = mask[:ny*nx].reshape(ny, nx).astype(bool)
            vals = np.frombuffer(vals_bytes, dtype="<u2", count=count)

            yield ssi, orig32, ny, nx, thr, mask, vals

def reconstruct(ny, nx, mask2d, vals, fill=0):
    img = np.full((ny, nx), fill, dtype=np.uint16)
    if vals.size:
        img[mask2d] = vals
    return img

# Example function
def read_first(path):
    it = iter_l1bm_frames(path)
    try:
        ssi, orig32, ny, nx, thr, mask, vals = next(it)
    except StopIteration:
        print("No frames left")
        return
    img = reconstruct(ny, nx, mask, vals, fill=0)
    print(f"size={ny}x{nx}, thr={thr}, hits={vals.size} ({vals.size/(ny*nx):.1%})")
    return ssi, orig32, img
