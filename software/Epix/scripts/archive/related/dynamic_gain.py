#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import struct
import numpy as np
#from tqdm import tqdm
import time 

# =========================
# L1BM binary layout (as inferred from your parser)
# =========================
E1_FMT = "<4sHHHHII"  # magic(4s), ny(u16), nx(u16), thr(u16), rsv(u16), count(u32), mbytes(u32)
E1_SZ = struct.calcsize(E1_FMT)


def iter_l1bm_frames(path):
    """
    Iterate frames stored in a single L1*.dat file.
    Each yielded item is:
      (ssi_8bytes, orig_u32[8], ny, nx, thr, mask2d(bool), vals(u16[count]))
    """
    with open(path, "rb") as f:
        while True:
            # 8 bytes SSI
            ssi = f.read(8)
            if not ssi or len(ssi) < 8:
                break

            # 32 bytes orig header (8 x u32)
            orig32 = f.read(32)
            if len(orig32) < 32:
                break
            orig_u32 = np.frombuffer(orig32, dtype="<u4", count=8)

            # E1BM header
            e1hdr = f.read(E1_SZ)
            if len(e1hdr) < E1_SZ:
                break

            magic, ny, nx, thr, _rsv, count, mbytes = struct.unpack(E1_FMT, e1hdr)
            if magic != b"E1BM":
                break

            # 4 bytes padding / gap
            _gap4 = f.read(4)
            if len(_gap4) < 4:
                break

            # bitmask payload
            mask_bytes = f.read(mbytes)
            if len(mask_bytes) < mbytes:
                break

            # values payload (u16 * count)
            vals_bytes = f.read(count * 2)
            if len(vals_bytes) < count * 2:
                break

            # unpack bitmask -> (ny,nx) boolean mask
            mask = np.unpackbits(np.frombuffer(mask_bytes, np.uint8), bitorder="big")
            mask = mask[: ny * nx].reshape(ny, nx).astype(bool)

            # unpack values
            vals = np.frombuffer(vals_bytes, dtype="<u2", count=count)

            yield ssi, orig_u32, ny, nx, thr, mask, vals


def build_pixel_hist(paths, vmin=0, vmax=2200):
    """
    Build per-pixel histogram hist(y,x,bin) by iterating frames in given files.
    vmin/vmax define histogram edges on integer ADC values.
    Returns: (hist, ny, nx, n_frames)
    """
    nbins = vmax - vmin + 1
    hist_flat = None
    ny = nx = None
    n_frames = 0

    def all_frames():
        for path in paths:
            for frame in iter_l1bm_frames(path):
                yield path, frame

    for path, (_, _, ny_i, nx_i, _, mask2d, vals) in all_frames():
        # allocate on first frame
        if hist_flat is None:
            ny, nx = ny_i, nx_i
            hist_flat = np.zeros((ny * nx, nbins), dtype=np.uint32)
        else:
            # enforce consistent frame shape
            if ny_i != ny or nx_i != nx:
                raise RuntimeError(
                    f"{path}: inconsistent frame shape: got {ny_i}x{nx_i}, expected {ny}x{nx}"
                )

        n_frames += 1

        # no hits in this frame
        if vals.size == 0:
            continue

        # flatten mask to align with vals order (assumed 1-to-1)
        mask_flat = mask2d.ravel()
        hit_idx_all = np.nonzero(mask_flat)[0]  # length should equal vals.size

        # keep values within [vmin, vmax]
        vals_i = vals.astype(np.int32, copy=False)
        valid = (vals_i >= vmin) & (vals_i <= vmax)
        if not np.any(valid):
            continue

        pix_idx = hit_idx_all[valid]
        bin_idx = vals_i[valid] - vmin  # 0..nbins-1

        # histogram accumulation
        np.add.at(hist_flat, (pix_idx, bin_idx), 1)

    if hist_flat is None:
        raise RuntimeError("No complete frames found in all files (histogram not built).")

    # fill 0-bin (no-hit counts) if vmin==0
    if vmin == 0:
        nonzero_counts = hist_flat[:, 1:].sum(axis=1)
        hist_flat[:, 0] = n_frames - nonzero_counts

    hist = hist_flat.reshape(ny, nx, nbins)
    return hist, ny, nx, n_frames


def gain_from_hist_argmax(hist, start_bin=100, denom=14.4):
    """
    Gain definition (your formula):
        gain = (argmax(hist[:,:,start_bin:], axis=2) + start_bin) / denom
    """
    if hist.ndim != 3:
        raise ValueError(f"hist must be (ny,nx,nbins), got {hist.shape}")

    ny, nx, nbins = hist.shape
    if start_bin < 0 or start_bin >= nbins:
        raise ValueError(f"start_bin={start_bin} out of range (0..{nbins-1})")

    peak_rel = np.argmax(hist[:, :, start_bin:], axis=2)          # 0..(nbins-start_bin-1)
    peak_abs = peak_rel.astype(np.float32) + float(start_bin)    # start_bin..nbins-1
    gain = peak_abs / float(denom)
    return gain.astype(np.float32)


def pick_2nd_to_5th_newest_L1dat(base_dir="/data/L1", pattern="L1*.dat", n=4):
    """
    Select 4 files by modification time (mtime):
      - sort all matching files by mtime descending (newest first)
      - pick the 2nd to 5th newest (indices 1..4), i.e. total n=4 files
    """
    paths = glob.glob(os.path.join(base_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]

    if len(paths) < 5:
        raise RuntimeError(
            f"Need at least 5 files to pick 2nd~5th newest, but got {len(paths)}. "
            f"pattern={pattern}"
        )

    paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    picked = paths[1:1 + n]  # 2nd~(1+n) newest
    if len(picked) < n:
        raise RuntimeError(f"Failed to pick {n} files, got {len(picked)}.")
    return picked


def main():
    base_dir = "/data/L1"
    out_dir = "/data/gain"
    out_filter_dir = '/data/gain/filter'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_filter_dir, exist_ok=True)

    # 1) pick files (2nd~5th newest by mtime) among L1*.dat
    paths = pick_2nd_to_5th_newest_L1dat(base_dir=base_dir, pattern="L1*.dat", n=4)

    print("Picked files (2nd~5th newest by mtime):")
    for p in paths:
        print(f"  - {p}  mtime={os.path.getmtime(p)}")

    # 2) build per-pixel histogram
    vmin, vmax = 0, 2200
    hist, ny, nx, n_frames = build_pixel_hist(paths, vmin=vmin, vmax=vmax)

    # 3) compute gain using your formula
    gain = gain_from_hist_argmax(hist, start_bin=100, denom=14.4)
    goodfilter = (gain>8) * (gain<20)

    # 4) save outputs
    ts = int(time.time())
    out_npy = os.path.join(out_dir, "gain.npy")
    out2_npy = os.path.join(out_dir, "/gain/gain_{}.npy".format(ts)) 
    np.save(out_npy, gain.astype(np.float32))
    np.save(out2_npy, gain.astype(np.float32))    
    
    
    fout_npy = os.path.join(out_filter_dir, "filter.npy")
    fout2_npy = os.path.join(out_filter_dir, "/filter/filter_{}.npy".format(ts)) 
    np.save(fout_npy, goodfilter.astype(bool))
    np.save(fout2_npy, goodfilter.astype(bool))

if __name__ == "__main__":
    main()