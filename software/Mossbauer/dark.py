#!/usr/bin/env python3
import os, argparse
import numpy as np
import bottleneck as bn
from epix import epix

def main():
    ap = argparse.ArgumentParser(description="Compute median & std masks for ePix data.")
    ap.add_argument("-i", "--input" , default="/data/thermal20_0924.dat",help="data file")
    ap.add_argument("-n", "--frames", type=int, default=10000, help="10000")
    ap.add_argument("-s", "--start",  type=int, default=0,      help="0")
    ap.add_argument("-o", "--outdir", default=".",              help="")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.outdir, exist_ok=True)

    det  = epix(args.input)
    data = det.data[args.start: args.start + args.frames]

    median_2d = bn.nanmedian(data, axis=0)
    std_map   = bn.nanstd   (data, axis=0)

    np.save(os.path.join(args.outdir, f"{base}_median.npy"), median_2d)
    for thr in (50, 100):
        mask = (std_map < thr).astype(bool)
        np.savez(os.path.join(args.outdir, f"{base}_std_{thr}.npz"), mask=mask, std=std_map)

if __name__ == "__main__":
    main()
