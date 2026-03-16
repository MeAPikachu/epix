#!/usr/bin/env python3
import os, time, glob
import numpy as np
import bottleneck as bn
from epix import epix

RAW_DIR  = "/data/raw"
RAW_GLOB = "*.dat.*"
OUT_DIR  = "/data/dark"
OUT_DARK_DIR = '/data/dark/dark'

FRAMES   = 10000
START    = 0
STD_THR  = 50.0

PICK_SECOND_NEWEST_RAW = True

def pick_raw():
    files = glob.glob(os.path.join(RAW_DIR, RAW_GLOB))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if PICK_SECOND_NEWEST_RAW and len(files) >= 2:
        return files[1]
    return files[0]

def raw_tag(raw_path: str) -> str:
    return os.path.basename(raw_path).replace(".", "_")

def sleep_until_next_hour():
    now = time.time()
    next_hour = ((int(now) // 3600) + 1) * 3600
    sleep_s = max(1, next_hour - now)
    print(f"[dark-builder] sleep until next hour: {sleep_s:.1f} s")
    time.sleep(sleep_s)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DARK_DIR, exist_ok=True)

    while True:
        try:
            raw = pick_raw()
            if raw is None:
                print("[dark-builder] no raw file found")
                sleep_until_next_hour()
                continue

            tag = raw_tag(raw)
            time_tag = int(time.time())

            dark_ver = os.path.join(OUT_DARK_DIR, f"dark_2D_{time_tag}.npy")
            dark_cur = os.path.join(OUT_DIR, "dark_2D.npy")

            det = epix(raw)
            data = det.data[START : START + FRAMES]
            nframes = data.shape[0]
            if nframes == 0:
                raise RuntimeError("no frames available")

            median_2d = bn.nanmedian(data, axis=0)

            np.save(dark_ver, median_2d)
            np.save(dark_cur, median_2d)

            print(f"[dark-builder] raw    : {raw}")
            print(f"[dark-builder] frames : {nframes}")
            print(f"[dark-builder] wrote  : {dark_ver}")
            print(f"[dark-builder] update : dark_2D.npy / filter.npy")

        except Exception as e:
            print(f"[dark-builder] ERROR: {e}")

        sleep_until_next_hour()

if __name__ == "__main__":
    main()