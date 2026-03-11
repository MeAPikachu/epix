#!/usr/bin/env python3
import os, time, glob
import numpy as np
import bottleneck as bn
from epix import epix

# Files Directory, we got the files from the raw data;
# Important that the newest version does not include filter part; 
# The filter comes from the precalibration and dynamic gain calibration
RAW_DIR  = "/data/raw"
RAW_GLOB = "*.dat.*"     
OUT_DIR  = "/data/dark"
OUT_DARK_DIR = '/data/dark/dark'


FRAMES   = 10000              # Maximum Frames that we are using; 
START    = 0
STD_THR  = 50.0 # The standard error threshold is still 50

SCAN_S   = 60                 # How Often we check this 
PICK_SECOND_NEWEST_RAW = True # Use the second newest raw data; 

# Find the raw data we want; 
def pick_raw():
    files = glob.glob(os.path.join(RAW_DIR, RAW_GLOB))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if PICK_SECOND_NEWEST_RAW and len(files) >= 2:
        return files[1]
    return files[0]

# the dot is a bad symbol, so replace it. 
def raw_tag(raw_path: str) -> str:
    # data_20251126_133252.dat.247 -> data_20251126_133252_dat_247
    return os.path.basename(raw_path).replace(".", "_")

# The calculation; 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DARK_DIR, exist_ok=True)

    while True:
        try:
            # Find the raw data that we are going to use; 
            raw = pick_raw()
            if raw is None:
                time.sleep(SCAN_S)
                continue
            tag = raw_tag(raw)
            
            # We shall update the tagged dark 
            dark_ver   = os.path.join(OUT_DARK_DIR, f"dark_2D_{tag}.npy")
            dark_cur   = os.path.join(OUT_DIR, "dark_2D.npy")

            # If it exists, skip it. 
            if os.path.exists(dark_ver):
                time.sleep(SCAN_S)
                continue

            det = epix(raw)
            data = det.data[START : START + FRAMES]  
            nframes = data.shape[0]
            if nframes == 0:
                raise RuntimeError("no frames available")

            # The background mainly focus on the median value of the background; 
            median_2d = bn.nanmedian(data, axis=0)

            # Save the newest version and archive the data; 
            np.save(dark_ver, median_2d)
            np.save(dark_cur, median_2d)

            print(f"[dark-builder] raw    : {raw}")
            print(f"[dark-builder] frames : {nframes}")
            print(f"[dark-builder] wrote  : {dark_ver}")
            print(f"[dark-builder] update : dark_2D.npy / filter.npy")

        except Exception as e:
            print(f"[dark-builder] ERROR: {e}")

        time.sleep(SCAN_S)

if __name__ == "__main__":
    main()
