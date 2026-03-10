#!/usr/bin/env python3
import os, time, glob
import numpy as np
import bottleneck as bn
from epix import epix

# ===== 固定配置 =====
RAW_DIR  = "/data/raw"
RAW_GLOB = "*.dat.*"     # data_YYYYmmdd_HHMMSS.dat.NNN
OUT_DIR  = "/data/dark"

FRAMES   = 10000              # 最大帧数（不足则自动用全部）
START    = 0
STD_THR  = 50.0

SCAN_S   = 60                 # 扫描频率（秒，可高）
PICK_SECOND_NEWEST_RAW = True # 用第二新防半文件
# ====================

def pick_raw():
    files = glob.glob(os.path.join(RAW_DIR, RAW_GLOB))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if PICK_SECOND_NEWEST_RAW and len(files) >= 2:
        return files[1]
    return files[0]

def raw_tag(raw_path: str) -> str:
    # data_20251126_133252.dat.247 -> data_20251126_133252_dat_247
    return os.path.basename(raw_path).replace(".", "_")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    while True:
        try:
            raw = pick_raw()
            if raw is None:
                time.sleep(SCAN_S)
                continue

            tag = raw_tag(raw)
            print(raw)
            dark_ver   = os.path.join(OUT_DIR, f"dark_2D_{tag}.npy")
            filter_ver = os.path.join(OUT_DIR, f"filter_{tag}.npy")

            # ★ 关键：已经算过就直接跳过
            if os.path.exists(dark_ver):
                time.sleep(SCAN_S)
                continue

            det = epix(raw)
            data = det.data[START : START + FRAMES]  # 不足FRAMES自动截断
            nframes = data.shape[0]
            if nframes == 0:
                raise RuntimeError("no frames available")

            dark_cur   = os.path.join(OUT_DIR, "dark_2D.npy")
            filter_cur = os.path.join(OUT_DIR, "filter.npy")

            median_2d = bn.nanmedian(data, axis=0)
            std_map   = bn.nanstd(data, axis=0)
            filt      = (std_map < STD_THR)

            # 保存版本化
            np.save(dark_ver, median_2d)
            np.save(filter_ver, filt)

            # 更新当前最新
            np.save(dark_cur, median_2d)
            np.save(filter_cur, filt)

            print(f"[dark-builder] raw    : {raw}")
            print(f"[dark-builder] frames : {nframes}")
            print(f"[dark-builder] wrote  : {dark_ver}")
            print(f"[dark-builder] update : dark_2D.npy / filter.npy")

        except Exception as e:
            print(f"[dark-builder] ERROR: {e}")

        time.sleep(SCAN_S)

if __name__ == "__main__":
    main()
