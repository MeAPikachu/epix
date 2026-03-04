# -*- coding: utf-8 -*-
import os
import time
import struct
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rogue.interfaces.stream


class L0Process(rogue.interfaces.stream.Slave, rogue.interfaces.stream.Master):
    """
    Level-0 processing:
      - dark subtraction
      - dynamic column common-mode correction
      - bad-pixel filtering
      - clamp and write back to uint16

    Optional post-step (enable_centroid=True):
      - centroid + 4-neighbor charge sum (center pixel becomes sum of itself + up/down/left/right)
      - non-centroid pixels are set to 0
    """

    HEAD_LEN  = 32
    NY        = 176
    NX        = 768
    U16_COUNT = NY * NX
    DATA_LEN  = U16_COUNT * 2

    def __init__(self,
                 dark_path="/data/epix/software/Mossbauer/dark_2D.npy",
                 filter_path="/data/epix/software/Mossbauer/filter.npy",

                 # === dynamic calib ===
                 dynamic_calib=False,
                 dynamic_calib_dir="/data/dark",
                 dynamic_filter_dir="/data/epix/software/Mossbauer",
                 dynamic_calib_period_s=3600,
                 # =====================

                 n1=8,
                 enable_common_mode=True,
                 clamp_min=0, clamp_max=0xFFFF,

                 # === threading/flow ===
                 n_workers=None,
                 max_inflight=2048,
                 drop_if_busy=False,

                 # === optional centroid ===
                 enable_centroid=False,
                 centroid_n2=15.0,
                 exclude_border=False  # kept for API compatibility; border is effectively excluded in this minimal impl
                 ):

        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        # === dynamic calib ===
        self.dynamic_calib = bool(dynamic_calib)
        self.dynamic_calib_period_s = int(dynamic_calib_period_s)

        if self.dynamic_calib:
            self.dark_path   = os.path.join(dynamic_calib_dir, "dark_2D.npy")
            self.filter_path = os.path.join(dynamic_filter_dir, "filter.npy")
        else:
            self.dark_path   = dark_path
            self.filter_path = filter_path

        self._calib_lock = threading.Lock()
        self._dark_mtime = None
        self._filter_mtime = None
        # =====================

        # ---- Load calib initially ----
        self._load_calib(initial=True)

        # ---- Parameters ----
        self.n1 = int(n1)
        self.enable_common_mode = bool(enable_common_mode)
        self.clamp_min = int(clamp_min)
        self.clamp_max = int(clamp_max)

        # ---- centroid parameters (NEW, default OFF) ----
        self.enable_centroid = bool(enable_centroid)
        self.centroid_n2 = float(centroid_n2)
        self.exclude_border = bool(exclude_border)

        # ---- Thread pool sizing ----
        if n_workers is None:
            cpu = os.cpu_count() or 8
            n_workers = max(4, min(16, cpu // 8))
        self.n_workers = int(n_workers)

        # ---- In-flight control ----
        self.max_inflight = int(max_inflight)
        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(self.max_inflight)

        # ---- Per-thread workspace ----
        self._tls = threading.local()

        # ---- Ordering and sender thread ----
        self._futs = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._exec = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix="L0W")
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

        # === dynamic calib watcher ===
        if self.dynamic_calib:
            self._calib_thread = threading.Thread(
                target=self._calib_watcher,
                name="L0CalibWatcher",
                daemon=True,
            )
            self._calib_thread.start()
        # ==============================

    # ------------------------------------------------------------------
    # dynamic calib
    # ------------------------------------------------------------------
    def _load_calib(self, initial=False):
        try:
            dark_mtime = os.path.getmtime(self.dark_path)
            filter_mtime = os.path.getmtime(self.filter_path)
        except OSError:
            if initial:
                raise RuntimeError(
                    f"Initial calib load failed: {self.dark_path} / {self.filter_path}"
                )
            return

        if (not initial and
            dark_mtime == self._dark_mtime and
            filter_mtime == self._filter_mtime):
            return

        dark = np.load(self.dark_path, mmap_mode="r")
        dark = np.asarray(dark, order="C")
        if dark.size != self.U16_COUNT:
            raise ValueError(f"dark size {dark.size} != {self.U16_COUNT}")
        dark_i32 = dark.reshape(self.NY, self.NX).astype(np.int32, copy=False)

        bad_mask = None
        if self.filter_path is not None:
            filt = np.load(self.filter_path, mmap_mode="r")
            filt = np.asarray(filt, order="C")
            if filt.size != self.U16_COUNT:
                raise ValueError(f"filter size {filt.size} != {self.U16_COUNT}")
            bad_mask = (filt.reshape(self.NY, self.NX) == 0)

        with self._calib_lock:
            self.dark_i32 = dark_i32
            self.bad_mask = bad_mask
            self._dark_mtime = dark_mtime
            self._filter_mtime = filter_mtime

        print(f"[L0Process] calib loaded: {os.path.basename(self.dark_path)}")

    def _calib_watcher(self):
        while not self._stop.is_set():
            time.sleep(self.dynamic_calib_period_s)
            try:
                self._load_calib(initial=False)
            except Exception as e:
                print(f"[L0Process] calib reload failed: {e}")

    # ------------------------------------------------------------------
    # Per-thread workspace
    # ------------------------------------------------------------------
    def _ws(self):
        ws = getattr(self._tls, "ws", None)
        if ws is not None:
            return ws

        class _WS:
            pass

        ws = _WS()
        ws.work_i32 = np.empty((self.NY, self.NX), dtype=np.int32)
        ws.mask_2d  = np.empty((self.NY, self.NX), dtype=bool)
        ws.col_med  = np.empty(self.NX, dtype=np.int32)
        self._tls.ws = ws
        return ws

    def _col_common_mode(self, w2d, mask_2d, col_med, thr):
        if not self.enable_common_mode:
            return

        np.less(w2d, thr, out=mask_2d)
        cnt_cols = mask_2d.sum(axis=0).astype(np.int64, copy=False)
        np.maximum(cnt_cols, 1, out=cnt_cols)

        try:
            sum_cols = np.sum(w2d, axis=0, dtype=np.int64, where=mask_2d)
        except TypeError:
            sum_cols = np.sum(np.where(mask_2d, w2d, 0), axis=0, dtype=np.int64)

        col_med[:] = (sum_cols // cnt_cols).astype(np.int32, copy=False)
        w2d -= col_med

    # ------------------------------------------------------------------
    # centroid + charge sharing (NEW, minimal / safe)
    # ------------------------------------------------------------------
    def _centroid_charge_sum_inplace(self, arr_u2):
        """
        Centroid + 4-neighbor charge sum (safe implementation).
        - Find strict 4-neighbor local maxima above thr_pix (= 4*centroid_n2)
        - Output: all zeros except centroid pixels, where value = self+up+down+left+right (clamped)

        Note: for robustness and minimal changes, centroid search is performed only on inner pixels (1:-1, 1:-1),
              so border pixels are never selected as centroid (effectively exclude_border=True).
        """
        # pixel threshold
        thr_pix = int(4.0 * float(self.centroid_n2))
        if thr_pix <= 0:
            thr_pix = 1
        if thr_pix > 0xFFFF:
            thr_pix = 0xFFFF
        thr_pix_u16 = np.uint16(thr_pix)

        ny, nx = arr_u2.shape
        if ny < 3 or nx < 3:
            arr_u2[:, :] = 0
            return

        # Views (no big index arrays)
        c  = arr_u2[1:-1, 1:-1]
        up = arr_u2[2:  , 1:-1]
        dn = arr_u2[0:-2, 1:-1]
        lf = arr_u2[1:-1, 2:  ]
        rt = arr_u2[1:-1, 0:-2]

        cen_mask = (c > thr_pix_u16) & (c > up) & (c > dn) & (c > lf) & (c > rt)
        if not np.any(cen_mask):
            arr_u2[:, :] = 0
            return

        # Sum cluster (uint32 to avoid overflow), then clamp
        cluster = (c.astype(np.uint32) + up.astype(np.uint32) + dn.astype(np.uint32) +
                   lf.astype(np.uint32) + rt.astype(np.uint32))

        cluster = np.clip(cluster, self.clamp_min, self.clamp_max).astype(np.uint16, copy=False)

        # Write output
        arr_u2[:, :] = 0
        out_inner = arr_u2[1:-1, 1:-1]
        out_inner[cen_mask] = cluster[cen_mask]

    # ------------------------------------------------------------------
    # heavy worker
    # ------------------------------------------------------------------
    def _process_buf(self, buf):
        arr_u2 = np.frombuffer(
            buf, dtype=np.dtype('<u2'),
            count=self.U16_COUNT, offset=self.HEAD_LEN
        ).reshape(self.NY, self.NX)

        ws = self._ws()
        work_i32 = ws.work_i32

        # snapshot calib
        with self._calib_lock:
            dark = self.dark_i32
            bad_mask = self.bad_mask

        np.subtract(arr_u2, dark, out=work_i32, casting='unsafe')

        mean_i32 = int(work_i32.sum(dtype=np.int64) // self.U16_COUNT)
        thr = max(0, min(0xFFFF, 4 * self.n1 + mean_i32))

        self._col_common_mode(work_i32, ws.mask_2d, ws.col_med, thr)

        if bad_mask is not None:
            work_i32[bad_mask] = 0

        np.clip(work_i32, self.clamp_min, self.clamp_max, out=work_i32)
        arr_u2[:, :] = work_i32.astype(np.uint16, copy=False)

        # ---- optional centroid (NEW, default OFF) ----
        if self.enable_centroid:
            try:
                self._centroid_charge_sum_inplace(arr_u2)
            except Exception:
                # Never kill the stream on centroid issues
                pass

        ts_s = int(time.time())
        struct.pack_into('<I', buf, 28, ts_s & 0xFFFFFFFF)

        return buf

    # ------------------------------------------------------------------
    # Rogue callbacks
    # ------------------------------------------------------------------
    def _acceptFrame(self, frame):
        if self._stop.is_set():
            return

        size = frame.getPayload()
        data_bytes = max(0, size - self.HEAD_LEN)
        if data_bytes < self.DATA_LEN:
            return

        if self.drop_if_busy:
            if not self._sem.acquire(blocking=False):
                return
        else:
            self._sem.acquire()

        if self._stop.is_set():
            self._sem.release()
            return

        buf = bytearray(size)
        frame.read(buf, 0)

        try:
            fut = self._exec.submit(self._process_buf, buf)
        except RuntimeError:
            self._sem.release()
            return

        fut.add_done_callback(lambda _f: self._sem.release())

        with self._lock:
            if self._stop.is_set():
                return
            self._futs.append(fut)

    def _sender_loop(self):
        while not self._stop.is_set():
            with self._lock:
                fut = self._futs[0] if self._futs else None

            if fut is None or not fut.done():
                time.sleep(0.0005)
                continue

            with self._lock:
                fut = self._futs.popleft()

            try:
                out_buf = fut.result()
            except Exception:
                continue

            out = self._reqFrame(len(out_buf), True)
            out.write(out_buf, 0)
            self._sendFrame(out)

    def stop(self):
        self._stop.set()

        try:
            self._sender.join(timeout=1.0)
        except Exception:
            pass

        try:
            self._exec.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            try:
                self._exec.shutdown(wait=False)
            except Exception:
                pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass