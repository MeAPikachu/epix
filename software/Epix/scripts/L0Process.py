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

    Optional post-step:
      - centroid + 4-neighbor charge-sum (center pixel becomes sum of itself + up/down/left/right)
      - non-centroid pixels are set to 0
    """

    HEAD_LEN = 32

    def __init__(self,
                 # =========================
                 # Frame geometry (NEW)
                 # =========================
                 ny=176,
                 nx=768,

                 # =========================
                 # Static calib paths
                 # =========================
                 dark_path="/data/epix/software/Mossbauer/dark_2D.npy",
                 filter_path="/data/epix/software/Mossbauer/filter.npy",

                 # =========================
                 # Dynamic calib
                 # =========================
                 dynamic_calib=False,
                 dynamic_calib_dir="/data/dark",
                 dynamic_filter_dir="/data/epix/software/Mossbauer",
                 dynamic_calib_period_s=3600,

                 # =========================
                 # L0 params
                 # =========================
                 n1=8,
                 enable_common_mode=True,
                 clamp_min=0,
                 clamp_max=0xFFFF,

                 # =========================
                 # threading / flow control
                 # =========================
                 n_workers=None,
                 max_inflight=2048,
                 drop_if_busy=False,

                 # =========================
                 # centroid + charge sharing (NEW)
                 # =========================
                 enable_centroid=False,
                 centroid_n2=15.0,   # pixel threshold: thr_pix = 4*n2
                 exclude_border=False  # if True, never keep border pixels as centroids
                 ):

        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        # ---- geometry ----
        self.NY = int(ny)
        self.NX = int(nx)
        self.U16_COUNT = self.NY * self.NX
        self.DATA_LEN = self.U16_COUNT * 2

        # ---- dynamic calib ----
        self.dynamic_calib = bool(dynamic_calib)
        self.dynamic_calib_period_s = int(dynamic_calib_period_s)

        if self.dynamic_calib:
            self.dark_path = os.path.join(dynamic_calib_dir, "dark_2D.npy")
            self.filter_path = os.path.join(dynamic_filter_dir, "filter.npy")
        else:
            self.dark_path = dark_path
            self.filter_path = filter_path

        self._calib_lock = threading.Lock()
        self._dark_mtime = None
        self._filter_mtime = None

        # ---- Load calib initially ----
        self._load_calib(initial=True)

        # ---- L0 parameters ----
        self.n1 = int(n1)
        self.enable_common_mode = bool(enable_common_mode)
        self.clamp_min = int(clamp_min)
        self.clamp_max = int(clamp_max)

        # ---- centroid parameters ----
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

        # ---- dynamic calib watcher ----
        if self.dynamic_calib:
            self._calib_thread = threading.Thread(
                target=self._calib_watcher,
                name="L0CalibWatcher",
                daemon=True,
            )
            self._calib_thread.start()

    # ------------------------------------------------------------------
    # dynamic calib
    # ------------------------------------------------------------------
    def _load_calib(self, initial=False):
        try:
            dark_mtime = os.path.getmtime(self.dark_path)
            filter_mtime = os.path.getmtime(self.filter_path) if self.filter_path is not None else None
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
            raise ValueError(f"dark size {dark.size} != {self.U16_COUNT} (NY*NX={self.U16_COUNT})")
        dark_i32 = dark.reshape(self.NY, self.NX).astype(np.int32, copy=False)

        bad_mask = None
        if self.filter_path is not None:
            filt = np.load(self.filter_path, mmap_mode="r")
            filt = np.asarray(filt, order="C")
            if filt.size != self.U16_COUNT:
                raise ValueError(f"filter size {filt.size} != {self.U16_COUNT} (NY*NX={self.U16_COUNT})")
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

        # centroid neighbor buffers (avoid per-frame allocations)
        if self.enable_centroid:
            ws.up    = np.empty((self.NY, self.NX), dtype=np.uint16)
            ws.down  = np.empty((self.NY, self.NX), dtype=np.uint16)
            ws.left  = np.empty((self.NY, self.NX), dtype=np.uint16)
            ws.right = np.empty((self.NY, self.NX), dtype=np.uint16)
            ws.cen_mask = np.empty((self.NY, self.NX), dtype=bool)

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
    # centroid + charge sharing (center = self + 4-neighbors)
    # ------------------------------------------------------------------
    def _centroid_charge_sum_inplace(self, arr_u2, ws):
        """
        Sparse centroid + 4-neighbor charge sum.
        Only check candidate pixels where arr_u2 > thr_pix (=4*n2).
        Keep strict local maxima (4-neighbor), and set center value = self+up+down+left+right.
        Others set to 0.
        """
        thr_pix = 4.0 * float(self.centroid_n2)
        ny, nx = self.NY, self.NX

        # quick reject: if frame max <= thr_pix, all zero
        if arr_u2.max(initial=0) <= thr_pix:
            arr_u2[:, :] = 0
            return

        # candidates above threshold
        cand = np.nonzero(arr_u2 > thr_pix)
        if cand[0].size == 0:
            arr_u2[:, :] = 0
            return

        y = cand[0].astype(np.int32, copy=False)
        x = cand[1].astype(np.int32, copy=False)

        # exclude border (because neighbors missing)
        inner = (y > 0) & (y < ny - 1) & (x > 0) & (x < nx - 1)
        if not np.any(inner):
            arr_u2[:, :] = 0
            return

        y = y[inner]
        x = x[inner]

        self_v  = arr_u2[y, x].astype(np.uint32, copy=False)
        up_v    = arr_u2[y + 1, x].astype(np.uint32, copy=False)
        down_v  = arr_u2[y - 1, x].astype(np.uint32, copy=False)
        left_v  = arr_u2[y, x + 1].astype(np.uint32, copy=False)
        right_v = arr_u2[y, x - 1].astype(np.uint32, copy=False)

        # strict local maxima
        cen = (self_v > up_v) & (self_v > down_v) & (self_v > left_v) & (self_v > right_v)
        if not np.any(cen):
            arr_u2[:, :] = 0
            return

        yc = y[cen]
        xc = x[cen]
        cluster = self_v[cen] + up_v[cen] + down_v[cen] + left_v[cen] + right_v[cen]

        # clamp and write back
        cluster = np.clip(cluster, self.clamp_min, self.clamp_max).astype(np.uint16, copy=False)

        # output frame: zeros except centroids
        arr_u2[:, :] = 0
        arr_u2[yc, xc] = cluster

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

        # dark subtraction
        np.subtract(arr_u2, dark, out=work_i32, casting='unsafe')

        # common-mode threshold (frame-dependent)
        mean_i32 = int(work_i32.sum(dtype=np.int64) // self.U16_COUNT)
        thr = max(0, min(0xFFFF, 4 * self.n1 + mean_i32))

        # column common-mode correction
        self._col_common_mode(work_i32, ws.mask_2d, ws.col_med, thr)

        # bad pixels
        if bad_mask is not None:
            work_i32[bad_mask] = 0

        # clamp and write back to uint16
        np.clip(work_i32, self.clamp_min, self.clamp_max, out=work_i32)
        arr_u2[:, :] = work_i32.astype(np.uint16, copy=False)

        # centroid + charge sharing
        if self.enable_centroid:
            try:
                self._centroid_charge_sum_inplace(arr_u2, ws)
            except Exception:
                # do not kill the stream on centroid errors
                pass

        # timestamp
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