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

    Optional centroid + charge sharing (only when enable_centroid=True):
      - Find local-maximum centroid pixels (interior only)
      - Center pixel must satisfy center >= thr_cs (else not a centroid and no output)
      - Charge sum = sum of {center, up, down, left, right} but each contributor
        is included ONLY if that pixel >= thr_cs (implemented by in-place thresholding)
      - Non-centroid pixels output 0
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
                 n_workers=None,
                 max_inflight=2048,
                 drop_if_busy=False,

                 # === centroid/charge sharing (ADD-ON; default OFF) ===
                 enable_centroid=False,
                 n2=15,                      # thr_cs = 4*n2
                 centroid_use_ge=False       # >= neighbors vs > neighbors
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

        # ---- Parameters (original) ----
        self.n1 = int(n1)
        self.enable_common_mode = bool(enable_common_mode)
        self.clamp_min = int(clamp_min)
        self.clamp_max = int(clamp_max)

        # ---- Centroid params (only used when enabled) ----
        self.enable_centroid = bool(enable_centroid)
        self.n2 = int(n2)
        self.centroid_use_ge = bool(centroid_use_ge)

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

        filt = None
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

        # === EXACTLY like original, always ===
        ws.work_i32 = np.empty((self.NY, self.NX), dtype=np.int32)
        ws.mask_2d  = np.empty((self.NY, self.NX), dtype=bool)
        ws.col_med  = np.empty(self.NX, dtype=np.int32)

        # === centroid buffers ONLY when enabled ===
        if self.enable_centroid:
            ws.out_i32     = np.empty((self.NY, self.NX), dtype=np.int32)
            ws.cmask_core  = np.empty((self.NY - 2, self.NX - 2), dtype=bool)

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
    # centroid + charge-sharing threshold (only called when enabled)
    # ------------------------------------------------------------------
    def _centroid_4n_sum_thr(self, work_i32, out_i32, cmask_core, thr_cs):
        """
        In-place thresholding:
        - work_i32[work_i32 < thr_cs] = 0   (so below-threshold never contributes)
        - centroid requires center >= thr_cs (equivalently center != 0 after thresholding)
        - sum = c+l+r+u+d on thresholded image
        - output only at centroid pixels
        """
        out_i32.fill(0)

        # threshold in-place
        work_i32[work_i32 < thr_cs] = 0

        c = work_i32[1:-1, 1:-1]
        l = work_i32[1:-1, 0:-2]
        r = work_i32[1:-1, 2:  ]
        u = work_i32[0:-2, 1:-1]
        d = work_i32[2:  , 1:-1]

        cmask_core[:] = (c != 0)
        if self.centroid_use_ge:
            cmask_core &= (c >= l) & (c >= r) & (c >= u) & (c >= d)
        else:
            cmask_core &= (c >  l) & (c >  r) & (c >  u) & (c >  d)

        acc = out_i32[1:-1, 1:-1]
        acc[:] = c
        acc += l
        acc += r
        acc += u
        acc += d

        acc[~cmask_core] = 0

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

        # snapshot calib (no lock in hot path)
        with self._calib_lock:
            dark = self.dark_i32
            bad_mask = self.bad_mask

        np.subtract(arr_u2, dark, out=work_i32, casting='unsafe')

        mean_i32 = int(work_i32.sum(dtype=np.int64) // self.U16_COUNT)
        thr = max(0, min(0xFFFF, 4 * self.n1 + mean_i32))

        self._col_common_mode(work_i32, ws.mask_2d, ws.col_med, thr)

        if bad_mask is not None:
            work_i32[bad_mask] = 0

        # ===== ONLY ADDITION: centroid branch =====
        if self.enable_centroid:
            thr_cs = max(0, min(0x7FFFFFFF, 4 * self.n2))
            self._centroid_4n_sum_thr(work_i32, ws.out_i32, ws.cmask_core, thr_cs)

            np.clip(ws.out_i32, self.clamp_min, self.clamp_max, out=ws.out_i32)
            arr_u2[:, :] = ws.out_i32.astype(np.uint16, copy=False)
        else:
            # ===== EXACT original block (unchanged) =====
            np.clip(work_i32, self.clamp_min, self.clamp_max, out=work_i32)
            arr_u2[:, :] = work_i32.astype(np.uint16, copy=False)
        # ==========================================

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