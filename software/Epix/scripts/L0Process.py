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

    Concurrency model:
      - _acceptFrame() stays lightweight: validate + (optional) gate + read + submit
      - heavy numpy work runs in a small ThreadPoolExecutor
      - a sender thread preserves output order by sending futures in submission order

    IMPORTANT shutdown fix:
      - stop() sets a stop flag first (gate) so _acceptFrame drops new frames
      - submit() is wrapped to handle "cannot schedule new futures after shutdown"
      - in-flight tokens (semaphore) are always released, even on exceptions
    """

    HEAD_LEN  = 32
    NY        = 176
    NX        = 768
    U16_COUNT = NY * NX                  # 135,168
    DATA_LEN  = U16_COUNT * 2            # 270,336 bytes

    def __init__(self,
                 dark_path="/data/epix/software/Mossbauer/dark_2D.npy",
                 filter_path="/data/epix/software/Mossbauer/filter.npy",
                 n1=8,
                 enable_common_mode=True,
                 clamp_min=0, clamp_max=0xFFFF,
                 n_workers=None,
                 max_inflight=2048,
                 drop_if_busy=False):

        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        # ---- Load dark (read-only) ----
        dark = np.load(dark_path, mmap_mode='r')
        dark = np.asarray(dark, order='C')
        if dark.size != self.U16_COUNT:
            raise ValueError(f"dark size {dark.size} != {self.U16_COUNT}")
        self.dark_i32 = dark.reshape(self.NY, self.NX).astype(np.int32, copy=False)

        # ---- Load bad-pixel filter (read-only) ----
        # Convention: filter.npy == 0 means "bad pixel" -> mask True
        self.bad_mask = None
        if filter_path is not None:
            filt = np.load(filter_path, mmap_mode='r')
            filt = np.asarray(filt, order='C')
            if filt.size != self.U16_COUNT:
                raise ValueError(f"filter size {filt.size} != {self.U16_COUNT}")
            self.bad_mask = (filt.reshape(self.NY, self.NX) == 0)

        # ---- Parameters ----
        self.n1 = int(n1)
        self.enable_common_mode = bool(enable_common_mode)
        self.clamp_min = int(clamp_min)
        self.clamp_max = int(clamp_max)

        # ---- Thread pool sizing (avoid oversubscription on big-core machines) ----
        if n_workers is None:
            cpu = os.cpu_count() or 8
            n_workers = max(4, min(16, cpu // 8))  # 128C->16, 32C->4
        self.n_workers = int(n_workers)

        # ---- In-flight control (memory bound) ----
        self.max_inflight = int(max_inflight)
        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(self.max_inflight)

        # ---- Per-thread workspace (avoid per-frame allocations) ----
        self._tls = threading.local()

        # ---- Ordering and sender thread ----
        self._futs = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._exec = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix="L0W")
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

    # ---- Per-thread workspace ----
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

    def _col_common_mode(self, w2d: np.ndarray, mask_2d: np.ndarray, col_med: np.ndarray, thr: int) -> None:
        """
        Column common-mode correction using "quiet" pixels (w2d < thr).
        """
        if not self.enable_common_mode:
            return

        # mask_2d True => selected quiet pixels
        np.less(w2d, thr, out=mask_2d)

        cnt_cols = mask_2d.sum(axis=0).astype(np.int64, copy=False)
        np.maximum(cnt_cols, 1, out=cnt_cols)

        # Use sum with "where" if available (avoids allocating temporaries)
        try:
            sum_cols = np.sum(w2d, axis=0, dtype=np.int64, where=mask_2d)
        except TypeError:
            sum_cols = np.sum(np.where(mask_2d, w2d, 0), axis=0, dtype=np.int64)

        col_med[:] = (sum_cols // cnt_cols).astype(np.int32, copy=False)
        w2d -= col_med

    def _process_buf(self, buf: bytearray) -> bytearray:
        """
        Heavy processing (runs in worker thread). Modifies buf in-place and returns it.
        """
        # Writable view into uint16 image region
        arr_u2 = np.frombuffer(
            buf, dtype=np.dtype('<u2'),
            count=self.U16_COUNT, offset=self.HEAD_LEN
        ).reshape(self.NY, self.NX)

        ws = self._ws()
        work_i32 = ws.work_i32

        # raw - dark -> i32
        np.subtract(arr_u2, self.dark_i32, out=work_i32, casting='unsafe')

        # Dynamic threshold: 4*n1 + mean(raw-dark)
        mean_i32 = int(work_i32.sum(dtype=np.int64) // self.U16_COUNT)
        thr = 4 * self.n1 + mean_i32
        if thr < 0:
            thr = 0
        elif thr > 0xFFFF:
            thr = 0xFFFF

        # Column common mode
        self._col_common_mode(work_i32, ws.mask_2d, ws.col_med, thr)

        # Bad pixels
        if self.bad_mask is not None:
            work_i32[self.bad_mask] = 0

        # Clamp and write back
        np.clip(work_i32, self.clamp_min, self.clamp_max, out=work_i32)
        arr_u2[:, :] = work_i32.astype(np.uint16, copy=False)

        # Timestamp: seconds, written at offset 28 (preserve behavior)
        ts_s = int(time.time())
        struct.pack_into('<I', buf, 28, ts_s & 0xFFFFFFFF)

        return buf

    def _acceptFrame(self, frame):
        """
        Rogue receive callback.
        Must be safe during shutdown: never submit after executor shutdown.
        """
        # Gate immediately if stopping
        if self._stop.is_set():
            return

        size = frame.getPayload()

        # Only process fully valid frames
        data_bytes = max(0, size - self.HEAD_LEN)
        valid_bytes = min(self.DATA_LEN, data_bytes)
        if valid_bytes < self.DATA_LEN:
            return

        # In-flight gate
        if self.drop_if_busy:
            if not self._sem.acquire(blocking=False):
                return
        else:
            self._sem.acquire()

        # stop could be set while waiting for semaphore
        if self._stop.is_set():
            self._sem.release()
            return

        # Read payload into mutable buffer
        buf = bytearray(size)
        frame.read(buf, 0)

        # Submit heavy work; handle shutdown race gracefully
        try:
            fut = self._exec.submit(self._process_buf, buf)
        except RuntimeError:
            # "cannot schedule new futures after shutdown"
            self._sem.release()
            return

        # Always release the in-flight token when the future completes (success or failure)
        fut.add_done_callback(lambda _f: self._sem.release())

        # Preserve output order
        with self._lock:
            # If stop is set right now, do not enqueue new futures
            if self._stop.is_set():
                return
            self._futs.append(fut)

    def _sender_loop(self):
        """
        Sender thread: sends frames in submission order.
        """
        while not self._stop.is_set():
            fut = None
            with self._lock:
                if self._futs:
                    fut = self._futs[0]

            if fut is None:
                time.sleep(0.0005)
                continue

            if not fut.done():
                time.sleep(0.0002)
                continue

            with self._lock:
                fut = self._futs.popleft()

            try:
                out_buf = fut.result()
            except Exception:
                # Worker failed; drop output for this frame
                continue

            out = self._reqFrame(len(out_buf), True)
            out.write(out_buf, 0)
            self._sendFrame(out)

    def stop(self):
        """
        Stop processing:
          1) gate _acceptFrame() immediately
          2) stop sender thread
          3) shutdown executor
        """
        self._stop.set()

        try:
            self._sender.join(timeout=1.0)
        except Exception:
            pass

        try:
            # cancel_futures requires Python 3.9+
            self._exec.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # older Python: no cancel_futures
            try:
                self._exec.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
