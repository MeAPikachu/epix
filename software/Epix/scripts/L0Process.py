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

    Key design goal:
      Keep Rogue's receive callback (_acceptFrame) lightweight by offloading heavy numpy
      work to a small thread pool. Numpy releases the GIL for these operations, so
      multiple worker threads can run concurrently on multiple CPU cores.

    Compatibility:
      - Same class name and constructor interface as your current version.
      - Same frame format in/out (header preserved; image data modified in-place in buffer).
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

        # ---- Load dark readout (read-only) ----
        dark = np.load(dark_path, mmap_mode='r')
        dark = np.asarray(dark, order='C')
        if dark.size != self.U16_COUNT:
            raise ValueError(f"dark size {dark.size} != {self.U16_COUNT}")
        self.dark_i32 = dark.reshape(self.NY, self.NX).astype(np.int32, copy=False)

        # ---- Bad pixels filter mask (read-only) ----
        # Your convention: filter.npy == 0 means "bad pixel" (mask True)
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

        # ---- Thread pool sizing ----
        # Do not oversubscribe on large-core machines (often memory-bandwidth bound).
        if n_workers is None:
            cpu = os.cpu_count() or 8
            n_workers = max(4, min(16, cpu // 8))  # 128C -> 16; 32C -> 4
        self.n_workers = int(n_workers)

        # ---- In-flight control ----
        # Use a semaphore instead of "busy-wait + lock release/acquire".
        # This is safer and avoids burning CPU in sleep loops.
        self.max_inflight = int(max_inflight)
        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(self.max_inflight)

        # ---- Per-thread workspace (avoids per-frame allocations) ----
        self._tls = threading.local()

        # ---- Ordering and sender thread ----
        # We keep a deque of futures in submission order, then the sender thread only
        # sends when the head future is done. This preserves output order.
        self._futs = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._exec = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix="L0W")
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

    # ---- Per-thread workspace holder ----
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
        Column-wise common-mode correction:
          - Build a mask selecting "quiet" pixels (w2d < thr)
          - Compute per-column mean of selected pixels
          - Subtract per-column mean from each column
        """
        if not self.enable_common_mode:
            return

        # mask_2d True means "quiet" pixel
        np.less(w2d, thr, out=mask_2d)

        # Count selected pixels per column; avoid divide-by-zero
        cnt_cols = mask_2d.sum(axis=0).astype(np.int64, copy=False)
        np.maximum(cnt_cols, 1, out=cnt_cols)

        # Sum selected pixels per column without allocating a full temporary
        try:
            sum_cols = np.sum(w2d, axis=0, dtype=np.int64, where=mask_2d)
        except TypeError:
            # Fallback for older numpy that lacks "where=" in np.sum
            sum_cols = np.sum(np.where(mask_2d, w2d, 0), axis=0, dtype=np.int64)

        col_med[:] = (sum_cols // cnt_cols).astype(np.int32, copy=False)
        w2d -= col_med

    def _process_buf(self, buf: bytearray) -> bytearray:
        """
        Heavy processing function (runs in worker threads).
        Operates in-place on the provided bytearray buffer.
        """
        # Writable view into the uint16 image inside the buffer
        arr_u2 = np.frombuffer(
            buf, dtype=np.dtype('<u2'),
            count=self.U16_COUNT, offset=self.HEAD_LEN
        ).reshape(self.NY, self.NX)

        ws = self._ws()
        work_i32 = ws.work_i32

        # raw -> i32: raw - dark
        # (casting='unsafe' is fine since both are numeric arrays)
        np.subtract(arr_u2, self.dark_i32, out=work_i32, casting='unsafe')

        # Dynamic threshold: 4*n1 + mean(raw-dark)
        mean_i32 = int(work_i32.sum(dtype=np.int64) // self.U16_COUNT)
        thr = 4 * self.n1 + mean_i32
        if thr < 0:
            thr = 0
        elif thr > 0xFFFF:
            thr = 0xFFFF

        # Column common-mode correction
        self._col_common_mode(work_i32, ws.mask_2d, ws.col_med, thr)

        # Bad pixel filter: set bad pixels to 0
        if self.bad_mask is not None:
            work_i32[self.bad_mask] = 0

        # Clip & write back to uint16 image area
        np.clip(work_i32, self.clamp_min, self.clamp_max, out=work_i32)
        arr_u2[:, :] = work_i32.astype(np.uint16, copy=False)

        # Timestamp (preserve original behavior: seconds, written at offset 28)
        ts_s = int(time.time())
        struct.pack_into('<I', buf, 28, ts_s & 0xFFFFFFFF)

        return buf

    def _acceptFrame(self, frame):
        """
        Rogue receive callback.
        Keep this fast:
          - Validate payload
          - Acquire in-flight token
          - Read into bytearray
          - Submit to thread pool
          - Append future to deque for in-order sending
        """
        size = frame.getPayload()

        # Only process fully valid frames
        data_bytes = max(0, size - self.HEAD_LEN)
        valid_bytes = min(self.DATA_LEN, data_bytes)
        if valid_bytes < self.DATA_LEN:
            return

        # In-flight gate: either drop or block
        if self.drop_if_busy:
            if not self._sem.acquire(blocking=False):
                return
        else:
            self._sem.acquire()

        # Copy payload into a mutable buffer (required because we modify it in-place)
        buf = bytearray(size)
        frame.read(buf, 0)

        # Submit heavy work. Release semaphore via callback no matter what.
        fut = self._exec.submit(self._process_buf, buf)
        fut.add_done_callback(lambda _f: self._sem.release())

        # Preserve submission order
        with self._lock:
            self._futs.append(fut)

    def _sender_loop(self):
        """
        Sender thread:
          - Maintains output order by only sending when the head future is done.
          - If a worker raises, we drop that frame's output (but semaphore already released).
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
                # Worker failed; skip sending this frame
                continue

            out = self._reqFrame(len(out_buf), True)
            out.write(out_buf, 0)
            self._sendFrame(out)

    def stop(self):
        """Stop threads and shutdown executor."""
        self._stop.set()
        try:
            self._sender.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._exec.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
