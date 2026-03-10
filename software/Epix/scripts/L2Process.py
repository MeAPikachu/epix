# The level 2 process will reduce the position resolution

import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rogue.interfaces.stream


class L2Process(rogue.interfaces.stream.Slave,
                rogue.interfaces.stream.Master):
    """
    The second-order processing reduces the spatial resolution by
    binning 4x4 pixels together, and counts how many events in each block
    fall in the [low_bin, high_bin) energy window.

    Output format:
      - keep 32-byte header unchanged
      - append one uint8 count per 4x4 block
      - total output payload = 32 + 44*192 bytes
    """

    HEAD_IN  = 32
    NY, NX   = 176, 768
    NPIX     = NY * NX
    DATA_IN  = NPIX * 2

    BY       = NY // 4      # 44
    BX       = NX // 4      # 192
    OUT_PIX  = BY * BX      # 8448
    DATA_OUT = OUT_PIX      # uint8 per output pixel

    def __init__(self,
                 low_bin: int = 12,
                 high_bin: int = 16,
                 scale: int = 256,
                 n_workers=None,
                 max_inflight=2048,
                 drop_if_busy=False):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        self.low_bin = int(low_bin)
        self.high_bin = int(high_bin)
        self.scale = int(scale)

        if n_workers is None:
            cpu = os.cpu_count() or 8
            n_workers = max(4, min(16, cpu // 8))
        self.n_workers = int(n_workers)

        self.max_inflight = int(max_inflight)
        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(self.max_inflight)

        self._tls = threading.local()

        self._futs = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._exec = ThreadPoolExecutor(
            max_workers=self.n_workers,
            thread_name_prefix="L2W"
        )
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

    # ------------------------------------------------------------------
    # per-thread workspace
    # ------------------------------------------------------------------
    def _ws(self):
        ws = getattr(self._tls, "ws", None)
        if ws is not None:
            return ws

        class _WS:
            pass

        ws = _WS()

        # output counts, shaped 44x192
        ws.counts_2d = np.empty((self.BY, self.BX), dtype=np.uint8)

        self._tls.ws = ws
        return ws

    # ------------------------------------------------------------------
    # heavy worker
    # ------------------------------------------------------------------
    def _process_buf(self, buf):
        img = np.frombuffer(
            buf,
            dtype=np.dtype('<u2'),
            count=self.NPIX,
            offset=self.HEAD_IN
        ).reshape(self.NY, self.NX)

        ws = self._ws()

        lo = self.low_bin * self.scale
        hi = self.high_bin * self.scale
        if hi <= lo:
            hi = lo + 1

        # Boolean mask of hits in requested energy window
        m = (img >= lo) & (img < hi)

        # 4x4 block count -> shape (44, 192)
        counts = m.reshape(self.BY, 4, self.BX, 4).sum(axis=(1, 3), dtype=np.uint16)

        # copy to uint8 workspace
        ws.counts_2d[:, :] = counts.astype(np.uint8, copy=False)

        out_len = self.HEAD_IN + self.DATA_OUT
        out_buf = bytearray(out_len)

        # preserve original header
        out_buf[:self.HEAD_IN] = buf[:self.HEAD_IN]

        # append flattened uint8 counts
        np.frombuffer(
            out_buf,
            dtype=np.uint8,
            count=self.OUT_PIX,
            offset=self.HEAD_IN
        )[:] = ws.counts_2d.reshape(-1)

        return out_buf

    # ------------------------------------------------------------------
    # Rogue callbacks
    # ------------------------------------------------------------------
    def _acceptFrame(self, frame):
        if self._stop.is_set():
            return

        size = frame.getPayload()
        data_bytes = max(0, size - self.HEAD_IN)
        if data_bytes < self.DATA_IN:
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
            except Exception as e:
                print(f"[L2Process] worker failed: {e}")
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