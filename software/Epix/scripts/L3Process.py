# L3StateAggregator_U16.py
import time
import struct
import threading
from collections import deque

import numpy as np
import rogue.interfaces.stream


class L3Process(rogue.interfaces.stream.Slave,
                rogue.interfaces.stream.Master):
    """
    L3 process:
      - consumes L2 output: 8448 x uint8 block counts + 32-byte header
      - accumulates counts over compression_ratio frames
      - keeps separate accumulation for two directions/states
      - emits 8448 x uint16 accumulated frame when enough frames arrive

    Notes:
      - this stage is stateful and order-sensitive
      - therefore it uses a single background worker thread, not a worker pool
      - this still lowers main-thread time in _acceptFrame()
    """

    HEAD_IN  = 32
    BY, BX   = 44, 192
    BLK_CNT  = BY * BX   # 8448
    DATA_IN  = BLK_CNT   # L2 output is uint8 per block
    DATA_OUT = BLK_CNT * 2  # L3 output is uint16 per block

    def __init__(self,
                 compression_ratio=400,
                 max_inflight=2048,
                 drop_if_busy=False):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        self.compression_ratio = int(compression_ratio)

        # state for direction 0 / 1
        self._state = [None, None]
        self._frames_acc = [0, 0]
        self._acc = [
            np.zeros(self.BLK_CNT, dtype=np.uint16),
            np.zeros(self.BLK_CNT, dtype=np.uint16)
        ]
        self._orig32_seg = [None, None]

        # bounded queue control
        self.max_inflight = int(max_inflight)
        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(self.max_inflight)

        # simple FIFO for input buffers
        self._queue = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # single worker thread
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @staticmethod
    def _state_from_word4(orig32: bytes) -> int:
        word = struct.unpack_from('<I', orig32, 16)[0]
        return 0 if word == 0 else 1

    # ------------------------------------------------------------------
    # emit one accumulated segment
    # ------------------------------------------------------------------
    def _emit_segment(self, st=0, orig32=None):
        if orig32 is None:
            orig32 = bytes(32)

        if self._state[st] is None or self._frames_acc[st] == 0:
            return

        out_len = self.HEAD_IN + self.DATA_OUT
        out_buf = bytearray(out_len)

        # keep header from the first frame of this segment
        out_buf[:self.HEAD_IN] = self._orig32_seg[st]

        # word 3: number of accumulated frames
        struct.pack_into('<I', out_buf, 12, int(self._frames_acc[st]))

        # word 6: run count from the last frame
        struct.pack_into('<I', out_buf, 24, struct.unpack_from('<I', orig32, 8)[0])

        # keep word 5 / word 7 timestamps from L0 header chain unchanged

        # append 8448 x uint16 counts
        np.frombuffer(
            out_buf,
            dtype=np.dtype('<u2'),
            count=self.BLK_CNT,
            offset=self.HEAD_IN
        )[:] = self._acc[st]

        f = self._reqFrame(out_len, True)
        f.write(out_buf, 0)
        self._sendFrame(f)

        # reset this state
        self._acc[st].fill(0)
        self._frames_acc[st] = 0
        self._orig32_seg[st] = None
        self._state[st] = None

    # ------------------------------------------------------------------
    # process one input buffer
    # ------------------------------------------------------------------
    def _process_buf(self, buf):
        orig32 = bytes(buf[:self.HEAD_IN])
        st = self._state_from_word4(orig32)

        # first frame of this segment for this state
        if self._state[st] is None:
            self._state[st] = st
            self._orig32_seg[st] = orig32

        vals_u8 = np.frombuffer(
            buf,
            dtype=np.uint8,
            count=self.BLK_CNT,
            offset=self.HEAD_IN
        )

        tmp32 = self._acc[st].astype(np.uint32) + vals_u8.astype(np.uint32)
        np.minimum(tmp32, 0xFFFF, out=tmp32)
        self._acc[st][:] = tmp32.astype(np.uint16)

        self._frames_acc[st] += 1

        if self._frames_acc[st] >= self.compression_ratio:
            self._emit_segment(st=st, orig32=orig32)

    # ------------------------------------------------------------------
    # background worker
    # ------------------------------------------------------------------
    def _worker_loop(self):
        while not self._stop.is_set():
            with self._lock:
                buf = self._queue.popleft() if self._queue else None

            if buf is None:
                time.sleep(0.0005)
                continue

            try:
                self._process_buf(buf)
            except Exception as e:
                print(f"[L3Process] worker failed: {e}")
            finally:
                self._sem.release()

    # ------------------------------------------------------------------
    # Rogue callback
    # ------------------------------------------------------------------
    def _acceptFrame(self, frame):
        if self._stop.is_set():
            return

        size = frame.getPayload()
        min_len = self.HEAD_IN + self.DATA_IN
        if size < min_len:
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

        with self._lock:
            if self._stop.is_set():
                self._sem.release()
                return
            self._queue.append(buf)

    # ------------------------------------------------------------------
    # flush remaining segments
    # ------------------------------------------------------------------
    def flush(self):
        # wait briefly for queue to drain
        while True:
            with self._lock:
                empty = (len(self._queue) == 0)
            if empty:
                break
            time.sleep(0.001)

        self._emit_segment(st=0)
        self._emit_segment(st=1)

    def stop(self):
        self._stop.set()

        try:
            self._worker.join(timeout=1.0)
        except Exception:
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass