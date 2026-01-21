import os
import struct
import time
import threading
import queue
import multiprocessing as mp
import numpy as np
import rogue.interfaces.stream


class L1BitmaskCompressor(rogue.interfaces.stream.Slave,
                          rogue.interfaces.stream.Master):
    HEAD_IN    = 32
    NY, NX     = 176, 768
    NPIX       = NY * NX
    DATA_IN    = NPIX * 2
    MAGIC      = b'E1BM'
    MASK_BYTES = NPIX // 8
    E1_FMT     = "<4sHHHHII"

    def __init__(self,
                 threshold=50,
                 emit_empty=False,
                 n_workers=None,
                 in_queue_max=None,
                 out_queue_max=None,
                 drop_if_busy=True,
                 stats_interval_s=2.0):

        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        self.thr = int(threshold)
        self.emit_empty = bool(emit_empty)
        self.drop_if_busy = bool(drop_if_busy)

        if n_workers is None:
            n_workers = 8
        self.n_workers = int(n_workers)

        if in_queue_max is None:
            in_queue_max = 2048
        if out_queue_max is None:
            out_queue_max = 2048

        self._stats_interval_s = float(stats_interval_s)

        self._in_cnt = 0
        self._out_cnt = 0
        self._drop_cnt = 0
        self._last_stats_t = time.time()
        self._lock = threading.Lock()

        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        self._stop_evt = ctx.Event()
        self._in_q = ctx.Queue(in_queue_max)
        self._out_q = ctx.Queue(out_queue_max)

        # ---- workers ----
        self._workers = []
        for _ in range(self.n_workers):
            p = ctx.Process(
                target=_l1bm_worker,
                args=(self._in_q, self._out_q, self._stop_evt,
                      self.thr, self.emit_empty),
                daemon=True
            )
            p.start()
            self._workers.append(p)

        # ---- sender thread ----
        self._sender_stop = threading.Event()
        self._sender_th = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_th.start()

        # ---- stats thread（可选但很有用）----
        self._stats_stop = threading.Event()
        self._stats_th = threading.Thread(target=self._stats_loop, daemon=True)
        self._stats_th.start()

    def _acceptFrame(self, frame):
        size = frame.getPayload()
        if size < self.HEAD_IN + self.DATA_IN:
            return

        buf = bytearray(size)
        frame.read(buf, 0)
        b = bytes(buf)

        try:
            if self.drop_if_busy:
                self._in_q.put_nowait(b)
            else:
                self._in_q.put(b)
            with self._lock:
                self._in_cnt += 1
        except queue.Full:
            with self._lock:
                self._drop_cnt += 1
            return

    def _sender_loop(self):
        while not self._sender_stop.is_set():
            try:
                out_bytes = self._out_q.get(timeout=0.2)
            except queue.Empty:
                continue

            out = self._reqFrame(len(out_bytes), True)
            out.write(out_bytes, 0)
            self._sendFrame(out)

            with self._lock:
                self._out_cnt += 1

    def _stats_loop(self):
        while not self._stats_stop.is_set():
            time.sleep(1)
            now = time.time()
            if now - self._last_stats_t < self._stats_interval_s:
                continue
            self._last_stats_t = now

            with self._lock:
                in_cnt = self._in_cnt
                out_cnt = self._out_cnt
                drop_cnt = self._drop_cnt
                try:
                    in_qsz = self._in_q.qsize()
                except Exception:
                    in_qsz = -1
                try:
                    out_qsz = self._out_q.qsize()
                except Exception:
                    out_qsz = -1

            print(f"[L1BM] in={in_cnt} out={out_cnt} drop={drop_cnt} "
                  f"in_q={in_qsz} out_q={out_qsz} workers={self.n_workers}")

    def stop(self):
        self._stop_evt.set()
        self._sender_stop.set()
        self._stats_stop.set()

        if self._sender_th.is_alive():
            self._sender_th.join(timeout=1.0)
        if self._stats_th.is_alive():
            self._stats_th.join(timeout=1.0)

        for p in self._workers:
            if p.is_alive():
                p.join(timeout=1.0)
        for p in self._workers:
            if p.is_alive():
                p.terminate()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


def _l1bm_worker(in_q, out_q, stop_evt, thr, emit_empty):
    HEAD_IN = 32
    NY, NX = 176, 768
    NPIX = NY * NX
    DATA_IN = NPIX * 2
    MAGIC = b'E1BM'
    MASK_BYTES = NPIX // 8
    E1_FMT = "<4sHHHHII"

    while not stop_evt.is_set():
        try:
            b = in_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if len(b) < HEAD_IN + DATA_IN:
            continue

        orig32 = b[:HEAD_IN]
        img = np.frombuffer(b, dtype="<u2", count=NPIX, offset=HEAD_IN).reshape(NY, NX)

        mask = img > thr
        cnt = int(mask.sum())
        if cnt == 0 and not emit_empty:
            continue

        mask_bytes = np.packbits(mask.reshape(-1), bitorder="big")
        vals = img[mask]

        out_len = HEAD_IN + 24 + MASK_BYTES + cnt * 2
        out = bytearray(out_len)

        out[:HEAD_IN] = orig32
        struct.pack_into(E1_FMT, out, HEAD_IN,
                         MAGIC, NY, NX, thr, 0,
                         cnt, MASK_BYTES)

        p = HEAD_IN + 24
        out[p:p + MASK_BYTES] = mask_bytes.tobytes()
        np.frombuffer(out, dtype="<u2", count=cnt, offset=p + MASK_BYTES)[:] = vals

        try:
            out_q.put(out, timeout=0.2)
        except Exception:
            pass
