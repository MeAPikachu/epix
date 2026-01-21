import struct
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
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

    def __init__(self, threshold=50, emit_empty=False,
                 n_workers=6, max_inflight=2048,
                 drop_if_busy=False):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        self.thr = int(threshold)
        self.emit_empty = bool(emit_empty)

        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(int(max_inflight))

        self._pool = ThreadPoolExecutor(max_workers=int(n_workers))

        self._out_q = queue.Queue()
        self._sender_stop = threading.Event()
        self._stop = threading.Event()  # global gate for shutdown

        self._sender_th = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_th.start()

        self._seq = 0
        self._seq_lock = threading.Lock()

    def _acceptFrame(self, frame):
        # Gate immediately if stopping
        if self._stop.is_set() or self._sender_stop.is_set():
            return

        size = frame.getPayload()
        if size < self.HEAD_IN + self.DATA_IN:
            return

        # In-flight gate
        if self.drop_if_busy:
            if not self._sem.acquire(blocking=False):
                return
        else:
            self._sem.acquire()

        # stop could be set while waiting for semaphore
        if self._stop.is_set() or self._sender_stop.is_set():
            self._sem.release()
            return

        buf = bytearray(size)
        frame.read(buf, 0)

        # NOTE: keep bytes immutable for frombuffer safety
        b = bytes(buf)

        with self._seq_lock:
            seq = self._seq
            self._seq += 1

        # Submit heavy work; handle shutdown race gracefully
        try:
            fut = self._pool.submit(self._compress_one, b, seq)
        except RuntimeError:
            # "cannot schedule new futures after shutdown"
            self._sem.release()
            return

        fut.add_done_callback(self._on_done)

    def _compress_one(self, b: bytes, seq: int):
        # If stopping, short-circuit (still returns a tuple)
        if self._stop.is_set() or self._sender_stop.is_set():
            return (seq, None)

        orig32 = b[:self.HEAD_IN]
        img = np.frombuffer(b, dtype="<u2", count=self.NPIX, offset=self.HEAD_IN).reshape(self.NY, self.NX)

        mask = img > self.thr
        cnt = int(mask.sum())
        if cnt == 0 and (not self.emit_empty):
            return (seq, None)

        mask_bytes = np.packbits(mask.reshape(-1), bitorder="big")
        vals = img[mask]

        out_len = self.HEAD_IN + 24 + self.MASK_BYTES + cnt * 2
        out = bytearray(out_len)

        out[:self.HEAD_IN] = orig32
        struct.pack_into(self.E1_FMT, out, self.HEAD_IN,
                         self.MAGIC, self.NY, self.NX, self.thr, 0,
                         cnt, self.MASK_BYTES)

        p = self.HEAD_IN + 24
        out[p:p + self.MASK_BYTES] = mask_bytes.tobytes()
        np.frombuffer(out, dtype="<u2", count=cnt, offset=p + self.MASK_BYTES)[:] = vals

        return (seq, out)

    def _on_done(self, fut):
        # Always enqueue a result or release the semaphore here on failure,
        # so we never leak in-flight tokens.
        try:
            seq, out = fut.result()
        except Exception:
            self._sem.release()
            return

        # If we're stopping, don't let the sender wait for gaps forever:
        # still enqueue (seq, None) so ordering can advance.
        if self._stop.is_set() or self._sender_stop.is_set():
            out = None

        self._out_q.put((seq, out))

    def _sender_loop(self):
        next_seq = 0
        pending = {}

        while not self._sender_stop.is_set():
            try:
                seq, out = self._out_q.get(timeout=0.2)
            except queue.Empty:
                continue

            pending[seq] = out

            while next_seq in pending:
                out_bytes = pending.pop(next_seq)
                next_seq += 1

                if out_bytes is not None:
                    fr = self._reqFrame(len(out_bytes), True)
                    fr.write(out_bytes, 0)
                    self._sendFrame(fr)

                # Release one in-flight token per completed sequence
                self._sem.release()

        # Drain remaining pending items on stop to avoid leaking sem tokens
        # (best-effort: release for any buffered completions)
        try:
            while True:
                seq, out = self._out_q.get_nowait()
                pending[seq] = out
        except queue.Empty:
            pass

        # Release any remaining items we've already received
        while next_seq in pending:
            pending.pop(next_seq, None)
            next_seq += 1
            self._sem.release()

    def stop(self):
        # Gate first: prevents new submissions immediately
        self._stop.set()
        self._sender_stop.set()

        if self._sender_th.is_alive():
            self._sender_th.join(timeout=1.0)

        # Shutdown executor last; any late _acceptFrame submit will be caught
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9: no cancel_futures
            self._pool.shutdown(wait=False)
        except Exception:
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
