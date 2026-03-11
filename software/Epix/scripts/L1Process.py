import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rogue.interfaces.stream
from numba import njit, prange, set_num_threads, get_num_threads


@njit(cache=True, parallel=True, fastmath=True)
def _centroid_sum_u16_to_f32(arr_u2, thr_cs, out_f32):
    ny, nx = arr_u2.shape

    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            c = arr_u2[y, x]
            if c < thr_cs:
                continue

            l = arr_u2[y, x - 1]
            r = arr_u2[y, x + 1]
            u = arr_u2[y - 1, x]
            d = arr_u2[y + 1, x]

            zl = 0 if l < thr_cs else l
            zr = 0 if r < thr_cs else r
            zu = 0 if u < thr_cs else u
            zd = 0 if d < thr_cs else d

            if c > zl and c > zr and c > zu and c > zd:
                out_f32[y, x] = float(c + zl + zr + zu + zd)


class L1Process(rogue.interfaces.stream.Slave, rogue.interfaces.stream.Master):
    """
    Level-1 processing:
      - optional centroid / charge-sharing sum
      - filter mask application
      - gain correction
      - clamp and write back to uint16

    Supports:
      - multithreaded processing
      - dynamic gain reload from /data/gain/gain.npy
      - dynamic filter reload from /data/gain/filter.npy
      - ordered output sending
    """

    HEAD_LEN  = 32
    NY        = 176
    NX        = 768
    U16_COUNT = NY * NX
    DATA_LEN  = U16_COUNT * 2

    def __init__(self,
                 gain_path="/data/gain/gain.npy",
                 filter_path="/data/gain/filter.npy",

                 # dynamic gain
                 dynamic_gain=False,
                 dynamic_gain_dir="/data/gain",
                 dynamic_gain_period_s=86400,

                 gain_scalar=None,
                 scale=256,
                 clamp_min=0,
                 clamp_max=0xFFFF,
                 round_mode='nearest',

                 # centroid
                 enable_centroid=True,
                 n2=15,
                 numba_threads=None,

                 # threading
                 n_workers=None,
                 max_inflight=2048,
                 drop_if_busy=False):

        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        # dynamic gain
        self.dynamic_gain = bool(dynamic_gain)
        self.dynamic_gain_period_s = int(dynamic_gain_period_s)

        if self.dynamic_gain:
            self.gain_path = os.path.join(dynamic_gain_dir, "gain.npy")
            self.filter_path = os.path.join(dynamic_gain_dir, "filter.npy")
        else:
            self.gain_path = gain_path
            self.filter_path = filter_path

        self._gain_lock = threading.Lock()
        self._gain_mtime = None
        self._filter_mtime = None

        # gain settings
        self.SCALE = float(scale)
        self._input_gain_scalar = gain_scalar
        self.coeff = None
        self.coeff_scalar = None
        self.filter_mask = None

        # processing settings
        self.clamp_min = int(clamp_min)
        self.clamp_max = int(clamp_max)
        self.round_mode = str(round_mode).lower()

        self.enable_centroid = bool(enable_centroid)
        self.n2 = int(n2)

        # numba threads
        self.numba_threads = numba_threads
        if self.numba_threads is not None:
            set_num_threads(int(self.numba_threads))
            print(f"[L1Process] numba threads = {get_num_threads()}")

        # initial gain load
        self._load_gain(initial=True)

        # thread pool sizing
        if n_workers is None:
            cpu = os.cpu_count() or 8
            n_workers = max(2, min(8, cpu // 8))
        self.n_workers = int(n_workers)

        # in-flight control
        self.max_inflight = int(max_inflight)
        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(self.max_inflight)

        # per-thread workspace
        self._tls = threading.local()

        # ordering and sender thread
        self._futs = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._exec = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix="L1W")
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

        # warm up numba once so the first real frame does not pay the compile cost
        if self.enable_centroid:
            dummy_in = np.zeros((self.NY, self.NX), dtype=np.uint16)
            dummy_out = np.zeros((self.NY, self.NX), dtype=np.float32)
            _centroid_sum_u16_to_f32(dummy_in, max(0, 4 * self.n2), dummy_out)

        # dynamic gain watcher
        if self.dynamic_gain:
            self._gain_thread = threading.Thread(
                target=self._gain_watcher,
                name="L1GainWatcher",
                daemon=True,
            )
            self._gain_thread.start()

    # ------------------------------------------------------------------
    # dynamic gain
    # ------------------------------------------------------------------
    def _load_gain(self, initial=False):
        if self.gain_path is not None:
            try:
                gain_mtime = os.path.getmtime(self.gain_path)
                filter_mtime = os.path.getmtime(self.filter_path) if self.filter_path else None
            except OSError:
                if initial:
                    raise RuntimeError(f"Initial gain/filter load failed: {self.gain_path}")
                return

            # Check if either file has changed
            if (not initial) and (gain_mtime == self._gain_mtime) and (filter_mtime == self._filter_mtime):
                return

            g = np.load(self.gain_path, mmap_mode="r")
            g = np.asarray(g, order="C")
            if g.size != self.U16_COUNT:
                raise ValueError(f"gain size {g.size} != {self.U16_COUNT}")

            g = g.reshape(self.NY, self.NX).astype(np.float32, copy=False)
            coeff = (self.SCALE / np.maximum(g, 1e-12)).astype(np.float32, copy=False)

            # Load filter
            f_mask = None
            if self.filter_path and os.path.exists(self.filter_path):
                f = np.load(self.filter_path, mmap_mode="r")
                f_mask = np.asarray(f).reshape(self.NY, self.NX).astype(np.float32, copy=False)

            with self._gain_lock:
                self.coeff = coeff
                self.coeff_scalar = None
                self.filter_mask = f_mask
                self._gain_mtime = gain_mtime
                self._filter_mtime = filter_mtime

            print(f"[L1Process] resources loaded: {os.path.basename(self.gain_path)}")
            return

        if self._input_gain_scalar is not None:
            gs = float(self._input_gain_scalar)
            if not np.isfinite(gs) or gs <= 0:
                raise ValueError(f"invalid gain_scalar={self._input_gain_scalar}")
            coeff_scalar = np.float32(self.SCALE / gs)
        else:
            coeff_scalar = np.float32(self.SCALE / 17.0)

        with self._gain_lock:
            self.coeff = None
            self.coeff_scalar = coeff_scalar
            self.filter_mask = None
            self._gain_mtime = None
            self._filter_mtime = None

    def _gain_watcher(self):
        while not self._stop.is_set():
            time.sleep(self.dynamic_gain_period_s)
            try:
                self._load_gain(initial=False)
            except Exception as e:
                print(f"[L1Process] gain reload failed: {e}")

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
        ws.work_f32 = np.empty((self.NY, self.NX), dtype=np.float32)

        self._tls.ws = ws
        return ws

    # ------------------------------------------------------------------
    # heavy worker
    # ------------------------------------------------------------------
    def _process_buf(self, buf):
        arr_u2 = np.frombuffer(
            buf,
            dtype=np.dtype('<u2'),
            count=self.U16_COUNT,
            offset=self.HEAD_LEN
        ).reshape(self.NY, self.NX)

        ws = self._ws()

        with self._gain_lock:
            coeff = self.coeff
            coeff_scalar = self.coeff_scalar
            f_mask = self.filter_mask

        # centroid on L0 output
        if self.enable_centroid:
            thr_cs = max(0, 4 * self.n2)
            ws.work_f32.fill(0.0)
            _centroid_sum_u16_to_f32(arr_u2, thr_cs, ws.work_f32)
        else:
            ws.work_f32[:] = arr_u2

        # filter application
        if f_mask is not None:
            np.multiply(ws.work_f32, f_mask, out=ws.work_f32)

        # gain correction
        if coeff is not None:
            np.multiply(ws.work_f32, coeff, out=ws.work_f32)
        else:
            ws.work_f32 *= coeff_scalar

        # clamp
        np.clip(ws.work_f32, self.clamp_min, self.clamp_max, out=ws.work_f32)

        # rounding
        if self.round_mode == 'nearest':
            np.rint(ws.work_f32, out=ws.work_f32)
        elif self.round_mode == 'floor':
            np.floor(ws.work_f32, out=ws.work_f32)
        elif self.round_mode == 'ceil':
            np.ceil(ws.work_f32, out=ws.work_f32)
        elif self.round_mode == 'none':
            pass
        else:
            raise ValueError(f"unsupported round_mode={self.round_mode}")

        arr_u2[:, :] = ws.work_f32.astype(np.uint16, copy=False)
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
            except Exception as e:
                print(f"[L1Process] worker failed: {e}")
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