import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rogue.interfaces.stream


class L1Process(rogue.interfaces.stream.Slave, rogue.interfaces.stream.Master):
    """
    Level-1 processing:
      - optional centroid / charge-sharing sum
      - gain correction
      - clamp and write back to uint16

    Supports:
      - multithreaded processing
      - dynamic gain reload from /data/gain/gain.npy
      - ordered output sending
    """

    HEAD_LEN  = 32
    NY        = 176
    NX        = 768
    U16_COUNT = NY * NX
    DATA_LEN  = U16_COUNT * 2

    def __init__(self,
                 gain_path="/data/gain/gain.npy",

                 # === dynamic gain ===
                 dynamic_gain=False,
                 dynamic_gain_dir="/data/gain",
                 dynamic_gain_period_s=86400,
                 # ====================

                 gain_scalar=None,
                 scale=256,
                 clamp_min=0,
                 clamp_max=0xFFFF,
                 round_mode='nearest',

                 # centroid
                 enable_centroid=True,
                 n2=15,
                 centroid_use_ge=False,

                 # threading
                 n_workers=None,
                 max_inflight=2048,
                 drop_if_busy=False
                 ):

        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        # === dynamic gain ===
        self.dynamic_gain = bool(dynamic_gain)
        self.dynamic_gain_period_s = int(dynamic_gain_period_s)

        if self.dynamic_gain:
            self.gain_path = os.path.join(dynamic_gain_dir, "gain.npy")
        else:
            self.gain_path = gain_path

        self._gain_lock = threading.Lock()
        self._gain_mtime = None

        # ---- gain settings ----
        self.SCALE = float(scale)
        self._input_gain_scalar = gain_scalar
        self.coeff = None
        self.coeff_scalar = None

        # ---- processing settings ----
        self.clamp_min = int(clamp_min)
        self.clamp_max = int(clamp_max)
        self.round_mode = str(round_mode).lower()

        self.enable_centroid = bool(enable_centroid)
        self.n2 = int(n2)
        self.centroid_use_ge = bool(centroid_use_ge)

        # ---- initial gain load ----
        self._load_gain(initial=True)

        # ---- thread pool sizing ----
        if n_workers is None:
            cpu = os.cpu_count() or 8
            n_workers = max(4, min(16, cpu // 8))
        self.n_workers = int(n_workers)

        # ---- in-flight control ----
        self.max_inflight = int(max_inflight)
        self.drop_if_busy = bool(drop_if_busy)
        self._sem = threading.Semaphore(self.max_inflight)

        # ---- per-thread workspace ----
        self._tls = threading.local()

        # ---- ordering and sender thread ----
        self._futs = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._exec = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix="L1W")
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

        # === dynamic gain watcher ===
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
        # file-based gain
        if self.gain_path is not None:
            try:
                gain_mtime = os.path.getmtime(self.gain_path)
            except OSError:
                if initial:
                    raise RuntimeError(f"Initial gain load failed: {self.gain_path}")
                return

            if (not initial) and (gain_mtime == self._gain_mtime):
                return

            g = np.load(self.gain_path, mmap_mode="r")
            g = np.asarray(g, order="C")
            if g.size != self.U16_COUNT:
                raise ValueError(f"gain size {g.size} != {self.U16_COUNT}")

            g = g.reshape(self.NY, self.NX).astype(np.float32, copy=False)
            coeff = (self.SCALE / np.maximum(g, 1e-12)).astype(np.float32, copy=False)

            with self._gain_lock:
                self.coeff = coeff
                self.coeff_scalar = None
                self._gain_mtime = gain_mtime

            print(f"[L1Process] gain loaded: {os.path.basename(self.gain_path)}")
            return

        # scalar fallback
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
            self._gain_mtime = None

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
        ws.work_i32 = np.empty((self.NY, self.NX), dtype=np.int32)
        ws.out_i32  = np.empty((self.NY, self.NX), dtype=np.int32)
        ws.cmask    = np.empty((self.NY - 2, self.NX - 2), dtype=bool)
        ws.work_f32 = np.empty((self.NY, self.NX), dtype=np.float32)

        self._tls.ws = ws
        return ws

    # ------------------------------------------------------------------
    # centroid
    # ------------------------------------------------------------------
    def _centroid_4n_sum_thr(self, work_i32, out_i32, cmask, thr_cs):
        """
        Threshold first:
            work_i32[work_i32 < thr_cs] = 0

        Centroid condition:
            center > left,right,up,down
        or:
            center >= left,right,up,down   if centroid_use_ge=True

        Output value at centroid:
            center + left + right + up + down

        Non-centroid pixels:
            0
        """
        out_i32.fill(0)
        work_i32[work_i32 < thr_cs] = 0

        c = work_i32[1:-1, 1:-1]
        l = work_i32[1:-1, 0:-2]
        r = work_i32[1:-1, 2:  ]
        u = work_i32[0:-2, 1:-1]
        d = work_i32[2:  , 1:-1]

        cmask[:] = (c != 0)
        if self.centroid_use_ge:
            cmask &= (c >= l) & (c >= r) & (c >= u) & (c >= d)
        else:
            cmask &= (c >  l) & (c >  r) & (c >  u) & (c >  d)

        acc = out_i32[1:-1, 1:-1]
        acc[:] = c
        acc += l
        acc += r
        acc += u
        acc += d
        acc[~cmask] = 0

    # ------------------------------------------------------------------
    # heavy worker
    # ------------------------------------------------------------------
    def _process_buf(self, buf):
        arr_u2 = np.frombuffer(
            buf, dtype=np.dtype('<u2'),
            count=self.U16_COUNT, offset=self.HEAD_LEN
        ).reshape(self.NY, self.NX)

        ws = self._ws()

        with self._gain_lock:
            coeff = self.coeff
            coeff_scalar = self.coeff_scalar

        # centroid on L0 output
        if self.enable_centroid:
            ws.work_i32[:] = arr_u2
            thr_cs = max(0, min(0x7FFFFFFF, 4 * self.n2))
            self._centroid_4n_sum_thr(ws.work_i32, ws.out_i32, ws.cmask, thr_cs)
            ws.work_f32[:] = ws.out_i32
        else:
            ws.work_f32[:] = arr_u2

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