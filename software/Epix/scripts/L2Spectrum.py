# L1SpectrumTap.py
import numpy as np
import rogue.interfaces.stream  


# L2 Spectrum is a side stream, Sample the frame and output it; 
class L2Spectrum(rogue.interfaces.stream.Slave,
                    rogue.interfaces.stream.Master):
    """
    Output the spectrum of the whole events, the 0-20keV will be precise while the 20-200keV will be difficult; 
    """
    HEAD_IN   = 32
    NY, NX    = 176, 768
    NPIX      = NY * NX           # 135,168
    DATA_IN   = NPIX * 2          # 270,336 B

    MAX_VAL   = 200 * 256         # 51200
    SPLIT     = 20  * 256         # 5120
    LOW_STEP  = 4
    HIGH_STEP = 256

    # From 0 to 20keV, the precision is 1/64keV, from 20keV to 200keV, the precision is 1keV
    NB_LO     = SPLIT // LOW_STEP                 # 5120/4   = 1280
    NB_HI     = (MAX_VAL - SPLIT) // HIGH_STEP    # (51200-5120)/256 = 180
    NBINS     = NB_LO + NB_HI                     # 1460

    def __init__(self, every_n: int = 100):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)
        self._counts = np.zeros(self.NBINS, dtype=np.uint32)
        self.every_n = max(1, int(every_n))
        self._idx    = 0  # Counters

    def _acceptFrame(self, frame):
        size = frame.getPayload()
        if size < self.HEAD_IN + self.DATA_IN:
            self._idx += 1
            return  # Discard Incomplete frames 

        # Only take the sampled data; 
        take = (self._idx % self.every_n) == 0
        if not take:
            self._idx += 1
            return

        # Read the whole frame; 
        buf = bytearray(size)
        frame.read(buf, 0)

        # The original head
        orig32 = buf[:self.HEAD_IN]

        # Reshape the data 
        img = np.frombuffer(buf, dtype=np.dtype('<u2'),
                            count=self.NPIX, offset=self.HEAD_IN
                           ).reshape(self.NY, self.NX)

        # Calculate the spectrum; 
        self._counts.fill(0)
        a = img.ravel().astype(np.int32, copy=False)
        np.clip(a, 0, self.MAX_VAL - 1, out=a)

        lo_mask = a < self.SPLIT
        if lo_mask.any():
            lo_bins = (a[lo_mask] >> 2)  # //4
            lo_cnt  = np.bincount(lo_bins, minlength=self.NB_LO)
            self._counts[:self.NB_LO] = lo_cnt[:self.NB_LO]

        if (~lo_mask).any():
            hi_vals = a[~lo_mask] - self.SPLIT
            hi_bins_rel = hi_vals // self.HIGH_STEP
            hi_cnt      = np.bincount(hi_bins_rel, minlength=self.NB_HI)
            self._counts[self.NB_LO:] = hi_cnt[:self.NB_HI]

        # Assemble and output 
        # Each bin includes 4Bytes, which is enough; 
        out_len = self.HEAD_IN + self.NBINS * 4
        out_buf = bytearray(out_len)
        out_buf[:self.HEAD_IN] = orig32

        mv = memoryview(out_buf)
        np.frombuffer(mv[self.HEAD_IN:], dtype='<u4', count=self.NBINS)[:] = self._counts

        f = self._reqFrame(out_len, True)
        f.write(out_buf, 0)
        self._sendFrame(f)

        self._idx += 1
