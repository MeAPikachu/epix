# L1BitmaskCompressor.py
import numpy as np
import struct
import rogue.interfaces.stream  

class L1BitmaskCompressor(rogue.interfaces.stream.Slave,
                          rogue.interfaces.stream.Master):
    HEAD_IN    = 32                 # The first 32B is the headers 
    NY, NX     = 176, 768
    NPIX       = NY * NX            # 135,168
    DATA_IN    = NPIX * 2           # 270,336B Raw frames ; 
    MAGIC      = b'E1BM'            # A special word for identification 
    MASK_BYTES = NPIX // 8          # 16,896B

    def __init__(self, threshold: int = 50, emit_empty: bool = False):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)
        self.thr = int(threshold)
        self.emit_empty = bool(emit_empty)

    def _acceptFrame(self, frame):
        size = frame.getPayload()
        
        if size < self.HEAD_IN + self.DATA_IN:
            return

        
        buf = bytearray(size)
        frame.read(buf, 0)

        # Keep the original 32B 
        orig32 = bytes(buf[:self.HEAD_IN])

        # Load the frame main data 
        img = np.frombuffer(buf, dtype=np.dtype('<u2'),
                            count=self.NPIX, offset=self.HEAD_IN
                           ).reshape(self.NY, self.NX)

        # Only keep the values that are greater than the threshold 
        mask_bool = img > self.thr
        count = int(mask_bool.sum())
        if count == 0 and not self.emit_empty:
            return  

        # The position is MSB 
        mask_bytes = np.packbits(mask_bool.reshape(-1), bitorder='big')

        # The value is LSB, uint16; 
        vals = img[mask_bool].astype(np.uint16, copy=False)

        # Calculate the length of the output frame ; 
        payload_len = self.HEAD_IN + 24 + self.MASK_BYTES + count * 2
        out_buf = bytearray(payload_len)

        # Send the original 32B head first 
        out_buf[0:self.HEAD_IN] = orig32

        # The special communication header
        struct.pack_into('<4sHHHHII', out_buf, self.HEAD_IN,
                         self.MAGIC, self.NY, self.NX, self.thr, 0,
                         count, self.MASK_BYTES)

        # The position itself 
        start = self.HEAD_IN + 24
        out_buf[start:start + self.MASK_BYTES] = mask_bytes.tobytes()

        # The values 
        mv = memoryview(out_buf)
        np.frombuffer(mv[start + self.MASK_BYTES:], dtype='<u2', count=count)[:] = vals

        # Send the frame out ; 
        out = self._reqFrame(payload_len, True)
        out.write(out_buf, 0)
        self._sendFrame(out)
