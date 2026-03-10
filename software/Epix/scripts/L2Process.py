# The level 2 process will reduce the position resolution

# L2BlockCounter.py
import numpy as np
import rogue.interfaces.stream  

class L2Process(rogue.interfaces.stream.Slave,
                     rogue.interfaces.stream.Master):
    """
    The Second order processing reduces the spatial resolution that
    bidding 4*4 pixels together, and then count how many 12-16keV events
    we have got. 
    """
    HEAD_IN  = 32
    NY, NX   = 176, 768
    NPIX     = NY * NX            # 135,168
    DATA_IN  = NPIX * 2           # 270,336 bytes
    BY      = NY // 4             # 44
    BX      = NX // 4             # 192
    OUT_PIX  = BY * BX            # 8,448

    def __init__(self, low_bin: int = 12, high_bin: int = 16, scale: int = 256):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)
        self.low_bin  = int(low_bin)
        self.high_bin = int(high_bin)
        self.scale    = int(scale)

    def _acceptFrame(self, frame):
        size = frame.getPayload()
        if size < self.HEAD_IN + self.DATA_IN:
            return  # 丢弃非完整帧

        # 读入整帧
        buf = bytearray(size)
        frame.read(buf, 0)

        # 32B 头 + 图像视图（<u2, 小端）
        orig32 = bytes(buf[:self.HEAD_IN])
        img = np.frombuffer(buf, dtype=np.dtype('<u2'),
                            count=self.NPIX, offset=self.HEAD_IN
                           ).reshape(self.NY, self.NX)

        # 阈值（闭开区间 [lo, hi)）
        lo = self.low_bin  * self.scale
        hi = self.high_bin * self.scale
        if hi <= lo:
            hi = lo  # 防止上限错误

        # 4x4 分块计数（向量化）
        # mask 为 True 的计入数量
        m = (img >= lo) & (img < hi)
        # 变形为 (44,4,192,4) 并在小块内求和 -> (44,192)
        counts = m.reshape(self.BY, 4, self.BX, 4).sum(axis=(1, 3)).astype(np.int8, copy=False)

        # 组装输出：32B 头 + 8448 B (i8)
        out_len = self.HEAD_IN + self.OUT_PIX
        out_buf = bytearray(out_len)
        out_buf[:self.HEAD_IN] = orig32
        mv = memoryview(out_buf)
        # int8 无端序问题
        np.frombuffer(mv[self.HEAD_IN:], dtype=np.uint8, count=self.OUT_PIX)[:] = counts.reshape(-1)

        # 发出压缩帧
        out = self._reqFrame(out_len, True)
        out.write(out_buf, 0)
        self._sendFrame(out)
