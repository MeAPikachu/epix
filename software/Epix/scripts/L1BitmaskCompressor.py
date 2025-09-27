# L1BitmaskCompressor.py
import numpy as np
import struct
import rogue.interfaces.stream  # 不用别名

class L1BitmaskCompressor(rogue.interfaces.stream.Slave,
                          rogue.interfaces.stream.Master):
    """
    位图+值压缩（保留上游原始32B帧头）：
      帧布局：
        [orig32]    32 字节：上游帧头原样拷贝
        [E1BM hdr]  24 字节，小端：
            magic[4] = b'E1BM'
            ny<u16>, nx<u16>, thr<u16>, rsv<u16>=0
            count<u32>        # 命中像素个数
            maskBytes<u32>    # 固定 16896
        [mask]      16896 字节：np.packbits(布尔位图, bitorder='big')
        [values]    count * 2 字节：<u2，小端，顺序与位图中 1 的顺序一致（行优先）
    """
    HEAD_IN    = 32                 # 上游“相机头”在payload最前端，占 32B
    NY, NX     = 176, 768
    NPIX       = NY * NX            # 135,168
    DATA_IN    = NPIX * 2           # 270,336B 原图像区
    MAGIC      = b'E1BM'
    MASK_BYTES = NPIX // 8          # 16,896B

    def __init__(self, threshold: int = 50, emit_empty: bool = False):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)
        self.thr = int(threshold)
        self.emit_empty = bool(emit_empty)

    def _acceptFrame(self, frame):
        size = frame.getPayload()
        # 仅处理完整帧：32B头 + 270336B数据
        if size < self.HEAD_IN + self.DATA_IN:
            return

        # 读入整帧
        buf = bytearray(size)
        frame.read(buf, 0)

        # 保留原始32字节头
        orig32 = bytes(buf[:self.HEAD_IN])

        # 图像视图（<u2, LE）-> (176,768)
        img = np.frombuffer(buf, dtype=np.dtype('<u2'),
                            count=self.NPIX, offset=self.HEAD_IN
                           ).reshape(self.NY, self.NX)

        # 阈值筛选
        mask_bool = img > self.thr
        count = int(mask_bool.sum())
        if count == 0 and not self.emit_empty:
            return  # 丢弃空帧以节省带宽

        # 位图：行优先打包，MSB先（解码时同样设置 bitorder='big'）
        mask_bytes = np.packbits(mask_bool.reshape(-1), bitorder='big')

        # 命中值（小端写出）
        vals = img[mask_bool].astype(np.uint16, copy=False)

        # 组装输出缓冲：32 + 24 + 16896 + 2*count
        payload_len = self.HEAD_IN + 24 + self.MASK_BYTES + count * 2
        out_buf = bytearray(payload_len)

        # 写入原32B头
        out_buf[0:self.HEAD_IN] = orig32

        # 写 E1BM 头（从 offset=32 开始）
        struct.pack_into('<4sHHHHII', out_buf, self.HEAD_IN,
                         self.MAGIC, self.NY, self.NX, self.thr, 0,
                         count, self.MASK_BYTES)

        # 写位图
        start = self.HEAD_IN + 24
        out_buf[start:start + self.MASK_BYTES] = mask_bytes.tobytes()

        # 写值数组（<u2, LE）
        mv = memoryview(out_buf)
        np.frombuffer(mv[start + self.MASK_BYTES:], dtype='<u2', count=count)[:] = vals

        # 发送变长压缩帧
        out = self._reqFrame(payload_len, True)
        out.write(out_buf, 0)
        self._sendFrame(out)
