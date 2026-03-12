import time
import struct
import rogue.interfaces.stream


class TimestampProcess(rogue.interfaces.stream.Slave,
                       rogue.interfaces.stream.Master):
    HEAD_LEN = 32

    def __init__(self):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

    def _acceptFrame(self, frame):
        size = frame.getPayload()
        if size < self.HEAD_LEN:
            return

        buf = bytearray(size)
        frame.read(buf, 0)

        ns = time.time_ns()
        second = ns // 1_000_000_000
        microsecond = (ns % 1_000_000_000) // 1_000

        struct.pack_into('<I', buf, 20, microsecond & 0xFFFFFFFF)
        struct.pack_into('<I', buf, 28, second & 0xFFFFFFFF)

        out = self._reqFrame(size, True)
        out.write(buf, 0)
        self._sendFrame(out)