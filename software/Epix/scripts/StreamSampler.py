
import time
import rogue.interfaces.stream
class StreamSampler(rogue.interfaces.stream.Slave, rogue.interfaces.stream.Master):
    def __init__(self, min_interval: float = 1.0):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)
        self.min_interval = float(min_interval)
        self._last_t = 0.0

    def _acceptFrame(self, frame):
        now = time.monotonic()
        if (now - self._last_t) < self.min_interval:
            return
        self._last_t = now

        size = frame.getPayload()
        buf  = bytearray(size)
        frame.read(buf, 0)
        out = self._reqFrame(size, True)
        out.write(buf, 0)
        self._sendFrame(out)