import time
import rogue.interfaces.stream
class StreamSampler(rogue.interfaces.stream.Slave, rogue.interfaces.stream.Master):
	HEAD_LEN  = 32
	NY        = 176
	NX        = 768
	U16_COUNT = NY * NX                  # 135,168
	DATA_LEN  = U16_COUNT * 2   
    
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
		# Only process the valid bytes; 
		data_bytes = max(0, size - self.HEAD_LEN)
		valid_bytes = min(self.DATA_LEN, data_bytes)
		if valid_bytes < self.DATA_LEN:
			return

		buf  = bytearray(size)
		frame.read(buf, 0)
		out = self._reqFrame(size, True)
		out.write(buf, 0)
		self._sendFrame(out)