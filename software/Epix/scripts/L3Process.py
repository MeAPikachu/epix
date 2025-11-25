# L3StateAggregator_U16.py
import numpy as np
import struct
import rogue.interfaces.stream  # 不用别名

class L3Process(rogue.interfaces.stream.Slave,
						rogue.interfaces.stream.Master):
	"""
    The L3 Process reduces the time resolution, based on the direction
    of the DAQ input, adds the 
	"""
	HEAD_IN  = 32
	BY, BX   = 44, 192
	BLK_CNT  = BY * BX  # 8448

	def __init__(self):
		rogue.interfaces.stream.Slave.__init__(self)
		rogue.interfaces.stream.Master.__init__(self)
		
		self._state = None                 # 当前段状态 0/1
		self._frames_acc = 0               # 当前段累计帧数（若需要可另行写回头部）
		self._acc = np.zeros(self.BLK_CNT, dtype=np.uint16)   # u16 累计
		self._orig32_seg = None            # 当前段第一帧的 32B 原头

	@staticmethod
	def _state_from_word3(orig32: bytes) -> int:
		# word3 小端 u32
		w3 = struct.unpack_from('<I', orig32, 12)[0]
		return 0 if w3 == 0 else 1

	def _emit_segment(self):
		if self._state is None or self._frames_acc == 0:
			return

		out_len = self.HEAD_IN + self.BLK_CNT * 2
		out_buf = bytearray(out_len)

		# 32B 原头（保持原样；如需写入 frames_acc 可在此覆盖某字）
		out_buf[:self.HEAD_IN] = self._orig32_seg

		# 8448×u16 小端
		mv = memoryview(out_buf)
		np.frombuffer(mv[self.HEAD_IN:], dtype='<u2', count=self.BLK_CNT)[:] = self._acc

		f = self._reqFrame(out_len, True)
		f.write(out_buf, 0)
		self._sendFrame(f)


		self._acc.fill(0)
		self._frames_acc = 0
		self._orig32_seg = None

	def _acceptFrame(self, frame):
		size = frame.getPayload()
		min_len = self.HEAD_IN + self.BLK_CNT  # 32 + 8448（L2 输出为 u8）
		if size < min_len:
			return  

		buf = bytearray(size)
		frame.read(buf, 0)

		orig32 = bytes(buf[:self.HEAD_IN])
		st = self._state_from_word3(orig32)

		if self._state is None:
			self._state = st
			self._orig32_seg = orig32
		elif st != self._state:
			self._emit_segment()
			self._state = st
			self._orig32_seg = orig32

		vals_u8 = np.frombuffer(buf, dtype=np.uint8,
								offset=self.HEAD_IN, count=self.BLK_CNT)
		tmp32 = self._acc.astype(np.uint32) + vals_u8.astype(np.uint32)
		np.minimum(tmp32, 0xFFFF, out=tmp32)
		self._acc[:] = tmp32.astype(np.uint16)

		self._frames_acc += 1

	def flush(self):
		
		self._emit_segment()
