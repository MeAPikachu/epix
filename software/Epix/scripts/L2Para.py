# L2BlockCounter8x8_U32.py
import numpy as np
import rogue.interfaces.stream  # 不用别名

# L2 Para is used to calculate the distribution of the 122keV; 
class L2Para(rogue.interfaces.stream.Slave,
						rogue.interfaces.stream.Master):
	"""
  The Parallel L2 Processing is used to get the spatial distribution
  of the 122keV events, the distribution has worse;
	"""

	# The basic parameters 
	HEAD_IN   = 32
	NY, NX    = 176, 768
	NPIX      = NY * NX                 # 135,168
	DATA_IN   = NPIX * 2                # 270,336 B

	# 8x8 Blocks, this is different from the 4*4 block of the 14keV; 
	BY, BX    = NY // 8, NX // 8        # 22, 96
	NBLOCK    = BY * BX                 # 2,112

	def __init__(self,
				 low_bin: int = 100,
				 high_bin: int = 140,
				 scale: int = 256,
				 group_frames: int = 100):
		rogue.interfaces.stream.Slave.__init__(self)
		rogue.interfaces.stream.Master.__init__(self)

		self.low_bin      = int(low_bin)
		self.high_bin     = int(high_bin)
		self.scale        = int(scale)
		self.group_frames = max(1, int(group_frames))

		# The acculmulated situation; 
		self._acc        = np.zeros((self.BY, self.BX), dtype=np.uint32)
		self._acc_frames = 0
		self._orig32     = None  # The 122keV does not care about the direction of the movement; 

		# Update the thresholds 
		self._update_thresholds()

	def _update_thresholds(self):
		self._lo = self.low_bin  * self.scale
		self._hi = self.high_bin * self.scale
		if self._hi < self._lo:
			self._hi = self._lo

	def _emit_group(self):
		# Group some of the frames together and then output ; s
		if self._acc_frames == 0 or self._orig32 is None:
			return

		out_len = self.HEAD_IN + self.NBLOCK * 4  # 32B + 2112*u32
		out_buf = bytearray(out_len)

		# Copy the original header 
		out_buf[:self.HEAD_IN] = self._orig32

		# 写 uint32 结果（LE）
		mv = memoryview(out_buf)
		np.frombuffer(mv[self.HEAD_IN:], dtype='<u4', count=self.NBLOCK)[:] = self._acc.reshape(-1)

		# Sent
		f = self._reqFrame(out_len, True)
		f.write(out_buf, 0)
		self._sendFrame(f)

		# Reset the data; 
		self._acc.fill(0)
		self._acc_frames = 0
		self._orig32 = None

	def _acceptFrame(self, frame):
		size = frame.getPayload()
		if size < self.HEAD_IN + self.DATA_IN:
			return  # 丢弃非完整帧

		buf = bytearray(size)
		frame.read(buf, 0)

		# 32B 头 + 图像（u16 LE）
		if self._acc_frames == 0:
			self._orig32 = bytes(buf[:self.HEAD_IN])
		img = np.frombuffer(buf, dtype=np.dtype('<u2'),
							count=self.NPIX, offset=self.HEAD_IN
						   ).reshape(self.NY, self.NX)

		# 阈值筛选（闭开区间 [lo, hi)）
		# 注：img 为 u16，无需拷贝
		m = (img >= self._lo) & (img < self._hi)

		# 8×8 计数：reshape 为 (22,8,96,8) 在小块内求和 -> (22,96)
		counts = m.reshape(self.BY, 8, self.BX, 8).sum(axis=(1, 3)).astype(np.uint32, copy=False)

		# 组内累计
		self._acc += counts
		self._acc_frames += 1

		# 满组则输出
		if self._acc_frames >= self.group_frames:
			self._emit_group()

	# 在退出/停止前调用，输出最后一组
	def flush(self):
		self._emit_group()
