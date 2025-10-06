# L2BlockCounter8x8_U32.py
import numpy as np
import rogue.interfaces.stream  # 不用别名

class L2Para(rogue.interfaces.stream.Slave,
						rogue.interfaces.stream.Master):
	"""
	输入（每帧）:
	  [32B 原头] + [176*768 × u16 (LE)]

	处理:
	  - 对每帧像素，筛选幅值 ∈ [low_bin*scale, high_bin*scale)
	  - 以 8×8 像素为一块（22×96 共 2112 块），统计每块内命中像素个数 (0..64)
	  - 连续累计 group_frames 帧（默认 10 帧），得到每块的 uint32 计数和

	输出（每 group_frames 帧输出一次）:
	  [32B 原头(沿用该组第1帧的原头)] + [2112 × <u4, LE>]

	说明:
	  - 本模块**不附加任何自定义头**；仅保留 32B 原头并紧跟结果数据。
	  - 若流末尾有未满 group_frames 的残留，也可调用 flush() 输出。
	"""

	# 输入/几何
	HEAD_IN   = 32
	NY, NX    = 176, 768
	NPIX      = NY * NX                 # 135,168
	DATA_IN   = NPIX * 2                # 270,336 B

	# 8x8 分块
	BY, BX    = NY // 8, NX // 8        # 22, 96
	NBLOCK    = BY * BX                 # 2,112

	def __init__(self,
				 low_bin: int = 100,
				 high_bin: int = 140,
				 scale: int = 256,
				 group_frames: int = 10):
		rogue.interfaces.stream.Slave.__init__(self)
		rogue.interfaces.stream.Master.__init__(self)

		self.low_bin      = int(low_bin)
		self.high_bin     = int(high_bin)
		self.scale        = int(scale)
		self.group_frames = max(1, int(group_frames))

		# 组内累计状态
		self._acc        = np.zeros((self.BY, self.BX), dtype=np.uint32)
		self._acc_frames = 0
		self._orig32     = None  # 本组第一帧的 32B 原头

		# 预先算好阈值（可在运行时修改成员变量后重算）
		self._update_thresholds()

	def _update_thresholds(self):
		self._lo = self.low_bin  * self.scale
		self._hi = self.high_bin * self.scale
		if self._hi < self._lo:
			self._hi = self._lo

	def _emit_group(self):
		"""把当前累计组打包输出；若当前为空则忽略。"""
		if self._acc_frames == 0 or self._orig32 is None:
			return

		out_len = self.HEAD_IN + self.NBLOCK * 4  # 32B + 2112*u32
		out_buf = bytearray(out_len)

		# 拷贝 32B 原头（不做修改）
		out_buf[:self.HEAD_IN] = self._orig32

		# 写 uint32 结果（LE）
		mv = memoryview(out_buf)
		np.frombuffer(mv[self.HEAD_IN:], dtype='<u4', count=self.NBLOCK)[:] = self._acc.reshape(-1)

		# 发送
		f = self._reqFrame(out_len, True)
		f.write(out_buf, 0)
		self._sendFrame(f)

		# 重置累计
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
