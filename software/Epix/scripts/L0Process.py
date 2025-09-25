import numpy as np
import rogue.interfaces.stream  

class L0Process(rogue.interfaces.stream.Slave, rogue.interfaces.stream.Master):
	"""
	Level 0 Data Processing: Dark Data Substraction , dynamic common mode correction, bad pixels filter;
	"""

	HEAD_LEN  = 32
	NY        = 176
	NX        = 768
	U16_COUNT = NY * NX                  # 135,168
	DATA_LEN  = U16_COUNT * 2            # 270,336 bytes

	def __init__(self,
				 dark_path="/data/epix/software/Mossbauer/dark_2D.npy",
				 filter_path="/data/epix/software/Mossbauer/filter.npy",
				 n1=8,                          
				 enable_common_mode=True,
				 clamp_min=0, clamp_max=0xFFFF):
		# Initialize 
		rogue.interfaces.stream.Slave.__init__(self)
		rogue.interfaces.stream.Master.__init__(self)

		# Get the dark readout; 
		dark = np.load(dark_path, mmap_mode='r')
		dark = np.asarray(dark, order='C')
		if dark.size != self.U16_COUNT:
			raise ValueError(f"dark size {dark.size} != {self.U16_COUNT}")
		self.dark_i32 = dark.reshape(self.NY, self.NX).astype(np.int32, copy=False)


		# Bad pixels filter
		self.bad_mask = None
		if filter_path is not None:
			filt = np.load(filter_path, mmap_mode='r')
			filt = np.asarray(filt, order='C')
			if filt.size != self.U16_COUNT:
				raise ValueError(f"filter size {filt.size} != {self.U16_COUNT}")
			filt2d = filt.reshape(self.NY, self.NX)
			self.bad_mask = (filt2d == 0)

		# Preload Workzone 
		self.work_i32 = np.empty((self.NY, self.NX), dtype=np.int32)
		self.mask_2d  = np.empty((self.NY, self.NX), dtype=bool)
		self.col_med  = np.empty(self.NX, dtype=np.int32)

		# Parameters;
		self.n1 = int(n1)
		self.enable_common_mode = bool(enable_common_mode)
		self.clamp_min = int(clamp_min)
		self.clamp_max = int(clamp_max)

	def _col_common_mode(self, thr: int) -> None:
		"""
		Common Mode correction after dark readout substraction; 
		"""
		if not self.enable_common_mode:
			return
		w2d = self.work_i32

		# Select the points without readout; 
		np.less(w2d, thr, out=self.mask_2d)                   
		ma = np.ma.array(w2d, mask=~self.mask_2d)
		m  = np.ma.median(ma, axis=0)
		if isinstance(m, np.ma.MaskedArray):
			m = m.filled(0)

		# Substract the column common mode noise; 
		self.col_med[:] = np.asarray(m, dtype=np.int32)
		w2d -= self.col_med

	def _acceptFrame(self, frame):
		size = frame.getPayload()
		buf  = bytearray(size)
		frame.read(buf, 0)

		# Only process the valid bytes; 
		data_bytes = max(0, size - self.HEAD_LEN)
		valid_bytes = min(self.DATA_LEN, data_bytes)
		if valid_bytes < self.DATA_LEN:
			# Wrong Frames, just send it. 
			out = self._reqFrame(size, True)
			out.write(buf, 0)
			self._sendFrame(out)
			return

		# Raw valid data;
		arr_u2 = np.frombuffer(
			buf, dtype=np.dtype('<u2'),
			count=self.U16_COUNT, offset=self.HEAD_LEN
		).reshape(self.NY, self.NX)

		# raw -> i32：raw - dark, dark
		np.subtract(arr_u2, self.dark_i32, out=self.work_i32, casting='unsafe')

		# Dynamic Threshold：4*n1 + mean(raw-dark)
		sum_i64  = self.work_i32.sum(dtype=np.int64)          # Sum to calculate the mean
		mean_i32 = int(sum_i64 // self.U16_COUNT)
		thr      = 4 * self.n1 + mean_i32
		if   thr < 0:       thr = 0
		elif thr > 0xFFFF:  thr = 0xFFFF                      # Threshold Range;

		# Column Common Mode Correction
		self._col_common_mode(thr)
		
		# Bad Filters ; 
		if self.bad_mask is not None:
			self.work_i32[self.bad_mask] = 0

		# Clip
		np.clip(self.work_i32, self.clamp_min, self.clamp_max, out=self.work_i32)
		arr_u2[:, :] = self.work_i32.astype(np.uint16, copy=False)

		# Send the frames
		out = self._reqFrame(size, True)
		out.write(buf, 0)
		self._sendFrame(out)
