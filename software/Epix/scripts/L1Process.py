# L1Gain.py
import numpy as np
import rogue.interfaces.stream  

class L1Process(rogue.interfaces.stream.Slave,
			 rogue.interfaces.stream.Master):

	HEAD_LEN  = 32                 # The header length is 32
	NY, NX    = 176, 768
	U16_COUNT = NY * NX
	DATA_LEN  = U16_COUNT * 2

	def __init__(self,
				 gain_path=None,        # (176,768) 的 float32/64；
				 gain_scalar=None,      
				 scale=256, 
				 clamp_min=0, clamp_max=0xFFFF,
				 round_mode='nearest'   # 'nearest' | 'floor' | 'ceil' | 'none'
				 ):
		rogue.interfaces.stream.Slave.__init__(self)
		rogue.interfaces.stream.Master.__init__(self)

		
		self.coeff = None
		self.coeff_scalar = None
		self.SCALE = scale

		# Calculate the gain
		if gain_path is not None:
			g = np.load(gain_path, mmap_mode='r')
			g = np.asarray(g, order='C')
			if g.size != self.U16_COUNT:
				raise ValueError(f"gain size {g.size} != {self.U16_COUNT}")
			g = g.reshape(self.NY, self.NX).astype(np.float32, copy=False)
			self.coeff = (self.SCALE / np.maximum(g, 1e-12)).astype(np.float32, copy=False)
		elif gain_scalar is not None:
			gs = float(gain_scalar)
			if not np.isfinite(gs) or gs <= 0:
				raise ValueError(f"invalid gain_scalar={gain_scalar}")
			self.coeff_scalar = np.float32(self.SCALE / gs)
		else:
			
			self.coeff_scalar = self.SCALE/17

		# Work buffer
		self.work_f32 = np.empty((self.NY, self.NX), dtype=np.float32)

		self.clamp_min = int(clamp_min)
		self.clamp_max = int(clamp_max)
		self.round_mode = str(round_mode).lower()

	def _acceptFrame(self, frame):
		size = frame.getPayload()
		if size < self.HEAD_LEN + self.DATA_LEN:
			return  

		buf = bytearray(size)
		frame.read(buf, 0)

		arr_u2 = np.frombuffer(buf, dtype=np.dtype('<u2'),
							   count=self.U16_COUNT, offset=self.HEAD_LEN
							  ).reshape(self.NY, self.NX)

		# U16 -> F32
		self.work_f32[:] = arr_u2

		if self.coeff is not None:
			np.multiply(self.work_f32, self.coeff, out=self.work_f32)
		else:
			self.work_f32 *= self.coeff_scalar

		# Clip
		np.clip(self.work_f32, self.clamp_min, self.clamp_max, out=self.work_f32)

		# Round
		if self.round_mode == 'nearest':
			np.rint(self.work_f32, out=self.work_f32)
		elif self.round_mode == 'floor':
			np.floor(self.work_f32, out=self.work_f32)
		elif self.round_mode == 'ceil':
			np.ceil(self.work_f32, out=self.work_f32)

		arr_u2[:, :] = self.work_f32.astype(np.uint16, copy=False)

		out = self._reqFrame(size, True)
		out.write(buf, 0)
		self._sendFrame(out)
