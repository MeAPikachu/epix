# L3StateAggregator_U16.py
import numpy as np
import struct
import rogue.interfaces.stream  # The interface for rogue; 
import time

# The new version of the L3 is not continously collecting data in the same direction;
# Instead, it will keep flipping;
# So we will keep two different buf for forward and backward; 

class L3Process(rogue.interfaces.stream.Slave,
						rogue.interfaces.stream.Master):
	"""
	The L3 Process reduces the time resolution, based on the direction
	of the DAQ input, adds the 
	"""
	HEAD_IN  = 32
	BY, BX   = 44, 192
	BLK_CNT  = BY * BX  # 8448

	def __init__(self,compression_ratio=10):
		rogue.interfaces.stream.Slave.__init__(self)
		rogue.interfaces.stream.Master.__init__(self)

		# The default compression ratio is 10;		
		self.compression_ratio=compression_ratio 

		# The state of the compressed frame; 
		self._state = [None,None ]             
		
		# How many frames have we got; 
		self._frames_acc= [0,0]
		
		# The sum of the total counts; 
		self._acc = [ np.zeros(self.BLK_CNT, dtype=np.uint16) , np.zeros(self.BLK_CNT, dtype=np.uint16) ] 
		
		# The head of the first frames ; 
		self._orig32_seg = [None, None]          



	@staticmethod
	def _state_from_word4(orig32: bytes) -> int:
		word = struct.unpack_from('<I', orig32, 16)[0]
		return 0 if word == 0 else 1


	# This function will send the frame;
	def _emit_segment(self,st=0,orig32=None):
		
		if orig32 is None:
			orig32 = bytes(32)
		if self._state[st] is None or self._frames_acc[st] == 0:
			return

		out_len = self.HEAD_IN + self.BLK_CNT * 2
		out_buf = bytearray(out_len)
		out_buf[:self.HEAD_IN] = self._orig32_seg[st]
		
		# Word 3  The frame count , which should be the compresssion ratio ; 
		struct.pack_into('<I', out_buf, 12, int(self._frames_acc[st]))
		
		# Word 6, the run count of the last frame;
		struct.pack_into('<I', out_buf, 24, struct.unpack_from('<I', orig32, 24)[0] )
		# Ideally, Word6-Word2 should be 2*(compression_ratio -1 )
		
		# Word 5, the microseconds timestamp from the computer ; 
		# Word 7, the seconds timestamp from the computer; 
		ns = time.time_ns()
		second = ns // 1_000_000_000
		microsecond = (ns % 1_000_000_000) // 1_000       
		struct.pack_into('<I', out_buf, 20, microsecond & 0xFFFFFFFF)
		struct.pack_into('<I', out_buf, 28, second & 0xFFFFFFFF)
		

		# 8448×u16 Small End, get the data from the acc variable; 
		mv = memoryview(out_buf)
		np.frombuffer(mv[self.HEAD_IN:], dtype='<u2', count=self.BLK_CNT)[:] = self._acc[st]

		# Send the frame ; 
		f = self._reqFrame(out_len, True)
		f.write(out_buf, 0)
		self._sendFrame(f)

		# Reset the relevant variables; 
		self._acc[st].fill(0)
		self._frames_acc[st] = 0
		self._orig32_seg[st] = None
		self._state[st]= None 

	# To some degree,  the _acceptFrame is the start of everythingl 
	def _acceptFrame(self, frame):
		
		# Fetch a frame and get the direction of it. 
		size = frame.getPayload()
		min_len = self.HEAD_IN + self.BLK_CNT  
		if size < min_len:
			return  
		buf = bytearray(size)
		frame.read(buf, 0)
		orig32 = bytes(buf[:self.HEAD_IN])
		# Readout the direction of the state; 
		st = self._state_from_word4(orig32)

		# If it is the first frame; 
		if self._state[st] is None:
			self._state[st] = st
			self._orig32_seg[st] = orig32

		# Add the new value to the buffer of the correlated array 
		vals_u8 = np.frombuffer(buf, dtype=np.uint8,offset=self.HEAD_IN, count=self.BLK_CNT)
		tmp32 = self._acc[st].astype(np.uint32) + vals_u8.astype(np.uint32)
		np.minimum(tmp32, 0xFFFF, out=tmp32)
		self._acc[st][:] = tmp32.astype(np.uint16)

		# Add the counts by one
		self._frames_acc[st] += 1
		
		# If it is the last frame; Send the data and reset things to zero; 
		if self._frames_acc[st]>= self.compression_ratio: 
			self._emit_segment(st,orig32)


	def flush(self):
		self._emit_segment(st=0)
		self._emit_segment(st=1)
