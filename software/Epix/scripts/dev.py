#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : ePix 10ka board instance
#-----------------------------------------------------------------------------
# File       : epix10kaDAQ.py evolved from evalBoard.py
# Author     : Ryan Herbst, rherbst@slac.stanford.edu
# Modified by: Dionisio Doering
# Created    : 2016-09-29
# Last update: 2017-02-01
#-----------------------------------------------------------------------------
# Description:
# Rogue interface to ePix 10ka board
#-----------------------------------------------------------------------------
# This file is part of the rogue_example software. It is subject to 
# the license terms in the LICENSE.txt file found in the top-level directory 
# of this distribution and at: 
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
# No part of the rogue_example software, including this file, may be 
# copied, modified, propagated, or distributed except according to the terms 
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import setupLibPaths
import rogue.hardware.pgp
import pyrogue as pr
import pyrogue.utilities.prbs
import pyrogue.utilities.fileio
import pyrogue.interfaces.simulation
import pyrogue.gui
import surf
import surf.axi
import surf.protocols.ssi
import threading
import signal
import atexit
import yaml
import time
import sys
import testBridge
import ePixViewer as vi
import ePixFpga as fpga
import argparse


# L0 Processing: subtract per-pixel dark (no wrap-around, clamp to [0, 65535])
import numpy as np
import rogue.interfaces.stream

import numpy as np
import rogue.interfaces.stream  # 不用别名

import numpy as np
import rogue.interfaces.stream  # 不用别名

class L0Process(rogue.interfaces.stream.Slave, rogue.interfaces.stream.Master):
    """VC0帧级处理：raw-dark -> 帧级CM(全帧) -> 列CM(阈值=4*n1+mean(raw-dark)) -> 裁剪 -> 回写<uint16>。"""

    HEAD_LEN  = 40
    NY        = 176
    NX        = 768
    U16_COUNT = NY * NX
    DATA_LEN  = U16_COUNT * 2  # bytes

    def __init__(self,
                 dark_path="/data/epix/software/Mossbauer/dark_2D.npy",
                 n0=8,                    # 帧级CM的阈值常数：4*n0
                 n1=8,                    # 列CM的阈值常数部分：4*n1
                 enable_common_mode=True,
                 clamp_min=0, clamp_max=0xFFFF):
        rogue.interfaces.stream.Slave.__init__(self)
        rogue.interfaces.stream.Master.__init__(self)

        # 读取 dark_2D 并展平（期望 176x768）
        dark = np.load(dark_path, mmap_mode='r')
        flat = np.asarray(dark, order='C').reshape(-1)
        if flat.size != self.U16_COUNT:
            raise ValueError(f"dark_2D size mismatch: got {flat.size} pixels, need {self.U16_COUNT}")

        # 预分配/常驻缓冲
        self.dark_i32 = flat.astype(np.int32, copy=False)
        self.work_i32 = np.empty(self.U16_COUNT, dtype=np.int32)
        self.work_2d  = self.work_i32.reshape(self.NY, self.NX)
        self.mask     = np.empty(self.U16_COUNT, dtype=bool)
        self.mask_2d  = self.mask.reshape(self.NY, self.NX)
        self.col_med  = np.empty(self.NX, dtype=np.int32)

        # 参数
        self.n0 = int(n0)   # 帧级CM
        self.n1 = int(n1)   # 列CM
        self.enable_common_mode = bool(enable_common_mode)
        self.clamp_min = int(clamp_min)
        self.clamp_max = int(clamp_max)

    # —— 帧级 common-mode：在 raw-dark 的整帧上，用 < 4*n0 的像素的中位数作为全帧偏置 —— 
    def _frame_common_mode(self, cnt: int) -> None:
        if cnt != self.U16_COUNT:
            return
        w = self.work_i32  # 长度 U16_COUNT
        # 选出“低于 4*n0”的像素
        np.less(w, 4 * self.n0, out=self.mask)
        if np.any(self.mask):
            m0 = np.median(w[self.mask])
            # 原地扣掉全帧偏置
            w -= int(m0)

    # —— 列方向 common-mode：在 raw-dark 空间，用 w<thr 的像素参与列中位数 —— 
    def _col_common_mode(self, cnt: int, thr: int) -> None:
        if not self.enable_common_mode or cnt != self.U16_COUNT:
            return
        f2d = self.work_2d
        # 仅 raw-dark < thr 的像素参与列中位数
        np.less(f2d, thr, out=self.mask_2d)
        ma = np.ma.array(f2d, mask=~self.mask_2d)
        m  = np.ma.median(ma, axis=0)
        if isinstance(m, np.ma.MaskedArray):
            m = m.filled(0)
        self.col_med[:] = np.asarray(m, dtype=np.int32)
        # 按列相减（原地）
        f2d -= self.col_med

    # —— Rogue 回调：每帧处理 —— 
    def _acceptFrame(self, frame):
        size = frame.getPayload()
        buf  = bytearray(size)
        frame.read(buf, 0)

        # 仅处理 [40, 40+NY*NX*2) 的有效区
        valid_bytes = min(self.DATA_LEN, max(0, size - self.HEAD_LEN))
        cnt = valid_bytes // 2
        if cnt > 0:
            # 原始有效区（<u2, little-endian）
            arr_u2 = np.frombuffer(buf, dtype=np.dtype('<u2'),
                                   count=cnt, offset=self.HEAD_LEN)

            # raw -> i32 工作区
            w = self.work_i32[:cnt]
            w[:] = arr_u2
            np.subtract(w, self.dark_i32[:cnt], out=w, casting='unsafe')

            # 帧级CM（与 epix.py 一致）
            self._frame_common_mode(cnt)

            # 动态阈值（列CM用）：4*n1 + mean(raw-dark-after-frameCM)
            mean_i32 = int(np.add.reduce(w, dtype=np.int64) // cnt)
            thr = 4 * self.n1 + mean_i32
            if   thr < 0:       thr = 0
            elif thr > 0xFFFF:  thr = 0xFFFF

            # 列CM（完全在 raw-dark 空间里选点/算中位数/扣偏置）
            self._col_common_mode(cnt, thr)

            # 裁剪并写回 <u2
            np.clip(w, self.clamp_min, self.clamp_max, out=w)
            arr_u2[:] = w

        # 送出帧
        out = self._reqFrame(size, True)
        out.write(buf, 0)
        self._sendFrame(out)


try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore    import *
    from PyQt5.QtGui     import *
except ImportError:
    from PyQt4.QtCore    import *
    from PyQt4.QtGui     import *

# Set the argument parser
parser = argparse.ArgumentParser()

# Convert str to bool
argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

# Add arguments
parser.add_argument(
    "--pollEn", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "Enable auto-polling",
) 

parser.add_argument(
    "--initRead", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "Enable read all variables at start",
)  

parser.add_argument(
    "--viewer", 
    type     = argBool,
    required = False,
    default  = True,
    help     = "Start viewer",
)  

parser.add_argument(
    "--gui", 
    type     = argBool,
    required = False,
    default  = True,
    help     = "Start control GUI",
)  


parser.add_argument(
    "--pgp", 
    type     = str,
    required = False,
    default  = '/dev/datadev_0',
    help     = "PGP device (default /dev/datadev_0)",
)  

parser.add_argument(
    "--verbose", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "Print debug info",
)  

parser.add_argument(
    "--simulation", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "Connect to VCS simulation",
)  

parser.add_argument(
    "--asic_rev", 
    type     = int,
    required = False,
    default  = 1,
    help     = "ASIC rev 1 or 2",
)


# Get the arguments
args = parser.parse_args()

#############################################
# Define if the GUI is started (1 starts it)
START_GUI = args.gui
START_VIEWER = args.viewer
#############################################
#print debug info
PRINT_VERBOSE = args.verbose
#############################################


# Create the PGP interfaces for ePix camera
if args.simulation:
   pgpVc1 = rogue.interfaces.stream.TcpClient('localhost',8000)
   pgpVc0 = rogue.interfaces.stream.TcpClient('localhost',8002)
   pgpVc2 = rogue.interfaces.stream.TcpClient('localhost',8004)
   pgpVc3 = rogue.interfaces.stream.TcpClient('localhost',8006)
else:
   pgpVc1 = rogue.hardware.pgp.PgpCard(args.pgp,0,0) # Registers 
   pgpVc0 = rogue.hardware.pgp.PgpCard(args.pgp,0,1) # Data
   pgpVc2 = rogue.hardware.pgp.PgpCard(args.pgp,0,2) # PseudoScope
   pgpVc3 = rogue.hardware.pgp.PgpCard(args.pgp,0,3) # Monitoring (Slow ADC)
   print("")
   print("PGP Card Version: %x" % (pgpVc0.getInfo().version))


# Add data stream to file as channel 1
# File writer
dataWriter = pyrogue.utilities.fileio.StreamWriter(name = 'dataWriter')
#pyrogue.streamConnect(pgpVc0, dataWriter.getChannel(0x1))
l0 = L0Process(dark_path="/data/epix/software/Mossbauer/dark_2D.npy",
               n1=8, enable_common_mode=True)
pyrogue.streamConnect(pgpVc0, l0)
pyrogue.streamConnect(l0, dataWriter.getChannel(0x1))

# Add pseudoscope to file writer
pyrogue.streamConnect(pgpVc2, dataWriter.getChannel(0x2))
pyrogue.streamConnect(pgpVc3, dataWriter.getChannel(0x3))

cmd = rogue.protocols.srp.Cmd()
pyrogue.streamConnect(cmd, pgpVc0)

# Create and Connect SRP to VC1 to send commands
srp = rogue.protocols.srp.SrpV0()
pyrogue.streamConnectBiDir(pgpVc1,srp)

# Add configuration stream to file as channel 0
# Removed to reduce amount of data going to file
#pyrogue.streamConnect(ePixBoard,dataWriter.getChannel(0x0))

## Add microblaze console stream to file as channel 2
#pyrogue.streamConnect(pgpVc3,dataWriter.getChannel(0x2))

# PRBS Receiver as secdonary receiver for VC1
#prbsRx = pyrogue.utilities.prbs.PrbsRx('prbsRx')
#pyrogue.streamTap(pgpVc1,prbsRx)
#ePixBoard.add(prbsRx)

# Microblaze console monitor add secondary tap
#mbcon = MbDebug()
#pyrogue.streamTap(pgpVc3,mbcon)

#br = testBridge.Bridge()
#br._setSlave(srp)

#ePixBoard.add(surf.SsiPrbsTx.create(memBase=srp1,offset=0x00000000*4))

# Create epics node
#epics = pyrogue.epics.EpicsCaServer('rogueTest',ePixBoard)
#epics.start()



#############################################
# Microblaze console printout
#############################################
class MbDebug(rogue.interfaces.stream.Slave):

    def __init__(self):
        rogue.interfaces.stream.Slave.__init__(self)
        self.enable = False

    def _acceptFrame(self,frame):
        if self.enable:
            p = bytearray(frame.getPayload())
            frame.read(p,0)
            print('-------- Microblaze Console --------')
            print(p.decode('utf-8'))

#######################################
# Custom run control
#######################################
class MyRunControl(pyrogue.RunControl):
    def __init__(self,name):
        pyrogue.RunControl.__init__(self,name, description='Run Controller ePix 10ka',  rates={1:'1 Hz', 2:'2 Hz', 4:'4 Hz', 8:'8 Hz', 10:'10 Hz', 30:'30 Hz', 60:'60 Hz', 120:'120 Hz'})
        self._thread = None

    def _setRunState(self,dev,var,value,changed):
        if changed: 
            if self.runState.get(read=False) == 'Running': 
                self._thread = threading.Thread(target=self._run) 
                self._thread.start() 
            else: 
                self._thread.join() 
                self._thread = None 


    def _run(self):
        self.runCount.set(0) 
        self._last = int(time.time()) 
 
 
        while (self.runState.value() == 'Running'): 
            delay = 1.0 / ({value: key for key,value in self.runRate.enum.items()}[self._runRate]) 
            time.sleep(delay) 
            self._root.ssiPrbsTx.oneShot() 
  
            self._runCount += 1 
            if self._last != int(time.time()): 
                self._last = int(time.time()) 
                self.runCount._updated() 


            
##############################
# Set base
##############################
class EpixBoard(pyrogue.Root):
    def __init__(self, guiTop, cmd, dataWriter, srp, asic_rev, **kwargs):
        super().__init__(name = 'ePixBoard',description = 'ePix 10ka Board', **kwargs)
        #self.add(MyRunControl('runControl'))
        self.add(dataWriter)
        self.guiTop = guiTop

        @self.command()
        def Trigger():
            cmd.sendCmd(0, 0)

        # Add Devices
        self.add(fpga.Epix10ka(name='Epix10ka', asic_rev=asic_rev, offset=0, memBase=srp, hidden=False, enabled=True))
        self.add(pyrogue.RunControl(name = 'runControl', description='Run Controller ePix 10ka', cmd=self.Trigger, rates={1:'1 Hz', 2:'2 Hz', 4:'4 Hz', 8:'8 Hz', 10:'10 Hz', 30:'30 Hz', 60:'60 Hz', 120:'120 Hz'}))
        

        


# debug
#mbcon = MbDebug()
#pyrogue.streamTap(pgpVc0,mbcon)

#mbcon1 = MbDebug()
#pyrogue.streamTap(pgpVc1,mbcon)

#mbcon2 = MbDebug()
#pyrogue.streamTap(pgpVc3,mbcon)

if (PRINT_VERBOSE): dbgData = rogue.interfaces.stream.Slave()
if (PRINT_VERBOSE): dbgData.setDebug(60, "DATA[{}]".format(0))
if (PRINT_VERBOSE): pyrogue.streamTap(pgpVc0, dbgData)


# Create GUI
appTop = QApplication(sys.argv)
guiTop = pyrogue.gui.GuiTop(group = 'ePix10kaGui')
ePixBoard = EpixBoard(guiTop, cmd, dataWriter, srp, args.asic_rev)
ePixBoard.start()
guiTop.addTree(ePixBoard)
guiTop.resize(1000,800)

# Viewer gui
if START_VIEWER:
   gui = vi.Window(cameraType = 'ePix10ka')
   gui.eventReader.frameIndex = 0
   #gui.eventReaderImage.VIEW_DATA_CHANNEL_ID = 0
   gui.setReadDelay(0)
   pyrogue.streamTap(pgpVc0, gui.eventReader) 
   pyrogue.streamTap(pgpVc2, gui.eventReaderScope)# PseudoScope
   pyrogue.streamTap(pgpVc3, gui.eventReaderMonitoring) # Slow Monitoring

# Create mesh node (this is for remote control only, no data is shared with this)
#mNode = pyrogue.mesh.MeshNode('rogueTest',iface='eth0',root=ePixBoard)
#mNode = pyrogue.mesh.MeshNode('rogueEpix10ka',iface='eth0',root=None)
#mNode.setNewTreeCb(guiTop.addTree)
#mNode.start()


# Run gui
if (START_GUI):
    appTop.exec_()

# Close window and stop polling
def stop():
    mNode.stop()
#    epics.stop()
    ePixBoard.stop()
    exit()

# Start with: ipython -i scripts/epix10kaDAQ.py for interactive approach
print("Started rogue mesh and epics V3 server. To exit type stop()")

