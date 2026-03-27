#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : ePix 10ka board instance
#-----------------------------------------------------------------------------
# File       : epix10kaDAQ.py evolved from evalBoard.py
# Author     : Ryan Herbst, rherbst@slac.stanford.edu
# Modified by: Chengjie Jia
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
import os 
import yaml
import time
import sys
import testBridge
import ePixViewer as vi
import ePixFpga as fpga
import argparse

import rogue.interfaces.stream
import time
import multiprocessing as mp

from L0Process import L0Process
from L1Process import L1Process
from L2Spectrum import L2Spectrum
from L2Process import L2Process
from L1BitmaskCompressor import L1BitmaskCompressor
from L2Para import L2Para
from StreamSampler import StreamSampler
from L3Process import L3Process
from L00Timestamp import TimestampProcess

from Board_utils import EpixBoard,MyRunControl,MbDebug
import Board_utils 

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
    "--yml", 
    type     = str,
    required = False,
    default  = '../yml/epix10ka_mossbauer_auto.yml',
    help     = "Default yml is the mossbauer 500Hz",
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
#The main stream processing; 
scale=256
l0 = L0Process(dark_path="/data/dark/dark_2D.npy",
               filter_path="/data/dark/filter.npy",
               enable_common_mode=True,
               # Initial Setup;
               
               # Dynamic Calib(Background and filter1 )
               dynamic_calib=True,
               dynamic_calib_dir="/data/dark",
               dynamic_filter_dir="/data/dark",
               dynamic_calib_period_s=3600,
               
               drop_if_busy=True,
               # Common mode noise correction;
               n1=8)
l1 = L1Process(gain_path="/data/epix/software/Mossbauer/gain.npy",
               # Dynamic Gain
               dynamic_gain=True,
               dynamic_gain_dir="/data/gain",
               dynamic_gain_period_s=86400,
               
               # Centroid Filter;
               enable_centroid=True,
               n2=15,
               numba_threads=None,
               
               drop_if_busy=True,
               # Gain Scale
               scale=scale
               )

l2 = L2Process(low_bin= 12.5, 
               high_bin= 15.9,
               scale=scale
               )

l3 = L3Process(compression_ratio=200)


# A small layer just to add the timestamp;
ts0 = TimestampProcess()

# Main Datastream
pyrogue.streamConnect(pgpVc0, ts0)
pyrogue.streamConnect(ts0, l0)
pyrogue.streamConnect(l0,l1)
pyrogue.streamConnect(l1,l2)
pyrogue.streamConnect(l2,l3)
pyrogue.streamConnect(l3, dataWriter.getChannel(0x1))

# Parallel Writing; 
# Create the Writer for sampling; 
rawWriter = pyrogue.utilities.fileio.StreamWriter(name='rawWriter',hidden=True)
L0Writer = pyrogue.utilities.fileio.StreamWriter(name='L0Writer',hidden=True)
L1Writer= pyrogue.utilities.fileio.StreamWriter(name='L1Writer',hidden=True) 
S2Writer = pyrogue.utilities.fileio.StreamWriter(name='S2Writer',hidden=True)
L2PWriter = pyrogue.utilities.fileio.StreamWriter(name='L2PWriter',hidden=True)

# Additional Channels for data writing; 
# Sampler for raw data; 
sampler = StreamSampler(min_interval=0.25)
pyrogue.streamTap(ts0,sampler)
pyrogue.streamConnect(sampler,rawWriter.getChannel(0x1))
# Sampler for L0 data; 
L0sampler = StreamSampler(min_interval=1.0)
pyrogue.streamTap(l0,L0sampler)
pyrogue.streamConnect(L0sampler,L0Writer.getChannel(0x1))
# All information Preserve; L1; 
l1bm = L1BitmaskCompressor(threshold=50, emit_empty=False)
pyrogue.streamTap(l0, l1bm)
pyrogue.streamConnect(l1bm, L1Writer.getChannel(0x1))

# S2, Spectrum
specTap = L2Spectrum(every_n=100)  
pyrogue.streamTap(l1, specTap)                         
pyrogue.streamConnect(specTap, S2Writer.getChannel(0x1))

# L2 Para for 122keV, each frame; 
L2PTap = L2Para(low_bin= 100,high_bin= 140,scale=scale,group_frames=200)
pyrogue.streamTap(l1,L2PTap)
pyrogue.streamConnect(L2PTap, L2PWriter.getChannel(0x1))


# Add pseudoscope to file writer
pyrogue.streamConnect(pgpVc2, dataWriter.getChannel(0x2))
pyrogue.streamConnect(pgpVc3, dataWriter.getChannel(0x3))
# Software
cmd = rogue.protocols.srp.Cmd()
pyrogue.streamConnect(cmd, pgpVc0)

# Create and Connect SRP to VC1 to send commands
srp = rogue.protocols.srp.SrpV0()
pyrogue.streamConnectBiDir(pgpVc1,srp)

# The debug function is just to output the head of data;  
if (PRINT_VERBOSE): dbgData = rogue.interfaces.stream.Slave()
if (PRINT_VERBOSE): dbgData.setDebug(60, "DATA[{}]".format(0))
if (PRINT_VERBOSE): pyrogue.streamTap(pgpVc0, dbgData)



# Create the automatic data path for the raw data, sample data and the real data; 
raw_path = Board_utils.make_data_path("/data/raw/",base_name='raw')
# The processed is stored at the /data/share that everyone could access; 
data_path = Board_utils.make_data_path("/data/share",base_name='data')
L0_path = Board_utils.make_data_path("/data/L0/",base_name='L0')
L1_path = Board_utils.make_data_path("/data/L1/",base_name='L1')
S2_path = Board_utils.make_data_path("/data/S2/",base_name='S2')
L2P_path = Board_utils.make_data_path("/data/L2P/",base_name='L2P')


# Create Gui
# The command is the software trigger system; 
appTop = QApplication(sys.argv)
guiTop = pyrogue.gui.GuiTop(group = 'ePix10kaGui')
ePixBoard = EpixBoard(guiTop, cmd, dataWriter, srp, args.asic_rev)

# Add Raw Writer and L0 Writer to the board for sampling;
#ePixBoard.add(rawWriter)
#ePixBoard.add(L0Writer)
ePixBoard.add(L1Writer)
ePixBoard.add(S2Writer)
ePixBoard.start()

# Load the mossbauer yaml file; 
ePixBoard.LoadConfig(args.yml)
time.sleep(0.2)
ePixBoard.LoadConfig(args.yml)
time.sleep(0.2)

interval=3600 
# Each Hour, We have a datafile; 



#Data Path;
# Each frame is 16936=44*48*4*2+40 Bytes, Compression Ratio 200 (121MB)
ePixBoard.dataWriter.DataFile.set(data_path)
ePixBoard.dataWriter._writer.setMaxSize(16936*interval*2)
ePixBoard.dataWriter.Open.set(True)
dataWriter._writer.open(data_path) 

# Enable the parallel raw record 
#ePixBoard.rawWriter.DataFile.set(raw_path)
#ePixBoard.rawWriter._writer.setMaxSize(500 * 1024**2)
#ePixBoard.rawWriter.Open.set(True) 

# The Raw frame is 274996 Bytes  178*192*2*4 + 40 + 1538 Bytes; 
# The raw data is taken at 4Hz;
rawWriter._writer.setMaxSize(274996*interval*4)
rawWriter._writer.open(raw_path)


# The Processed frame is 270376 Bytes  176*192*2*4 + 40 Bytes; 
# Enable the Processed L0 record 
#ePixBoard.L0Writer.DataFile.set(L0_path)
#ePixBoard.L0Writer._writer.setMaxSize(500 * 1024**2)
#ePixBoard.L0Writer.Open.set(True) 
L0Writer._writer.setMaxSize(270376*interval*1)
L0Writer._writer.open(L0_path)


# S2 Writer, The Spectrum of whole frame
# Each frame is 1460*4+40= 5880 , running at 4Hz 
ePixBoard.S2Writer.DataFile.set(S2_path)
ePixBoard.S2Writer._writer.setMaxSize(5880*interval*4)
ePixBoard.S2Writer.Open.set(True) 
S2Writer._writer.open(S2_path)


# L2P Writer 
# Running at 4Hz
L2PWriter._writer.setMaxSize(4264*interval*4)
L2PWriter._writer.open(L2P_path)


# Enable the Bitmask L1 compressor
# This length is not so controllable;
# The current rate is about 10 minute per file , 5GB (8MB/s), 
# This part is the biggest, which requires the elm to do the backup; 
ePixBoard.L1Writer.DataFile.set(L1_path)
ePixBoard.L1Writer._writer.setMaxSize(5*1024 * 1024**2)
ePixBoard.L1Writer.Open.set(True) 
L1Writer._writer.open(L1_path)


# GUI
guiTop.addTree(ePixBoard)
guiTop.resize(500,500)
# Viewer gui
if START_VIEWER:
   gui = vi.Window(cameraType = 'ePix10ka')
   gui.eventReader.frameIndex = 0
   #gui.eventReaderImage.VIEW_DATA_CHANNEL_ID = 0
   gui.setReadDelay(0)
   pyrogue.streamTap(l0, gui.eventReader) 
   pyrogue.streamTap(pgpVc2, gui.eventReaderScope)# PseudoScope
   pyrogue.streamTap(pgpVc3, gui.eventReaderMonitoring) # Slow Monitoring


# Run gui
if (START_GUI):
    appTop.exec_()













# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------
_cleanup_once_lock = threading.Lock()
_cleanup_done = False

def stop(exit_code: int = 0):
    """Stop background threads/processes and hardware cleanly.

    Notes:
      - Do NOT call exit() inside this function (avoid recursion via atexit).
      - Safe to call multiple times.
    """
    global _cleanup_done
    with _cleanup_once_lock:
        if _cleanup_done:
            return
        _cleanup_done = True

    # 1) Stop software processing threads (L0/L1BM etc.)
    for name in ("l1bm", "l0", "l1", "l2", "l3",'ts0'):
        obj = globals().get(name, None)
        if obj is None:
            continue
        try:
            if hasattr(obj, "stop"):
                obj.stop()
        except Exception:
            pass

    # 2) Stop StreamWriters (flush/close if supported)
    for name in ("dataWriter", "rawWriter", "L0Writer", "L1Writer", "S2Writer", "L2PWriter"):
        w = globals().get(name, None)
        if w is None:
            continue
        try:
            if hasattr(w, "close"):
                w.close()
            elif hasattr(w, "stop"):
                w.stop()
        except Exception:
            pass

    # 3) Stop top-level Rogue/PyRogue nodes
    for name in ("mNode", "ePixBoard"):
        node = globals().get(name, None)
        if node is None:
            continue
        try:
            if hasattr(node, "stop"):
                node.stop()
        except Exception:
            pass

    # If a GUI is running, request quit (safe even if not running)
    try:
        if globals().get("START_GUI", False):
            # appTop is a QApplication/QCoreApplication wrapper in your stack
            app = globals().get("appTop", None)
            if app is not None and hasattr(app, "quit"):
                app.quit()
    except Exception:
        pass

    # Optional: if called from a signal handler, exit here
    if exit_code is not None:
        try:
            sys.exit(exit_code)
        except SystemExit:
            raise

def _sig_handler(signum, frame):
    # Try to clean up, then exit immediately
    try:
        stop(exit_code=0)
    except SystemExit:
        raise
    except Exception:
        try:
            sys.exit(0)
        except SystemExit:
            raise

# Ensure cleanup on normal interpreter exit
atexit.register(lambda: stop(exit_code=None))

# Ensure cleanup on Ctrl+C / SIGTERM
try:
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
except Exception:
    pass