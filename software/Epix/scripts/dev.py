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

from L0Process import L0Process
from L1Process import L1Process
from L2Spectrum import L2Spectrum
from L1BitmaskCompressor import L1BitmaskCompressor
from StreamSampler import StreamSampler

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
    default  = '../yml/epix10ka_mossbauer_500Hz.yml',
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
l0 = L0Process(dark_path="/data/epix/software/Mossbauer/dark_2D.npy",filter_path="/data/epix/software/Mossbauer/filter.npy",
               n1=8, enable_common_mode=True)
#l1 = L1Process(gain_path="/data/epix/software/Mossbauer/new_gain.npy")
#l1 = L1Process(gain_scalar=17)
l1 = L1Process(gain_path="/data/epix/software/Mossbauer/gain.npy")

# Main Data Stream
pyrogue.streamConnect(pgpVc0, l0)
pyrogue.streamConnect(l0,l1)
pyrogue.streamConnect(l1, dataWriter.getChannel(0x1))


# Parallel Writing; 
# Create the Writer for sampling; 
rawWriter = pyrogue.utilities.fileio.StreamWriter(name='rawWriter')
L0Writer = pyrogue.utilities.fileio.StreamWriter(name='L0Writer')
L1Writer= pyrogue.utilities.fileio.StreamWriter(name='L1Writer') 
S2Writer = pyrogue.utilities.fileio.StreamWriter(name='S2Writer')

# Additional Channels for data writing; 
# Sampler for raw data; 
sampler = StreamSampler(min_interval=1.0)
pyrogue.streamTap(pgpVc0,sampler)
pyrogue.streamConnect(sampler,rawWriter.getChannel(0x1))
# Sampler for L0 data; 
L0sampler = StreamSampler(min_interval=1.0)
pyrogue.streamTap(l0,L0sampler)
pyrogue.streamConnect(L0sampler,L0Writer.getChannel(0x1))
# All information Preserve; 
l1bm = L1BitmaskCompressor(threshold=50, emit_empty=False)
pyrogue.streamTap(l0, l1bm)
pyrogue.streamConnect(l1bm, L1Writer.getChannel(0x1))
# Spectrum
specTap = L2Spectrum(every_n=10)  # 每10帧输出一次
pyrogue.streamTap(l1, specTap)                         # 从 L1 旁路
pyrogue.streamConnect(specTap, S2Writer.getChannel(0x6))

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
raw_path = Board_utils.make_data_path("/data/raw/")
data_path = Board_utils.make_data_path("/data/")
L0_path = Board_utils.make_data_path("/data/L0/")
L1_path = Board_utils.make_data_path("/data/L1/")
S2_path = Board_utils.make_data_path("/data/S2/")


# Create Gui
# The command is the software trigger system; 
appTop = QApplication(sys.argv)
guiTop = pyrogue.gui.GuiTop(group = 'ePix10kaGui')
ePixBoard = EpixBoard(guiTop, cmd, dataWriter, srp, args.asic_rev)

# Add Raw Writer and L0 Writer to the board for sampling;
ePixBoard.add(rawWriter)
ePixBoard.add(L0Writer)
ePixBoard.add(L1Writer)
ePixBoard.add(S2Writer)
ePixBoard.start(serverPort=9099)

# Load the mossbauer yaml file; 
ePixBoard.LoadConfig(args.yml)
time.sleep(0.2)
ePixBoard.LoadConfig(args.yml)
time.sleep(0.2)

#Data Path;
ePixBoard.dataWriter.DataFile.set(data_path)
ePixBoard.dataWriter._writer.setMaxSize(5*1024 * 1024**2)
ePixBoard.dataWriter.Open.set(True) 

# Enable the parallel raw record 
ePixBoard.rawWriter.DataFile.set(raw_path)
ePixBoard.rawWriter._writer.setMaxSize(500 * 1024**2)
ePixBoard.rawWriter.Open.set(True) 
rawWriter._writer.open(raw_path)
# Enable the Processed L0 record 
ePixBoard.L0Writer.DataFile.set(L0_path)
ePixBoard.L0Writer._writer.setMaxSize(500 * 1024**2)
ePixBoard.L0Writer.Open.set(True) 
L0Writer._writer.open(L0_path)
# Enable the Bitmask L1 compressor
ePixBoard.L1Writer.DataFile.set(L1_path)
ePixBoard.L1Writer._writer.setMaxSize(5*1024 * 1024**2)
ePixBoard.L1Writer.Open.set(True) 
L1Writer._writer.open(L1_path)
# S2 Writer
ePixBoard.S2Writer.DataFile.set(L1_path)
ePixBoard.S2Writer._writer.setMaxSize(500 * 1024**2)
ePixBoard.S2Writer.Open.set(True) 
S2Writer._writer.open(S2_path)

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

# Close window and stop polling
def stop():
    mNode.stop()
#    epics.stop()
    ePixBoard.stop()
    exit()

