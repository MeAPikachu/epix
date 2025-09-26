#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Acquire data from epix10ka

:Author: Faisal Abu-Nimeh (abunimeh@slac.stanford.edu)
:License: https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html
:Date: 20170725
:Style: OpenStack Style Guidelines https://docs.openstack.org/developer/hacking/
:vcs_id: $Id$
"""
import setupLibPaths
import argparse
import ePixFpga as fpga
import logging
import os
import pyrogue.utilities.fileio
import rogue
import sys
import time

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


class frameit(rogue.interfaces.stream.Slave):
    """Stream Slave subclass. Counts frames in real-time."""

    def __init__(self):
        """Sub Class Constructor."""
        rogue.interfaces.stream.Slave.__init__(self)
        self.EPIX10KA_FS = 274988  # frame size in bytes
        self.accepted = 0
        self.lost = 0
        self.processed = 0
        self.lastseq = 0
        self.log = logging.getLogger("epix10ka.acquirelog")

    def _acceptFrame(self, frame):
        """Override acceptframe."""
        p = bytearray(4)  # create an array of 4 bytes i.e. 32-bit word contintaing sequence #
        # store frame data (4 bytes) in p
        frame.read(p, 8)  # skip the 1st 4 bytes to get to sequence number, store it
        framesize = frame.getPayload()  # store size of frame
        seq = int.from_bytes(p, 'little')  # store sequence number
        if framesize != self.EPIX10KA_FS:  # test correct frame size
            self.log.error("frame size is not correct: acceptedFrames=%d lastseq=%d \
            seq=%d newframesize=%d" % (self.accepted, self.lastseq, seq, framesize))
            self.lost += 1
        else:
            if self.processed > 0:
                if self.lastseq+1 != seq:
                    self.log.error("lost a frame had=%d, got=%d" % (self.lastseq, seq))
                    self.lost += 1
            self.processed += 1

        self.lastseq = seq  # remember this sequence for next frame
        self.accepted += 1


def main():
    """Routine to acquire data. This uses argparse to get cli params.

    Example script using rogue library.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputfile", nargs=1, metavar=('FILE'),
                        help="File name to save acquired data to.")
    parser.add_argument("-y", "--yml", nargs=1, metavar=('YMLFILE'),
                        help="yml config file.", required=True)
    parser.add_argument("-t", "--time", nargs=1, metavar=('DURATION'), type=float,
                        help="total acquisition time.", required=True)
    args = parser.parse_args()

    if args.verbose:
        myloglevel = logging.DEBUG
    else:
        myloglevel = logging.INFO
    
    # create own logger, rogue hijacks default logger
    acq_log = logging.getLogger("epix10ka.acquirelog")
    acq_log.setLevel(myloglevel)
    # logging.getLogger('pyrogue').setLevel(logging.DEBUG)

    if args.yml:
        if not os.path.isfile(args.yml[0]):
            acq_log.error("[%s] yml config file is missing!", args.yml[0])
            sys.exit(1)

    if args.outputfile:
        ofilename = args.outputfile[0]
        if os.path.isfile(args.outputfile[0]):
            acq_log.warning("[%s] output file already exists, appending...!", args.outputfile[0])
    else:
        ofilename = time.strftime("%Y%m%d-%H%M%S") + ".dat"  # default file name

    if args.time[0] <= 0:
        acq_log.error("duration [%f] must be larger than 0", args.outputfile[0])
        sys.exit(1)

    # Set base
    board = pyrogue.Root(name='ePixBoard', description='ePix 10ka Board')

    # open pgpcard file descriptor
    pgpVc0 = rogue.hardware.pgp.PgpCard('/dev/datadev_0', 0, 1)  # Data & cmds
    pgpVc1 = rogue.hardware.pgp.PgpCard('/dev/datadev_0', 0, 0)  # Registers for ePix board
    acq_log.debug("PGP Card Version: %x" % (pgpVc0.getInfo().version))

    # config path
    srp = rogue.protocols.srp.SrpV0()  # construct register proto
    pyrogue.streamConnectBiDir(pgpVc1, srp)  # connect srp <--> pgpVc1

    # data path
    dw = pyrogue.utilities.fileio.StreamWriter(name='dataWriter')
    pyrogue.streamConnect(pgpVc0, dw.getChannel(0x1))  # connect pgpvc0 --> file

    # add devices to board
    board.add(dw)
    board.add(fpga.Epix10ka(name='Epix10ka', offset=0, memBase=srp, enabled=True))
    board.add(pyrogue.RunControl(name = 'runControl', description='Run Controller ePix 10ka'))
    board.start()

    # command board to read ePix config
    board.LoadConfig(args.yml[0])
    time.sleep(0.5)
    board.LoadConfig(args.yml[0])
    time.sleep(0.5)
    #targetASIC = getattr(board.Epix10ka, 'Epix10kaAsic' + str(asic))

 
    board.dataWriter.DataFile.set(ofilename)  # tell datawriter where to write data
    board.dataWriter.Open.set(True) 
    dw._writer.open(ofilename)

    acq_log.info("Finished configuration. Acquiring data ...")
    # start with a clean slate
    board.Epix10ka.EpixFpgaRegisters.AutoDaqEnable.set(False)
    board.Epix10ka.EpixFpgaRegisters.AutoRunEnable.set(False)
    #board.Epix10ka.EpixFpgaRegisters.AutoDaqEnable.set(True)
    #board.Epix10ka.EpixFpgaRegisters.AutoRunEnable.set(True)
 
    time.sleep(args.time[0])  # acquisition time
    acq_log.info("Flushing...")
    #board.Epix10ka.EpixFpgaRegisters.AutoDaqEnable.set(False)
    #board.Epix10ka.EpixFpgaRegisters.AutoRunEnable.set(False)

    board.dataWriter.Open.set(False)
    dw._writer.close()
    board.stop()
    acq_log.info("Done")

if __name__ == "__main__":
    main()
