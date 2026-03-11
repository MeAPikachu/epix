#-----------------------------------------------------------------------------
# Title      : read images from file script
#-----------------------------------------------------------------------------
# File       : read_image_from_file.py
# Created    : 2017-06-19
# Last update: 2017-06-21
#-----------------------------------------------------------------------------
# Description:
# Simple image viewer that enble a local feedback from data collected using
# ePix cameras. The initial intent is to use it with stand alone systems
#
#-----------------------------------------------------------------------------
# This file is part of the ePix rogue. It is subject to 
# the license terms in the LICENSE.txt file found in the top-level directory 
# of this distribution and at: 
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
# No part of the ePix rogue, including this file, may be 
# copied, modified, propagated, or distributed except according to the terms 
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import os, sys, time
import numpy as np
import ePixViewer.Cameras as cameras
import ePixViewer.imgProcessing as imgPr
# 
import matplotlib   
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import h5py
import argparse


#################################################################

# Set the argument parser
parser = argparse.ArgumentParser()

# Convert str to bool
argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

# Add arguments

parser.add_argument(
    "--f", 
    type     = str,
    required = True,
    help     = "Data file to convert",
)  

parser.add_argument(
    "--maxn", 
    type     = int,
    required = False,
    default  = 8000,
    help     = "Maximum number of frames to convert",
)

parser.add_argument(
    "--asic", 
    type     = int,
    required = True,
    default  = 0,
    help     = "ASIC number",
)

parser.add_argument(
    "--cnt", 
    type     = int,
    required = True,
    default  = 0,
    help     = "1 - counter A, 0 - counter B",
)

parser.add_argument(
    "--one_cnt", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "Use for ASIC data without default counter toggle mode",
)

parser.add_argument(
    "--df5", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "Create HDF5 file",
)

# Get the arguments
args = parser.parse_args()


##################################################
# convert selected ASIC/counter to HDF5
##################################################
filesNum = 0
onlyfiles = []
dir = ''
if os.path.isfile(args.f):
   if os.path.splitext(args.f)[1] == '.dat':
      filesNum = 1
      onlyfiles.append(os.path.split(args.f)[1])
      dir = os.path.split(args.f)[0]
elif os.path.isdir(args.f):
   onlyfiles = [f for f in os.listdir(args.f) if os.path.isfile(os.path.join(args.f, f))]
   i = 0
   while i < len(onlyfiles):
      if os.path.splitext(onlyfiles[i])[1] != '.dat':
         del onlyfiles[i]
      else:
         i = i + 1
   filesNum = len(onlyfiles)
   dir = args.f
else:
   print('No data files in %s'%args.f)

if filesNum == 0:
   print('No .dat files in %s' &args.f)
   exit()


## Filter files containing only TH1 > 750
#i=0
#while i < len(onlyfiles):
#   th1 = int(onlyfiles[i][20:23])
#   if th1 < 751:
#      del onlyfiles[i]
#   else:
#      i = i + 1
#filesNum = len(onlyfiles)


frmCnt = 7*[0]
progInd = 0
for i in range(filesNum):
   f = open(dir + '/' + onlyfiles[i], mode = 'rb')
   if args.cnt == 1:
      cntStr = '_CNTA'
   else:
      cntStr = '_CNTB'
   if args.asic == 0:
      asicStr = '_ASIC0'
   else:
      asicStr = '_ASIC1'
   out_filename = os.path.splitext(dir + '/' +  onlyfiles[i])[0]+cntStr+asicStr
   if args.df5:
      f_h5 = h5py.File(out_filename+".hdf5", "w")
   f_bin = open(out_filename+".bin", "wb")

   file_header = [0]
   numberOfFrames = 0
   prevSeq = -1
   prevPayload = []

   while ((len(file_header)>0) and (numberOfFrames<args.maxn)):
      try:
         
         # look for two packets from the same ASIC
         # only keep frame if both packets are in the file
         
         # reads file header 
         file_header = np.fromfile(f, dtype='uint32', count=2)
         payloadSize = int(file_header[0]/4)-1 
         newPayload = np.fromfile(f, dtype='uint32', count=payloadSize) #(frame size splited by four to read 32 bit 
            
         # store header info
         hdrSeq = newPayload[1]
         hdrAsic = newPayload[2] & 0x7
         hdrCnt = (newPayload[2] & 0x8)>>3
         
         # select asic
         if hdrAsic == args.asic:
            
            if args.one_cnt == False:
               # look for two equal sequence numbers for selected asic
               if prevSeq > 0 and prevSeq == hdrSeq and len(newPayload) == 1155 and prevPayloadLen == 1155:
                  if (numberOfFrames == 0):
                     if args.cnt == 1: # counter A - take 1st packet with the same sequence
                        allFrames = [prevPayload.copy()]
                        allSeq = np.array([hdrSeq])
                     else:             # counter B - take 2nd packet with the same sequence
                        allFrames = [newPayload.copy()]
                        allSeq = np.array([hdrSeq])
                  else:
                     if args.cnt == 1: # counter A - take 1st packet with the same sequence
                        newFrame  = [prevPayload.copy()]
                     else:             # counter B - take 2nd packet with the same sequence
                        newFrame  = [newPayload.copy()]
                     allFrames = np.append(allFrames, newFrame, axis = 0)
                     allSeq = np.append(allSeq, hdrSeq)
                  numberOfFrames = numberOfFrames + 1 
            
               #print('Prev seq %d, curr seq %d'%(prevSeq, hdrSeq))
               prevSeq = hdrSeq
               prevPayload = [newPayload.copy()]
               prevPayloadLen = len(newPayload)
            else:
               
               if len(newPayload) == 1155:
                  if (numberOfFrames == 0):
                     allFrames = [newPayload.copy()]
                     allSeq = np.array([hdrSeq])
                  else:
                     newFrame  = [newPayload.copy()]
                     allFrames = np.append(allFrames, newFrame, axis = 0)
                     allSeq = np.append(allSeq, hdrSeq)
                  numberOfFrames = numberOfFrames + 1 
            
               #print('Prev seq %d, curr seq %d'%(prevSeq, hdrSeq))
               prevPayload = [newPayload.copy()]
               
      except Exception: 
         pass
         e = sys.exc_info()[0]
         print ("Message\n", e)
         print("numberOfFrames read: " ,numberOfFrames)

   f.close()
   ###################################################
   ## image descrambling
   ###################################################



   #print("numberOfFrames in the 3D array: " ,numberOfFrames)

   currentRawData = []
   imgDesc = []
   if(numberOfFrames==1):
      #[frameComplete, readyForDisplay, rawImgFrame] = currentCam.buildImageFrame(currentRawData = [], newRawData = allFrames[0])
      #imgDesc = np.array([currentCam.descrambleImage(bytearray(rawImgFrame.tobytes()))])
      rawImgBytes = allFrames[0].tobytes()
      rawImgBytes = rawImgBytes[12:] # remove header
      imgDesc = np.frombuffer(rawImgBytes,dtype='uint16')
      imgDesc = np.array(imgDesc.reshape(48,48))
   else:
      for i in range(0, numberOfFrames):
         rawImgBytes = allFrames[i].tobytes()
         rawImgBytes = rawImgBytes[12:] # remove header
         if len(imgDesc) == 0:
            imgDesc = np.frombuffer(rawImgBytes,dtype='uint16')
            imgDesc = np.array(imgDesc.reshape(48,48))
         else:
            imgDescTmp = np.frombuffer(rawImgBytes,dtype='uint16')
            imgDescTmp = imgDescTmp.reshape(48,48)
            imgDesc = np.concatenate( (imgDesc, np.array(imgDescTmp)),0)
            
            #imgDesc = np.concatenate((imgDesc, np.array([currentCam.descrambleImage(bytearray(rawImgFrame.tobytes()))])),0)
   
   if numberOfFrames > 0:
      imgDesc = imgDesc.reshape(-1,48,48)
      index_h5 ='data'
      if args.df5:
         f_h5[index_h5] = imgDesc.astype('uint16')
         print('Saving %s with %d frames'%(out_filename, numberOfFrames))
         f_h5.close()
      f_bin.write(imgDesc.tobytes())
      f_bin.close()
   else:
      #print('Empty file %s with %d frames !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'%(out_filename, numberOfFrames))
      if args.df5:
         f_h5.close()
      f_bin.close()
   
   if numberOfFrames == 0:
      frmCnt[0] = frmCnt[0] + 1
   elif numberOfFrames > 0 and numberOfFrames < 10:
      frmCnt[1] = frmCnt[1] + 1
   elif numberOfFrames >= 10 and numberOfFrames < 20:
      frmCnt[2] = frmCnt[2] + 1
   elif numberOfFrames >= 20 and numberOfFrames < 50:
      frmCnt[3] = frmCnt[3] + 1
   elif numberOfFrames >= 50 and numberOfFrames < 100:
      frmCnt[4] = frmCnt[4] + 1
   elif numberOfFrames >= 100 and numberOfFrames < 1000:
      frmCnt[5] = frmCnt[5] + 1
   elif numberOfFrames >= 1000:
      frmCnt[6] = frmCnt[6] + 1
   progInd = progInd + 1
   
   if progInd % 100 == 0:
      print('Files with zero frames %d'%frmCnt[0])
      print('Files with 1 to 9 frames %d'%frmCnt[1])
      print('Files with 10 to 19 frames %d'%frmCnt[2])
      print('Files with 20 to 49 frames %d'%frmCnt[3])
      print('Files with 50 to 99 frames %d'%frmCnt[4])
      print('Files with 100 to 999 frames %d'%frmCnt[5])
      print('Files with more than 1000 frames %d'%frmCnt[6])


print('Files with zero frames %d'%frmCnt[0])
print('Files with 1 to 9 frames %d'%frmCnt[1])
print('Files with 10 to 19 frames %d'%frmCnt[2])
print('Files with 20 to 49 frames %d'%frmCnt[3])
print('Files with 50 to 99 frames %d'%frmCnt[4])
print('Files with 100 to 999 frames %d'%frmCnt[5])
print('Files with more than 1000 frames %d'%frmCnt[6])