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
#matplotlib.pyplot.ion()
MAX_NUMBER_OF_FRAMES_PER_BATCH  = 100000
_10KAFRAMESIZEBYTES = 274988

##################################################
# Global variables
##################################################
cameraType = 'ePix10ka'
bitMask = 0x3fff

##################################################
# Dark images
##################################################
if (len(sys.argv[1])>0):
    filename = sys.argv[1]
else:
    filename = '/u1/ddoering/10kaImages/darkImage_10ka_120Hz_afterClearMatrix.dat'

f = open(filename, mode = 'rb')

file_header = [0]
frame_header = [0]
numberOfFrames = 0
while ((len(file_header)>0) and (numberOfFrames<MAX_NUMBER_OF_FRAMES_PER_BATCH)):
    try:
        # reads file header [the number of bytes to read, EVIO]
        file_header = np.fromfile(f, dtype='uint32', count=2)
        payloadSize = int(file_header[0]/4)-1 #-1 is need because size info includes the second word from the header
        if (payloadSize != _10KAFRAMESIZEBYTES/4): #for 10ka
            print('Size test')
            print('Frame Number',numberOfFrames, 'size',  file_header)
            print('')
        newPayload = np.fromfile(f, dtype='uint32', count=payloadSize) #(frame size splited by four to read 32 bit 
        # test sequence number
        if (numberOfFrames != 0):
            if(frame_header[2]+1 != newPayload[2]):
                print('Sequence counter test')
                print ('Frame Number',numberOfFrames, 'size',  file_header)
                print('Previous sequence counter: ', frame_header[2], 'Current sequence counter: ', newPayload[2])
                print('')
        
        frame_header = newPayload[0:3]
        
#        if (numberOfFrames == 0):
#            allFrames = [newPayload.copy()]
#        else:
#            newFrame  = [newPayload.copy()]
#            allFrames = np.append(allFrames, newFrame, axis = 0)
        numberOfFrames = numberOfFrames + 1 
        #print ("Payload" , numberOfFrames, ":",  (newPayload[0:15]))
        previousSize = file_header
    except Exception: 
        #e = sys.exc_info()[0]
        #print ("Message\n", e)
        #print ('size', file_header, 'previous size', previousSize)
        print("numberOfFrames read: " ,numberOfFrames)



