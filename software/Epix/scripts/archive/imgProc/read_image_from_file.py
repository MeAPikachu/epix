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
MAX_NUMBER_OF_FRAMES_PER_BATCH  = 10

##################################################
# Global variables
##################################################
cameraType = 'ePix100a'
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
numberOfFrames = 0
while ((len(file_header)>0) and (numberOfFrames<MAX_NUMBER_OF_FRAMES_PER_BATCH)):
    try:
        # reads file header [the number of bytes to read, EVIO]
        file_header = np.fromfile(f, dtype='uint32', count=2)
        payloadSize = int(file_header[0]/4)-1 #-1 is need because size info includes the second word from the header
        print ('size',  file_header)
        newPayload = np.fromfile(f, dtype='uint32', count=payloadSize) #(frame size splited by four to read 32 bit 
        if (numberOfFrames == 0):
            allFrames = [newPayload.copy()]
        else:
            newFrame  = [newPayload.copy()]
            allFrames = np.append(allFrames, newFrame, axis = 0)
        numberOfFrames = numberOfFrames + 1 
        print ("Payload" , numberOfFrames, ":",  (newPayload[0:15]))
        previousSize = file_header
    except Exception: 
        #e = sys.exc_info()[0]
        #print ("Message\n", e)
        print ('size', file_header, 'previous size', previousSize)
        print("numberOfFrames read: " ,numberOfFrames)


##################################################
# image descrambling
##################################################
currentCam = cameras.Camera(cameraType = cameraType)
currentCam.bitMask = bitMask
#numberOfFrames = allFrames.shape[0]
print("numberOfFrames in the 3D array: " ,numberOfFrames)

if(numberOfFrames==1):
    [frameComplete, readyForDisplay, rawImgFrame] = currentCam.buildImageFrame(currentRawData = [], newRawData = allFrames[0])
    imgDesc = np.array([currentCam.descrambleImage(bytearray(rawImgFrame.tobytes()))])
else:
    for i in range(0, numberOfFrames):
        #get an specific frame
        [frameComplete, readyForDisplay, rawImgFrame] = currentCam.buildImageFrame(currentRawData = [], newRawData = allFrames[i,:])
    
        #get descrambled image from camera
        if (i==0):
            imgDesc = np.array([currentCam.descrambleImage(bytearray(rawImgFrame.tobytes()))])
        else:
            imgDesc = np.concatenate((imgDesc, np.array([currentCam.descrambleImage(bytearray(rawImgFrame.tobytes()))])),0)

##################################################
#from here on we have a set of images to work with
##################################################
#show first image
plt.imshow(imgDesc[0,:,:], interpolation='nearest')
plt.gray()
plt.colorbar()
plt.title('First image of :'+filename)
plt.show()


