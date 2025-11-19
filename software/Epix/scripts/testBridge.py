#!/usr/bin/env python

import rogue.interfaces.memory
import time

class Bridge(rogue.interfaces.memory.Master,rogue.interfaces.memory.Slave):

   def __init__(self):
      rogue.interfaces.memory.Master.__init__(self)
      rogue.interfaces.memory.Slave.__init__(self,4,4)
      self.done = False
      self.err  = 0

   def _doMaxAccess(self):
      """ Respond to max access request by forwarding to downstream slave"""
      return(self._reqMaxAccess())

   def _doMinAccess(self):
      """ Respond to min access request by forwarding to downstream slave"""
      return(self._reqMinAccess())

   def _doTransaction(self,tid,master,address,size,type):
      """ Incoming transaction request"""
      self.done = False
      ba = bytearray(size)

      # Write request
      if type == rogue.interfaces.memory.Write or type == rogue.interfaces.memory.Post:

         # First get data from incoming master
         # Data will be put in local byte array
         master._getTransactionData(tid,0,ba)

         # Request write transaction to downstream slave
         # Downstream slave will pull data from local byte array
         self._reqTransaction(address,ba,write,type)

         # Wait for downstream write request to complete. 
         # Indicated by doneTransaction called from downstream.
         while not self.done:
            time.sleep(.1)

         # Indicate that transaction is down to upstream master
         master._doneTransaction(tid,self.err)
         #print("End transaction. id=%i" % (tid))

         # Mark transaction as complete in local master
         self._endTransaction()

      else:
         # Request read transaction to downstream slave
         # Data will be put in local byte array
         self._reqTransaction(address,ba,type)

         # Wait for downstream read request to complete. 
         # Indicated by doneTransaction called from downstream.
         while not self.done:
            time.sleep(.1)

         # Push read data from local byte array to upstream master
         master._setTransactionData(tid,0,ba)

         # Indicate transaction is down to upstream master
         master._doneTransaction(tid,self.err)

         # Mark transaction as complete in local master
         self._endTransaction()

   def _doneTransaction(self,dId,err):
      """ Callback function called from downstream slave when transaction is complete"""
      self.err = err
      self.done = True

