#model read_FoF"""
#readfof(Base,snapnum,endian=None)
#Read FoF files from Gadget, P-FoF
     #Parameters:
        #basedir: where your FoF folder located
        #snapnum: snapshot number
        #long_ids: whether particles ids are uint32 or uint64
        #swap: False or True
     #return structures:
        #TotNgroups,TotNids,GroupLen,GroupOffset,GroupMass,GroupPos,GroupIDs...
     #Example:
        #--------
        #FoF_halos=readfof("/data1/villa/b500p512nu0.6z99tree",17,long_ids=True,swap=False)
        #Masses=FoF_halos.GroupMass
        #IDs=FoF_halos.GroupIDs
        #--------
        #updated time 19 Oct 2012 by wgcui

import numpy as np
import os
from struct import unpack

class FoF_catalog:
    def __init__(self,basedir,snapnum,long_ids=False,swap=False):

        if long_ids: format=np.uint64
        else: format=np.uint32

        exts='000'+str(snapnum)
        exts=exts[-3:]

        #################  READ TAB FILES ################# 
        fnb,skip=0,0 
        Final=False
        while not(Final):
            fname=basedir+"/groups_" + exts +"/group_tab_"+exts +"."+str(fnb)
            f=open(fname,'rb')
            self.Ngroups=np.fromfile(f,dtype=np.int32,count=1)[0]
            self.TotNgroups=np.fromfile(f,dtype=np.int32,count=1)[0]
            self.Nids=np.fromfile(f,dtype=np.int32,count=1)[0]
            self.TotNids=np.fromfile(f,dtype=np.uint64,count=1)[0]
            self.Nfiles=np.fromfile(f,dtype=np.uint32,count=1)[0]

            TNG=self.TotNgroups
            NG=self.Ngroups
            if fnb == 0:
                self.GroupLen=np.empty(TNG,dtype=np.int32)
                self.GroupOffset=np.empty(TNG,dtype=np.int32)
                self.GroupMass=np.empty(TNG,dtype=np.float32)
                self.GroupPos=np.empty(TNG,dtype=np.dtype((np.float32,3)))
                self.GroupVel=np.empty(TNG,dtype=np.dtype((np.float32,3)))
                self.GroupTLen=np.empty(TNG,dtype=np.dtype((np.float32,6)))
                self.GroupTMass=np.empty(TNG,dtype=np.dtype((np.float32,6)))
            if NG>0:
                locs=slice(skip,skip+NG)
                self.GroupLen[locs]=np.fromfile(f,dtype=np.int32,count=NG)
                self.GroupOffset[locs]=np.fromfile(f,dtype=np.int32,count=NG)
                self.GroupMass[locs]=np.fromfile(f,dtype=np.float32,count=NG)
                self.GroupPos[locs]=np.fromfile(f,dtype=np.dtype((np.float32,3)),count=NG)
                self.GroupVel[locs]=np.fromfile(f,dtype=np.dtype((np.float32,3)),count=NG)
                self.GroupTLen[locs]=np.fromfile(f,dtype=np.dtype((np.float32,6)),count=NG)
                self.GroupTMass[locs]=np.fromfile(f,dtype=np.dtype((np.float32,6)),count=NG)
                skip+=NG

                if swap:
                    self.GroupLen.byteswap(True)
                    self.GroupOffset.byteswap(True)
                    self.GroupMass.byteswap(True)
                    self.GroupPos.byteswap(True)
                    self.GroupVel.byteswap(True)
                    self.GroupTLen.byteswap(True)
                    self.GroupTMass.byteswap(True)

            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                print "Warning: finished reading before EOF for tab file",fnb
                print curpos,f.tell()
            f.close()
            fnb+=1
            if fnb==self.Nfiles: Final=True

        #################  READ IDS FILES ################# 
        fnb,skip=0,0
        Final=False
        while not(Final):
            fname=basedir+"/groups_" + exts +"/group_ids_"+exts +"."+str(fnb)
            f=open(fname,'rb')
            Ngroups=np.fromfile(f,dtype=np.uint32,count=1)[0]
            TotNgroups=np.fromfile(f,dtype=np.uint32,count=1)[0]
            Nids=np.fromfile(f,dtype=np.uint32,count=1)[0]
            TotNids=np.fromfile(f,dtype=np.uint64,count=1)[0]
            Nfiles=np.fromfile(f,dtype=np.uint32,count=1)[0]
            Send_offset=np.fromfile(f,dtype=np.uint32,count=1)[0]
            if fnb==0:
                self.GroupIDs=np.zeros(dtype=format,shape=TotNids)
            if Ngroups>0:
                if long_ids:
                    IDs=np.fromfile(f,dtype=np.uint64,count=Nids)
                else:
                    IDs=np.fromfile(f,dtype=np.uint32,count=Nids)
                if swap:
                    IDs=IDs.byteswap(True)
                self.GroupIDs[skip:skip+Nids]=IDs[:]
                skip+=Nids
            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                print "Warning: finished reading before EOF for IDs file",fnb
            f.close()
            fnb+=1
            if fnb==Nfiles: Final=True
