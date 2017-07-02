import numpy as np
import sys,os

#this routine reads the information about AMIGA
#USAGE: halos=read_amiga(file) ---> this will read a file called "file_no_subhalos" which contains only the halos. To read the file with the halos and subhalos do halos=read_amiga(file,subhalos=True). To read the file containing only subhalos do halos=read_amiga(file,only_subhalos=True)
#OUTPUT: the positions will be stored as halos.pos, the velocities as halos.pos...etc. For other fields needed modify the routine where indicated
class read_amiga:
    def __init__(self,filename,subhalos=False,only_subhalos=False):

        if not(subhalos) and not(only_subhalos):
            filename=filename+'_no_subhalos'
        elif subhalos:
            filename=filename
        else:
            filename=filename+'_subhalos'
        
        if not(os.path.exists(filename)):
            print "file not found:", filename
            sys.exit()

        #if other fields are needed, add them here
        Mvir,Rvir,pos,vel=[],[],[],[]
        f=open(filename,'r')
        f.readline() #we ignore AMIGA header
        for line in f.readlines():
            a=line.split()
            pos.append([float(a[2]),float(a[3]),float(a[4])])
            vel.append([float(a[5]),float(a[6]),float(a[7])])
            Mvir.append(float(a[8]))
            Rvir.append(float(a[9]))
        f.close()
            
        pos=np.array(pos)*1e3 #we transfer Mpc/h to kpc/h
        vel=np.array(vel)
        Mvir=np.array(Mvir)
        Rvir=np.array(Rvir)

        self.pos=pos
        self.vel=vel
        self.M=Mvir
        self.R=Rvir
        
