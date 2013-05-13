#LATEST MODIFICATION: 12/05/2013
#This file contains the functions needed to compute the 2pt correlation function

from mpi4py import MPI
import numpy as np
import scipy.weave as wv
import sys,os
import time

###### MPI DEFINITIONS ######
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()





################################################################################
#This functions computes the TPCF (2pt correlation function) 
#from an N-body simulation. It takes into account boundary conditions
#VARIABLES:
#pos_g: array containing the positions of the galaxies
#pos_r: array containing the positions of the random particles catalogue
#BoxSize: Size of the Box. Units must be equal to those of pos_r/pos_g
#dims: number of divisions in one dimension to divide the box into subboxes
#DD_action: compute number of galaxy pairs from data or read them---compute/read
#RR_action: compute number of random pairs from data or read them---compute/read
#DR_action: compute number of galaxy-random pairs or read them---compute/read
#DD_name: file name to write/read galaxy-galaxy pairs results
#RR_name: file name to write/read random-random pairs results
#DR_name: file name to write/read galaxy-random pairs results
#bins: number of bins to compute the 2pt correlation function
#Rmin: minimum radius to compute the 2pt correlation function
#Rmax: maximum radius to compute the 2pt correlation function
#USAGE: at the end of the file there is a example of how to use this function
def TPCF(pos_g,pos_r,BoxSize,dims,DD_action,RR_action,DR_action,
         DD_name,RR_name,DR_name,bins,Rmin,Rmax,verbose=False):

    dims2=dims**2; dims3=dims**3

    ##### MASTER #####
    if myrank==0:

        #compute the indexes of the halo/subhalo/galaxy catalogue
        Ng=len(pos_g)*1.0; indexes_g=[]
        coord=np.floor(dims*pos_g/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_g.append(ids)
        indexes_g=np.array(indexes_g)

        #compute the indexes of the random catalogue
        Nr=len(pos_r)*1.0; indexes_r=[]
        coord=np.floor(dims*pos_r/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_r.append(ids)
        indexes_r=np.array(indexes_r)


        #compute galaxy-galaxy pairs
        if DD_action=='compute':
            DD=DDR_histogram(bins,Rmin,Rmax,BoxSize,dims,indexes_g,pos_g,pos2=None)
            if verbose:
                print DD
                print np.sum(DD)
            #write results to a file
            write_results(DD_name,DD,bins,'radial')
        else:
            #read results from a file
            DD,bins_aux=read_results(DD_name,'radial')
            if bins_aux!=bins:
                print 'Sizes are different!'
                sys.exit()

        #compute random-random pairs
        if RR_action=='compute':
            RR=DDR_histogram(bins,Rmin,Rmax,BoxSize,dims,indexes_r,pos_r,pos2=None)   
            if verbose:
                print RR
                print np.sum(RR)
            #write results to a file
            write_results(RR_name,RR,bins,'radial')
        else:
            #read results from a file
            RR,bins_aux=read_results(RR_name,'radial')
            if bins_aux!=bins:
                print 'Sizes are different!'
                sys.exit()

        #compute galaxy-random pairs
        if DR_action=='compute':
            DR=DDR_histogram(bins,Rmin,Rmax,BoxSize,dims,indexes_r,pos_g,pos_r)
            if verbose:
                print DR
                print np.sum(DR)
            #write results to a file
            write_results(DR_name,DR,bins,'radial')
        else:
            #read results from a file
            DR,bins_aux=read_results(DR_name,'radial')
            if bins_aux!=bins:
                print 'Sizes are different!'
                sys.exit()


        #final procesing
        bins_histo=np.logspace(np.log10(Rmin),np.log10(Rmax),bins+1)
        middle=0.5*(bins_histo[:-1]+bins_histo[1:])
        DD*=1.0; RR*=1.0; DR*=1.0

        r,xi_r,error_xi_r=[],[],[]
        for i in range(bins):
            if (RR[i]>0.0): #avoid divisions by 0
                xi_aux,error_xi_aux=xi(DD[i],RR[i],DR[i],Ng,Nr)
                r.append(middle[i])
                xi_r.append(xi_aux)
                error_xi_r.append(error_xi_aux)

        r=np.array(r)
        xi_r=np.array(xi_r)
        error_xi_r=np.array(error_xi_r)
        
        return r,xi_r,error_xi_r



    ##### SLAVES #####
    else:
        if DR_action=='compute':
            DDR_histogram(bins,Rmin,Rmax,BoxSize,dims,
                          indexes=None,pos1=None,pos2=None)                          
        if RR_action=='compute':
            DDR_histogram(bins,Rmin,Rmax,BoxSize,dims,
                          indexes=None,pos1=None,pos2=None)                          
        if DR_action=='compute':
            DDR_histogram(bins,Rmin,Rmax,BoxSize,dims,
                          indexes=None,pos1=None,pos2=None)                          
################################################################################






################################################################################
####### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ####### (x,y,z) very fast
################################################################################
def DDR_histogram(bins,Rmin,Rmax,BoxSize,dims,indexes,pos1,pos2):

    #we put bins+1. The last bin is only for ocassions when r=Rmax
    total_histogram=np.zeros(bins+1,dtype=np.int64) 

    ##### MASTER #####
    if myrank==0:
        #Master sends the indexes and particle positions to the slaves
        for i in range(1,nprocs):
            comm.send(pos1,dest=i,tag=7)
            comm.send(pos2,dest=i,tag=8)
            comm.send(indexes,dest=i,tag=9)

        #Masters distributes the calculation among slaves
        if pos2==None: #galaxy-galaxy or random-random case
            for subbox in range(dims**3):
                b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
                comm.send(False,dest=b,tag=2)
                comm.send(subbox,dest=b,tag=3)
        else:          #galaxy-random case
            i=0; number=len(pos1); IL=number/(nprocs-1)
            while i<number:
                b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
                comm.send(False,dest=b,tag=2)
                if i+IL<number:
                    #a=np.arange(0,10) -- a=array([0,1,2,3,4,5,6,7,8,9])
                    comm.send(np.arange(i,i+IL),dest=b,tag=4)
                else:
                    comm.send(np.arange(i,number),dest=b,tag=4)
                i+=IL

        #Master gathers partial results from slaves and return the final result
        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            total_histogram+=total_histogram_aux

        #the last element is just for situations in which r=Rmax
        total_histogram[bins-1]+=total_histogram[bins]

        return total_histogram[:-1]


    ##### SLAVES #####
    else:

        #slaves receive the positions and indexes
        pos1=comm.recv(source=0,tag=7)
        pos2=comm.recv(source=0,tag=8)
        indexes=comm.recv(source=0,tag=9)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            
            #galaxy-galaxy or random-random case
            if pos2==None: 
                subbox=comm.recv(source=0,tag=3)
                core_ids=indexes[subbox] #ids of the particles in the subbox
                #print subbox
                distances_core(pos1[core_ids],BoxSize,bins,Rmin,Rmax,total_histogram)

                for index in core_ids:
                    #second: compute the pairs of particles in the subbox with 
                    #particles in the neighboord subboxes
                    pos0=pos1[index]
                    ids=indexes_subbox_neigh(pos0,Rmax,dims,BoxSize,indexes,subbox)
                    if ids!=[]:
                        posN=pos1[ids]
                        distances(pos0,posN,BoxSize,bins,Rmin,Rmax,total_histogram)      
            
            #galaxy-random case
            else:          
                numbers=comm.recv(source=0,tag=4)
                #if np.any(numbers%10000==0):
                #    print numbers[np.where(numbers%10000==0)[0]]

                for i in numbers:
                    pos0=pos1[i]
                    #compute the ids of the particles in the neighboord subboxes
                    posN=pos2[indexes_subbox(pos0,Rmax,dims,BoxSize,indexes)]
                    distances(pos0,posN,BoxSize,bins,Rmin,Rmax,total_histogram)

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)

        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################


################################################################################
#this function computes the distances between all the particles-pairs and
#return the result in the histogram
def distances_core(pos,BoxSize,bins,Rmin,Rmax,histogram):
    x=pos[:,0]
    y=pos[:,1]
    z=pos[:,2]

    support = """
       #include <iostream>
       using namespace std;
    """
    code = """
       float middle=BoxSize/2.0;
       float dx,dy,dz,r;
       float delta=log10(Rmax/Rmin)/bins;
       int bin,i,j;
       int length=x.size();

       for (i=0;i<length;i++){
            for (j=i+1;j<length;j++){
                dx=(fabs(x(i)-x(j))<middle)?x(i)-x(j):BoxSize-fabs(x(i)-x(j));
                dy=(fabs(y(i)-y(j))<middle)?y(i)-y(j):BoxSize-fabs(y(i)-y(j));
                dz=(fabs(z(i)-z(j))<middle)?z(i)-z(j):BoxSize-fabs(z(i)-z(j));
                r=sqrt(dx*dx+dy*dy+dz*dz);

               if (r>=Rmin && r<=Rmax){
                   bin=(int)(log10(r/Rmin)/delta);
                   histogram(bin)+=1; 
               }
            }   
       }
    """
    wv.inline(code,
              ['BoxSize','Rmin','Rmax','bins','x','y','z','histogram'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'])

    return histogram
################################################################################
#pos1---a single position
#pos2---an array of positions
#the function returns the histogram of the computed distances between 
#pos1 and pos2
def distances(pos1,pos2,BoxSize,bins,Rmin,Rmax,histogram):

    x=pos2[:,0]
    y=pos2[:,1]
    z=pos2[:,2]

    support = """
         #include <iostream>
         using namespace std;
    """
    code = """
         float x0 = pos1(0);
         float y0 = pos1(1);
         float z0 = pos1(2);
         
         float middle = BoxSize/2.0;
         float dx,dy,dz,r;
         float delta = log10(Rmax/Rmin)/bins;
         int bin;

         for (int i=0;i<x.size();i++){
             dx = (fabs(x0-x(i))<middle) ? x0-x(i) : BoxSize-fabs(x0-x(i));
             dy = (fabs(y0-y(i))<middle) ? y0-y(i) : BoxSize-fabs(y0-y(i));
             dz = (fabs(z0-z(i))<middle) ? z0-z(i) : BoxSize-fabs(z0-z(i));
             r=sqrt(dx*dx+dy*dy+dz*dz);

             if (r>=Rmin && r<=Rmax){
                bin = (int)(log10(r/Rmin)/delta);
                histogram(bin)+=1; 
             }
         }
    """
    wv.inline(code,
              ['pos1','x','y','z','BoxSize','Rmin','Rmax','bins','histogram'],
              type_converters = wv.converters.blitz,
              support_code = support)

    return histogram
################################################################################
#this routine computes the IDs of all the particles within the neighboord cells
#that which can lie within the radius Rmax
def indexes_subbox(pos,Rmax,dims,BoxSize,indexes):

    #we add dims to avoid negative numbers. For example
    #if something hold between -1 and 5, the array to be
    #constructed should have indexes -1 0 1 2 3 4 5. 
    #To achieve this in a clever way we add dims
    i_min=int(np.floor((pos[0]-Rmax)*dims/BoxSize))+dims
    i_max=int(np.floor((pos[0]+Rmax)*dims/BoxSize))+dims
    j_min=int(np.floor((pos[1]-Rmax)*dims/BoxSize))+dims
    j_max=int(np.floor((pos[1]+Rmax)*dims/BoxSize))+dims
    k_min=int(np.floor((pos[2]-Rmax)*dims/BoxSize))+dims
    k_max=int(np.floor((pos[2]+Rmax)*dims/BoxSize))+dims
    
    i_array=np.arange(i_min,i_max+1)%dims
    j_array=np.arange(j_min,j_max+1)%dims
    k_array=np.arange(k_min,k_max+1)%dims

    PAR_indexes=np.array([])
    for i in i_array:
        for j in j_array:
            for k in k_array:
                num=dims**2*i+dims*j+k
                ids=indexes[num]
                PAR_indexes=np.concatenate((PAR_indexes,ids)).astype(np.int32)

    return PAR_indexes
################################################################################
#this routine returns the ids of the particles in the neighboord cells
#that havent been already selected
def indexes_subbox_neigh(pos,Rmax,dims,BoxSize,indexes,subbox):

    #we add dims to avoid negative numbers. For example
    #if something hold between -1 and 5, the array to be
    #constructed should have indexes -1 0 1 2 3 4 5. 
    #To achieve this in a clever way we add dims
    i_min=int(np.floor((pos[0]-Rmax)*dims/BoxSize))+dims
    i_max=int(np.floor((pos[0]+Rmax)*dims/BoxSize))+dims
    j_min=int(np.floor((pos[1]-Rmax)*dims/BoxSize))+dims
    j_max=int(np.floor((pos[1]+Rmax)*dims/BoxSize))+dims
    k_min=int(np.floor((pos[2]-Rmax)*dims/BoxSize))+dims
    k_max=int(np.floor((pos[2]+Rmax)*dims/BoxSize))+dims
    
    i_array=np.arange(i_min,i_max+1)%dims
    j_array=np.arange(j_min,j_max+1)%dims
    k_array=np.arange(k_min,k_max+1)%dims

    ids=np.array([])
    for i in i_array:
        for j in j_array:
            for k in k_array:
                num=dims**2*i+dims*j+k
                if num>subbox:
                    ids_subbox=indexes[num]
                    ids=np.concatenate((ids,ids_subbox)).astype(np.int32)
    return ids
################################################################################
#This function computes the correlation function and its error once the number
#of galaxy-galaxy, random-random & galaxy-random pairs are given together
#with the total number of galaxies and random points
def xi(GG,RR,GR,Ng,Nr):
    
    normGG=2.0/(Ng*(Ng-1.0))
    normRR=2.0/(Nr*(Nr-1.0))
    normGR=1.0/(Ng*Nr)

    GGn=GG*normGG
    RRn=RR*normRR
    GRn=GR*normGR
    
    xi=GGn/RRn-2.0*GRn/RRn+1.0

    fact=normRR/normGG*RR*(1.0+xi)+4.0/Ng*(normRR*RR/normGG*(1.0+xi))**2
    err=normGG/(normRR*RR)*np.sqrt(fact)
    err=err*np.sqrt(3.0)

    return xi,err
################################################################################









################################################################################
#This function writes partial results to a file
def write_results(fname,histogram,bins,case):
    f=open(fname,'w')
    if case=='par-perp':
        for i in range(len(histogram)):
            coord_perp=i/bins
            coord_par=i%bins
            f.write(str(coord_par)+' '+str(coord_perp)+' '+str(histogram[i])+'\n')
    elif case=='radial':
        for i in range(len(histogram)):
            f.write(str(i)+' '+str(histogram[i])+'\n')
    else:
        print 'Error in the description of case:'
        print 'Choose between: par-perp or radial'
    f.close()        
################################################################################
#This functions reads partial results of a file
def read_results(fname,case):

    histogram=[]

    if case=='par-perp':
        bins=np.around(np.sqrt(size)).astype(np.int64)

        if bins*bins!=size:
            print 'Error finding the size of the matrix'
            sys.exit()

        f=open(fname,'r')
        for line in f.readlines():
            a=line.split()
            histogram.append(int(a[2]))
        f.close()
        histogram=np.array(histogram)
        return histogram,bins
    elif case=='radial':
        f=open(fname,'r')
        for line in f.readlines():
            a=line.split()
            histogram.append(int(a[1]))
        f.close()
        histogram=np.array(histogram)
        return histogram,histogram.shape[0]
    else:
        print 'Error in the description of case:'
        print 'Choose between: par-perp or radial'
################################################################################






############ EXAMPLE OF USAGE ############

points_g=150000
points_r=200000

BoxSize=500.0 #Mpc/h
Rmin=1.0      #Mpc/h
Rmax=50.0     #Mpc/h
dims=10
bins=30

DD_action='compute'
RR_action='compute'
DR_action='compute'
DD_name='DD.dat'
RR_name='RR.dat'
DR_name='DR.dat'

if myrank==0:
    pos_g=np.random.random((points_g,3))*BoxSize
    pos_r=np.random.random((points_r,3))*BoxSize

    start=time.clock()
    r,xi_r,error_xi=TPCF(pos_g,pos_r,BoxSize,dims,DD_action,RR_action,DR_action,DD_name,RR_name,DR_name,bins,Rmin,Rmax,verbose=True)

    print r
    print xi_r
    print error_xi
    end=time.clock()
    print 'time:',end-start
else:
    pos_g=None; pos_r=None
    TPCF(pos_g,pos_r,BoxSize,dims,DD_action,RR_action,DR_action,DD_name,RR_name,DR_name,bins,Rmin,Rmax,verbose=True)

