from mpi4py import MPI
import numpy as np
import sys
import scipy.weave as wv

########### READ MOCK CATALOG ###########
################################################################################
### modify this function to read any other column
def read_mock_catalog(root,lines_header,number_of_files):
    #### FIND THE SIZE OF THE ARRAY ####
    size=0
    for i in range(number_of_files):
        fname=root+str(i)
        f=open(fname,'r')
        size+=len(f.readlines())-lines_header
        f.close()

    #### DEFINE ARRAYS ####
    z=np.empty(size,np.float64)
    ra=np.empty(size,np.float64)
    dec=np.empty(size,np.float64)
    L_tot_Halpha=np.empty(size,np.float64)

    #### READ THE FILES ####
    count=0
    for i in range(number_of_files):
        fname=root+str(i)
        f=open(fname,'r')
        print fname
        ## READ HEADER ##
        for i in range(lines_header):
            f.readline()
        ## READ DATA ##
        for line in f.readlines():
            a=line.split()
            L_tot_Halpha[count]=float(a[9])
            z[count]=float(a[60])
            ra[count]=float(a[63])
            dec[count]=float(a[64])
            count+=1
        f.close()

    return ra,dec,z,L_tot_Halpha
################################################################################

########### READ MOCK CATALOG ###########
################################################################################
def read_random_catalog(fname):
    f=open(fname,'r')
    size=len(f.readlines())
    f.close()

    ra=np.empty(size,dtype=np.float64)
    dec=np.empty(size,dtype=np.float64)
    z=np.empty(size,dtype=np.float64)

    f=open(fname,'r')
    count=0
    for line in f.readlines():
        a=line.split()
        ra[count]=float(a[0])
        dec[count]=float(a[1])
        z[count]=float(a[2])
        count+=1
    f.close()

    return ra,dec,z
################################################################################

####### WRITE PARTIAL RESULTS TO FILE #######
################################################################################
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

####### READ PARTIAL RESULTS OF FILE #######
################################################################################
def read_results(fname,case):
    f=open(fname,'r')
    size=len(f.readlines())
    f.close()

    histogram=np.empty(size,dtype=np.int64)

    if case=='par-perp':
        bins=np.around(np.sqrt(size)).astype(np.int64)

        if bins*bins!=size:
            print 'Error finding the size of the matrix'
            sys.exit()

        f=open(fname,'r')
        count=0
        for line in f.readlines():
            a=line.split()
            histogram[count]=int(a[2])
            count+=1
        f.close()
        return histogram,bins
    elif case=='radial':
        f=open(fname,'r')
        count=0
        for line in f.readlines():
            a=line.split()
            histogram[count]=int(a[1])
            count+=1
        f.close()
        return histogram,size
    else:
        print 'Error in the description of case:'
        print 'Choose between: par-perp or radial'
################################################################################

####### READ RESULTS FROM A FILE #######
################################################################################
def read_final_results(fname):
    f=open(fname,'r')
    size=len(f.readlines())
    f.close()

    bins=np.around(np.sqrt(size)).astype(np.int64)

    if bins*bins!=size:
        print 'Error while finding the size of the matrix'
        sys.exit()

    bins_par=np.empty(size,dtype=np.int32)
    bins_perp=np.empty(size,dtype=np.int32)
    histogram=np.empty(size,dtype=np.float64)

    f=open(fname,'r')
    count=0
    for line in f.readlines():
        a=line.split()
        bins_par[count]=int(a[0])
        bins_perp[count]=int(a[1])
        histogram[count]=float(a[2])
        count+=1
    f.close()
    
    return bins_par,bins_perp,histogram
################################################################################

####### CREATE THE INTERVALS FOR THE HISTOGRAM ####### (linear)
################################################################################
def histogram_bins(min_distance,max_distance,bins):
    bins_histo=np.linspace(min_distance,max_distance,bins+1)
    return bins_histo
################################################################################

####### CREATE THE INTERVALS FOR THE HISTOGRAM ####### (logaritmic)
################################################################################
def histogram_bins_log(min_distance,max_distance,bins):
    minimum=np.log10(min_distance)
    maximum=np.log10(max_distance)
    bins_histo=np.logspace(minimum,maximum,bins+1)
    return bins_histo
################################################################################

####### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ####### (X-Y-Z)
################################################################################
def DD_histogram(myrank,nprocs,comm,x,y,z,axis_dir,bins,bin_size,IL):

    total_histogram=np.zeros(bins**2,dtype=np.int64)
    bins_histo=histogram_bins(-0.5,bins**2-0.5,bins**2)

    if myrank==0:
        number=len(x)
        pos=np.empty((number,3),dtype=np.float64)
        pos[:,0]=x; pos[:,1]=y; pos[:,2]=z

        for i in range(1,nprocs):
            comm.send(pos,dest=i,tag=9)

        i=0
        while i<number:
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            if i+IL<number:
                comm.send(np.arange(i,i+IL),dest=b,tag=3)
            else:
                comm.send(np.arange(i,number-1),dest=b,tag=3)
            i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            print 'array received from',b
            total_histogram+=total_histogram_aux

        return total_histogram

    else:
        pos=comm.recv(source=0,tag=9)
        number=len(pos)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            indexes=comm.recv(source=0,tag=3)
            for i in range(len(indexes)):
                if indexes[i]%10000==0:
                    print indexes[i]
                
                pos_new=pos[indexes[i]+1:number]
                difference=pos_new-pos[indexes[i]]
                dist2=np.sum(difference*difference,axis=1)

                r_par2=(np.dot(difference,axis_dir))**2
                r_perp2=dist2-r_par2
                if np.any(r_perp2<0.0):
                    inside=np.where(r_perp2<0.0)[0]
                    if np.min(r_perp2[inside])<-0.01:
                        print 'there is a problem'
                        sys.exit()
                    else:
                        r_perp2[inside]=0.0
                    
                r_par=np.sqrt(r_par2)
                index_par=(r_par/bin_size).astype(np.int64)
                index_par[np.where(index_par>=bins)[0]]=bins**2

                r_perp=np.sqrt(r_perp2)
                index_perp=(r_perp/bin_size).astype(np.int64)
                index_perp[np.where(index_perp>=bins)[0]]=bins**2

                ids=bins*index_perp+index_par

                hist=np.histogram(ids,bins=bins_histo)[0]
                total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)

        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################

####### COMPUTE THE NUMBER OF PAIRS BETWEEN TWO CATALOGS ####### (X-Y-Z)
################################################################################
def DR_histogram(myrank,nprocs,comm,x1,y1,z1,x2,y2,z2,axis_dir,bins,bin_size,IL):

    total_histogram=np.zeros(bins**2,dtype=np.int64)
    bins_histo=histogram_bins(-0.5,bins**2-0.5,bins**2)

    if myrank==0:
        number1=len(x1);     number2=len(x2)
        pos1=np.empty((number1,3),dtype=np.float64)
        pos2=np.empty((number2,3),dtype=np.float64)

        pos1[:,0]=x1;  pos1[:,1]=y1;  pos1[:,2]=z1
        pos2[:,0]=x2;  pos2[:,1]=y2;  pos2[:,2]=z2

        for i in range(1,nprocs):
            comm.send(pos1,dest=i,tag=8)
            comm.send(pos2,dest=i,tag=9)

        i=0
        while i<number1:
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            if i+IL<number1:
                comm.send(np.arange(i,i+IL),dest=b,tag=3)
            else:
                comm.send(np.arange(i,number1),dest=b,tag=3)
            i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            print 'array received from',b
            total_histogram+=total_histogram_aux

        return total_histogram

    else:
        pos1=comm.recv(source=0,tag=8)
        pos2=comm.recv(source=0,tag=9)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            indexes=comm.recv(source=0,tag=3)
            for i in range(len(indexes)):
                if indexes[i]%10000==0:
                    print indexes[i]
                
                difference=pos2-pos1[indexes[i]]
                dist2=np.sum(difference*difference,axis=1)

                r_par2=(np.dot(difference,axis_dir))**2
                r_perp2=dist2-r_par2
                if np.any(r_perp2<0.0):
                    inside=np.where(r_perp2<0.0)[0]
                    if np.min(r_perp2[inside])<-0.01:
                        print 'there is a problem'
                    r_perp2[inside]=0.0
                    
                r_par=np.sqrt(r_par2)
                index_par=(r_par/bin_size).astype(np.int64)
                index_par[np.where(index_par>=bins)[0]]=bins**2

                r_perp=np.sqrt(r_perp2)
                index_perp=(r_perp/bin_size).astype(np.int64)
                index_perp[np.where(index_perp>=bins)[0]]=bins**2

                ids=bins*index_perp+index_par

                hist=np.histogram(ids,bins=bins_histo)[0]
                total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)
        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################

####### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ####### (x,y,z) White
################################################################################
def DD_histogram_P(myrank,nprocs,comm,pos,bins,bin_size,IL):

    total_histogram=np.zeros(bins**2,dtype=np.int64)
    bins_histo=histogram_bins(-0.5,bins**2-0.5,bins**2)

    if myrank==0:
        number=len(pos)

        for i in range(1,nprocs):
            comm.send(pos,dest=i,tag=7)

        i=0
        while i<number:
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            if i+IL<number:
                #a=np.arange(0,10) -- a=array([0,1,2,3,4,5,6,7,8,9])
                comm.send(np.arange(i,i+IL),dest=b,tag=3)
            else:
                #to avoid get to the end, set number-1
                comm.send(np.arange(i,number-1),dest=b,tag=3)
            i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            print 'array received from',b
            total_histogram+=total_histogram_aux

        return total_histogram

    else:
        pos=comm.recv(source=0,tag=7)
        number=len(pos)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            indexes=comm.recv(source=0,tag=3)
            for i in indexes:
                if i%10000==0:
                    print i
                
                posN=pos[i+1:number]
                pos0=pos[i]

                s=posN-pos0
                s2=np.sum(s**2,axis=1)

                l=0.5*(posN+pos0)
                l_mod=np.sqrt(np.sum(l**2,axis=1))
                for j in range(3):
                    l[:,j]=l[:,j]/l_mod

                A=np.sum((s+l)**2,axis=1)

                r_par=0.5*(A-s2-1.0)

                arg=s2-r_par**2

                if any(arg<0.0):
                    arg[np.where(arg<0.0)[0]]=0.0
                r_perp=np.sqrt(arg)

                index_par=(r_par/bin_size).astype(np.int64)
                index_par[np.where(index_par>=bins)[0]]=bins**2

                index_perp=(r_perp/bin_size).astype(np.int64)
                index_perp[np.where(index_perp>=bins)[0]]=bins**2

                ids=bins*index_perp+index_par

                hist=np.histogram(ids,bins=bins_histo)[0]
                total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)

        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################

####### COMPUTE THE NUMBER OF PAIRS BETWEEN TWO CATALOGS ####### (x,y,z) White
################################################################################
def DR_histogram_P(myrank,nprocs,comm,pos1,pos2,bins,bin_size,IL):

    total_histogram=np.zeros(bins**2,dtype=np.int64)
    bins_histo=histogram_bins(-0.5,bins**2-0.5,bins**2)

    if myrank==0:
        number=len(pos1)

        for i in range(1,nprocs):
            comm.send(pos1,dest=i,tag=4)
            comm.send(pos2,dest=i,tag=5)

        i=0
        while i<number:
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            if i+IL<number:
                comm.send(np.arange(i,i+IL),dest=b,tag=3)
            else:
                comm.send(np.arange(i,number),dest=b,tag=3)
            i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            print 'array received from',b
            total_histogram+=total_histogram_aux

        return total_histogram

    else:
        pos1=comm.recv(source=0,tag=4)
        pos2=comm.recv(source=0,tag=5)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            indexes=comm.recv(source=0,tag=3)
            for i in indexes:
                if i%10000==0:
                    print i

                pos0=pos1[i]

                s=pos2-pos0
                s2=np.sum(s**2,axis=1)

                l=0.5*(pos2+pos0)
                l_mod=np.sqrt(np.sum(l**2,axis=1))
                for j in range(3):
                    l[:,j]=l[:,j]/l_mod

                A=np.sum((s+l)**2,axis=1)

                r_par=0.5*(A-s2-1.0)

                arg=s2-r_par**2

                if any(arg<0.0):
                    arg[np.where(arg<0.0)[0]]=0.0
                r_perp=np.sqrt(arg)

                index_par=(r_par/bin_size).astype(np.int64)
                index_par[np.where(index_par>=bins)[0]]=bins**2

                index_perp=(r_perp/bin_size).astype(np.int64)
                index_perp[np.where(index_perp>=bins)[0]]=bins**2

                ids=bins*index_perp+index_par

                hist=np.histogram(ids,bins=bins_histo)[0]
                total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)
        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################

####### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ####### (RA-DEC-R)
################################################################################
def DD_histogram_M(myrank,nprocs,comm,ra,dec,r,bins,bin_size,IL):

    total_histogram=np.zeros(bins**2,dtype=np.int64)
    bins_histo=histogram_bins(-0.5,bins**2-0.5,bins**2)

    if myrank==0:
        number=len(ra)

        for i in range(1,nprocs):
            comm.send(ra,dest=i,tag=7)
            comm.send(dec,dest=i,tag=8)
            comm.send(r,dest=i,tag=9)

        i=0
        while i<number:
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            if i+IL<number:
                #a=np.arange(0,10) -- a=array([0,1,2,3,4,5,6,7,8,9])
                comm.send(np.arange(i,i+IL),dest=b,tag=3)
            else:
                #to avoid get to the end, set number-1
                comm.send(np.arange(i,number-1),dest=b,tag=3)
            i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            print 'array received from',b
            total_histogram+=total_histogram_aux

        return total_histogram

    else:
        ra=comm.recv(source=0,tag=7)
        dec=comm.recv(source=0,tag=8)
        r=comm.recv(source=0,tag=9)

        number=len(ra)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            indexes=comm.recv(source=0,tag=3)
            for i in indexes:
                if i%10000==0:
                    print i
                
                raN=ra[i+1:number]
                decN=dec[i+1:number]
                rN=r[i+1:number]
                
                ra0=ra[i]
                dec0=dec[i]
                r0=r[i]

                value=np.sin(decN)*np.sin(dec0)+np.cos(decN)*np.cos(dec0)*np.cos(raN-ra0)

                if np.any(value>1.0):
                    value[np.where(value>1.0)[0]]=1.0
                if np.any(value<-1.0):
                    print 'strange result: already fixed'
                    value[np.where(value<-1.0)[0]]=-1.0
                theta=np.arccos(value)

                r_par=np.absolute(rN-r0)
                r_perp=np.tan(0.5*theta)*4.0*rN*r0/(rN+r0)
                
                index_par=(r_par/bin_size).astype(np.int64)
                index_par[np.where(index_par>=bins)[0]]=bins**2

                index_perp=(r_perp/bin_size).astype(np.int64)
                index_perp[np.where(index_perp>=bins)[0]]=bins**2

                ids=bins*index_perp+index_par

                hist=np.histogram(ids,bins=bins_histo)[0]
                total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)

        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################

####### COMPUTE THE NUMBER OF PAIRS BETWEEN TWO CATALOGS ####### (RA-DEC-R)
################################################################################
def DR_histogram_M(myrank,nprocs,comm,ra1,dec1,r1,ra2,dec2,r2,bins,bin_size,IL):

    total_histogram=np.zeros(bins**2,dtype=np.int64)
    bins_histo=histogram_bins(-0.5,bins**2-0.5,bins**2)

    if myrank==0:
        number=len(ra1)

        for i in range(1,nprocs):
            comm.send(ra1,dest=i,tag=4)
            comm.send(dec1,dest=i,tag=5)
            comm.send(r1,dest=i,tag=6)
            comm.send(ra2,dest=i,tag=7)
            comm.send(dec2,dest=i,tag=8)
            comm.send(r2,dest=i,tag=9)

        i=0
        while i<number:
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            if i+IL<number:
                comm.send(np.arange(i,i+IL),dest=b,tag=3)
            else:
                comm.send(np.arange(i,number),dest=b,tag=3)
            i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            print 'array received from',b
            total_histogram+=total_histogram_aux

        return total_histogram

    else:
        ra1=comm.recv(source=0,tag=4)
        dec1=comm.recv(source=0,tag=5)
        r1=comm.recv(source=0,tag=6)
        ra2=comm.recv(source=0,tag=7)
        dec2=comm.recv(source=0,tag=8)
        r2=comm.recv(source=0,tag=9)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            indexes=comm.recv(source=0,tag=3)
            for i in indexes:
                if i%10000==0:
                    print i
                
                ra0=ra1[i]
                dec0=dec1[i]
                r0=r1[i]

                value=np.sin(dec2)*np.sin(dec0)+np.cos(dec2)*np.cos(dec0)*np.cos(ra2-ra0)

                if np.any(value>1.0):
                    value[np.where(value>1.0)[0]]=1.0
                if np.any(value<-1.0):
                    print 'strange result: already fixed'
                    value[np.where(value<-1.0)[0]]=-1.0
                theta=np.arccos(value)

                r_par=np.absolute(r2-r0)
                r_perp=np.tan(0.5*theta)*4.0*r2*r0/(r2+r0)

                index_par=(r_par/bin_size).astype(np.int64)
                index_par[np.where(index_par>=bins)[0]]=bins**2

                index_perp=(r_perp/bin_size).astype(np.int64)
                index_perp[np.where(index_perp>=bins)[0]]=bins**2

                ids=bins*index_perp+index_par

                hist=np.histogram(ids,bins=bins_histo)[0]
                total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)
        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################

####### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ####### (x,y,z) White
################################################################################
def DDR_histogram(myrank,nprocs,comm,bins,minimum,maximum,IL,BoxSize,pos1,pos2):

    total_histogram=np.zeros(bins,dtype=np.int64)
    bins_histo=histogram_bins_log(minimum,maximum,bins)
    #bins_histo=histogram_bins(minimum,maximum,bins)

    if myrank==0:

        number=len(pos1)
        for i in range(1,nprocs):
            comm.send(pos1,dest=i,tag=7)
            comm.send(pos2,dest=i,tag=8)

        i=0
        while i<number:
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            if i+IL<number:
                #a=np.arange(0,10) -- a=array([0,1,2,3,4,5,6,7,8,9])
                comm.send(np.arange(i,i+IL),dest=b,tag=3)
            else:
                if pos2!=None:
                    comm.send(np.arange(i,number),dest=b,tag=3)
                else:
                    #to avoid get to the end, set number-1
                    comm.send(np.arange(i,number-1),dest=b,tag=3)
            i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            total_histogram+=total_histogram_aux

        return total_histogram

    else:
        pos1=comm.recv(source=0,tag=7)
        pos2=comm.recv(source=0,tag=8)
        number=len(pos1)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            indexes=comm.recv(source=0,tag=3)
            if np.any(indexes%10000==0):
                print indexes[np.where(indexes%10000==0)[0]]

            if pos2!=None:
                for i in indexes:
                    posN=pos2
                    pos0=pos1[i]
                    
                    s=posN-pos0

                    #periodic conditions
                    outside1=np.where(s>BoxSize/2.0)
                    s[outside1]=s[outside1]-BoxSize

                    outside2=np.where(s<-BoxSize/2.0)
                    s[outside2]=s[outside2]+BoxSize

                    s2=np.sum(s**2,axis=1)
                    s=np.sqrt(s2)

                    hist=np.histogram(s,bins=bins_histo)[0]
                    total_histogram+=hist
            else:
                for i in indexes:
                    posN=pos1[i+1:number]
                    pos0=pos1[i]
                    
                    s=posN-pos0

                    #periodic conditions
                    outside1=np.where(s>BoxSize/2.0)
                    s[outside1]=s[outside1]-BoxSize

                    outside2=np.where(s<-BoxSize/2.0)
                    s[outside2]=s[outside2]+BoxSize

                    s2=np.sum(s**2,axis=1)
                    s=np.sqrt(s2)

                    hist=np.histogram(s,bins=bins_histo)[0]
                    total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)

        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################

####### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ####### (x,y,z) White-improved
################################################################################
def DDR_histogram3(myrank,nprocs,comm,bins,minimum,maximum,IL,BoxSize,indexes,dims,pos1,pos2):

    total_histogram=np.zeros(bins,dtype=np.int64)
    bins_histo=histogram_bins_log(minimum,maximum,bins)
    #bins_histo=histogram_bins(minimum,maximum,bins)

    #Master distributes the jobs
    if myrank==0:
        for i in range(1,nprocs):
            comm.send(pos1,dest=i,tag=7)
            comm.send(pos2,dest=i,tag=8)
            number=len(pos1)

        if pos2==None:
            for subbox in range(dims**3):
                b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
                comm.send(False,dest=b,tag=2)
                comm.send(subbox,dest=b,tag=3)
        else:
            i=0
            while i<number:
                b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
                comm.send(False,dest=b,tag=2)
                if i+IL<number:
                    #a=np.arange(0,10) -- a=array([0,1,2,3,4,5,6,7,8,9])
                    comm.send(np.arange(i,i+IL),dest=b,tag=4)
                else:
                    comm.send(np.arange(i,number),dest=b,tag=4)
                i+=IL

        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            total_histogram+=total_histogram_aux

        return total_histogram

    #slaves compute the pairs and return results to master
    else:
        pos1=comm.recv(source=0,tag=7)
        pos2=comm.recv(source=0,tag=8)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            if pos2==None:
                subbox=comm.recv(source=0,tag=3)
                core_ids=indexes[subbox] #ids of the particles in the subbox
                print subbox
            else:
                numbers=comm.recv(source=0,tag=4)
                if np.any(numbers%10000==0):
                    print numbers[np.where(numbers%10000==0)[0]]

            if pos2!=None:
                for i in numbers:
                    pos0=pos1[i]
                    #compute the ids of the particles in the neighboord subboxes
                    posN=pos2[indexes_subbox(pos0,maximum,dims,BoxSize,indexes)]
                    
                    s=posN-pos0

                    #periodic conditions
                    outside1=np.where(s>BoxSize/2.0)
                    s[outside1]=s[outside1]-BoxSize

                    outside2=np.where(s<-BoxSize/2.0)
                    s[outside2]=s[outside2]+BoxSize

                    s2=np.sum(s**2,axis=1)
                    s=np.sqrt(s2)

                    hist=np.histogram(s,bins=bins_histo)[0]
                    total_histogram+=hist
            else:
                length=len(core_ids)
                for i in range(length):
                    #first: compute the pairs in the subbox
                    pos0=pos1[core_ids[i]]
                    posN=pos1[core_ids[i+1:length]]
                    
                    s=posN-pos0

                    #periodic conditions
                    outside1=np.where(s>BoxSize/2.0)
                    s[outside1]=s[outside1]-BoxSize

                    outside2=np.where(s<-BoxSize/2.0)
                    s[outside2]=s[outside2]+BoxSize

                    s2=np.sum(s**2,axis=1)
                    s=np.sqrt(s2)

                    hist=np.histogram(s,bins=bins_histo)[0]
                    total_histogram+=hist

                for index in core_ids:
                    #second: compute the pairs of particles in the subbox with 
                    #particles in the neighboord subboxes
                    pos0=pos1[index]
                    ids=indexes_subbox_neigh(pos0,maximum,dims,BoxSize,indexes,subbox)
                    if ids!=[]:
                        posN=pos1[ids]

                        s=posN-pos0

                        #periodic conditions
                        outside1=np.where(s>BoxSize/2.0)
                        s[outside1]=s[outside1]-BoxSize

                        outside2=np.where(s<-BoxSize/2.0)
                        s[outside2]=s[outside2]+BoxSize
                        
                        s2=np.sum(s**2,axis=1)
                        s=np.sqrt(s2)

                        hist=np.histogram(s,bins=bins_histo)[0]
                        total_histogram+=hist

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)

        print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(total_histogram,dest=0,tag=10)
################################################################################
####### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ####### (x,y,z) very fast
################################################################################
def DDR_histogram4(myrank,nprocs,comm,bins,Rmin,Rmax,IL,BoxSize,dims,indexes,pos1,pos2):

    #we put bins+1 because the last bin is only when r=maximum
    total_histogram=np.zeros(bins+1,dtype=np.int64) 

    #Master sends the positions of the particles to the slaves
    if myrank==0:
        for i in range(1,nprocs):
            comm.send(pos1,dest=i,tag=7)
            comm.send(pos2,dest=i,tag=8)
            comm.send(indexes,dest=i,tag=9)
            number=len(pos1)

    #Masters distributes the calculation among slaves
        if pos2==None:
            for subbox in range(dims**3):
                b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
                comm.send(False,dest=b,tag=2)
                comm.send(subbox,dest=b,tag=3)
        else:
            i=0
            while i<number:
                b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
                comm.send(False,dest=b,tag=2)
                if i+IL<number:
                    #a=np.arange(0,10) -- a=array([0,1,2,3,4,5,6,7,8,9])
                    comm.send(np.arange(i,i+IL),dest=b,tag=4)
                else:
                    comm.send(np.arange(i,number),dest=b,tag=4)
                i+=IL

    #Master gathers the partial results from slaves and return the final result
        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            total_histogram_aux=comm.recv(source=b,tag=10)
            total_histogram+=total_histogram_aux

    #the last element is just for situations in which r=maximum
        total_histogram[bins-1]+=total_histogram[bins]

        return total_histogram[:-1]

    #slaves compute the pairs and return results to master
    else:
        #we put bins+1 because the last bin is only when r=maximum
        total_histogram=np.zeros(bins+1,dtype=np.int64) 

        #slaves receive the positions
        pos1=comm.recv(source=0,tag=7)
        pos2=comm.recv(source=0,tag=8)
        indexes=comm.recv(source=0,tag=9)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            if pos2==None:
                subbox=comm.recv(source=0,tag=3)
                core_ids=indexes[subbox] #ids of the particles in the subbox
                print subbox

                distances_core(pos1[core_ids],BoxSize,bins,Rmin,Rmax,total_histogram)

                for index in core_ids:
                    #second: compute the pairs of particles in the subbox with 
                    #particles in the neighboord subboxes
                    pos0=pos1[index]
                    ids=indexes_subbox_neigh(pos0,Rmax,dims,BoxSize,indexes,subbox)
                    if ids!=[]:
                        posN=pos1[ids]

                        distances(pos0,posN,BoxSize,bins,Rmin,Rmax,total_histogram)        

            else:
                numbers=comm.recv(source=0,tag=4)
                if np.any(numbers%10000==0):
                    print numbers[np.where(numbers%10000==0)[0]]

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
#pos1---a single position
#pos2---an array of positions
#the function returns the histogram of the computed distances between 
#pos1 and pos2
def distances(pos1,pos2,BoxSize,bins,Rmin,Rmax,histogram):

    x=pos2[:,0]
    y=pos2[:,1]
    z=pos2[:,2]

    support = """
            #include <iostream>"
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
#this function computes the distances between all the particles-pairs and
#return the result in the histogram
def distances_core(pos,BoxSize,bins,Rmin,Rmax,histogram):
    x=pos[:,0]
    y=pos[:,1]
    z=pos[:,2]

    support = """
            #include <iostream>"
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
