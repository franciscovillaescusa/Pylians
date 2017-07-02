import numpy as np
import sys
import os


#routine used to read the indexes of the particles sorted by the routine grid_distribution.py
#it reads the file/s and create a list of dims**3 elements. Each element contains an array with the indexes of the particles which belong to the given subbox
#for large data sets (i.e. 2048**3 particles), the data_type should be np.uint64 instead of np.uint32
#USAGE: sorted_indexes(filename), where filename is the root. For example for the files data_20.0 data_20.1 data_20.2.... just use sorted_indexes(data_20)
def sorted_indexes(filename,data_type=np.uint32):

    if os.path.exists(filename):
        curfilename = filename
    elif os.path.exists(filename+".0"):
        curfilename = filename+".0"
    else:
        print "file not found:", filename
        sys.exit()

    f=open(curfilename,'rb')
    number_of_files=np.fromfile(f,dtype=np.uint32,count=1)[0]
    dims=np.fromfile(f,dtype=np.uint32,count=1)[0]
    dims3=dims**3
    total_size=np.fromfile(f,dtype=data_type,count=dims3)
    total_array=[]
    for j in range(dims3):
        total_array.append(np.empty(total_size[j],dtype=data_type))
    f.close()
    total_array=np.array(total_array)

    offset=np.zeros(dims3,dtype=data_type)
    for i in range(number_of_files):
        curfilename=filename+'.'+str(i)
        f=open(curfilename,'rb')
        f.seek(4*(2+dims3),os.SEEK_CUR)
        for j in range(dims3):
            size=np.fromfile(f,dtype=data_type,count=1)[0]
            array=np.fromfile(f,dtype=data_type,count=size)
            total_array[j][offset[j]:offset[j]+size]=array
            offset[j]+=size
        f.close()

    return total_array
        
