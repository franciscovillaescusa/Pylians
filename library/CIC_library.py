#version 1.1
#Library to compute the overdensity of a point-set distribution using the 
#CIC interpolation technique

#IMPORTANT!! If the functions need to be modified, the code has to be compiled
#by calling the function within this file, otherwise it gives errors

import numpy as np
import scipy.weave as wv
import sys

#################################################################################
#This function computes the overdensity, within a cubic box of size BoxSize,
#in each point of a grid of dimensions dims x dims x dims
#The input is positions, which is an array of shape (N,3), with N the number
#of particles, and cic_densities: an array of dimensions dims**3. This array
#can be set to 0 for the first time or can have the values of a previous 
#computation (this is useful when there are several particle types)
#For reasons regarding weave, the size of the positions array, N, has to be 
#smaller than 2^31/3. For that reason, if the array is larger, we split it into
#several pieces to guaranty that requirement

#In order to avoid problems with particles at the edge x==BoxSize or y==BoxSize
#or z==BoxSize, we first compute x[i]=dims*1.0 (note that this coordenate goes
#from 0 to dims-1, i.e. it cant take a value equal to dims). 
#This particle will contribute to the cell 0, not dims or dims-1. 
#We first compute fx[i]==dims, and thus dx[i]=0.0 and tx[i]=1.0
#Once this is computed we recalculate fx[i] by fx[i]%dims, and thus, fx[i]=0
#and dx[i]=1

def CIC_serial(positions,dims,BoxSize,cic_densities,weights=None):
    n_max=850**3 #maximum number of elements weave can deal with

    units=dims*1.0/BoxSize
    total_siz=positions.shape[0]
    invrho=dims**3*1.0/total_siz

    support = """
         #include <math.h>
    """
    code = """
         int dims2=dims*dims;
         float dx[3],tx[3],x[3],cont[8];
         int fx[3],nx[3],index[8];
         int i,j;

         for (int n=0;n<siz;n++){

             for (i=0;i<3;i++){
                x[i]=pos(n,i)*units;
                fx[i]=(int)floor(x[i]);
                dx[i]=x[i]-fx[i];
                tx[i]=1.0-dx[i];
                fx[i]=fx[i]%dims;
                nx[i]=(fx[i]+1)%dims;
             } 

             cont[0]=invrho*tx[0]*tx[1]*tx[2];
	     cont[1]=invrho*dx[0]*tx[1]*tx[2];
	     cont[2]=invrho*tx[0]*dx[1]*tx[2];
	     cont[3]=invrho*dx[0]*dx[1]*tx[2];
	     cont[4]=invrho*tx[0]*tx[1]*dx[2];
	     cont[5]=invrho*dx[0]*tx[1]*dx[2];
	     cont[6]=invrho*tx[0]*dx[1]*dx[2];
	     cont[7]=invrho*dx[0]*dx[1]*dx[2];

             index[0]=dims2*fx[0] + dims*fx[1] + fx[2];
	     index[1]=dims2*nx[0] + dims*fx[1] + fx[2];
	     index[2]=dims2*fx[0] + dims*nx[1] + fx[2];
	     index[3]=dims2*nx[0] + dims*nx[1] + fx[2];
	     index[4]=dims2*fx[0] + dims*fx[1] + nx[2];
	     index[5]=dims2*nx[0] + dims*fx[1] + nx[2];
	     index[6]=dims2*fx[0] + dims*nx[1] + nx[2];
	     index[7]=dims2*nx[0] + dims*nx[1] + nx[2];

             for (j=0;j<8;j++)
                 cic_densities(index[j])+=cont[j];
             
         }
    """
    code_w = """
         int dims2=dims*dims;
         float dx[3],tx[3],x[3],cont[8];
         int fx[3],nx[3],index[8];
         int i,j;

         for (int n=0;n<siz;n++){

             for (i=0;i<3;i++){
                x[i]=pos(n,i)*units;
                fx[i]=(int)floor(x[i]);
                dx[i]=x[i]-fx[i];
                tx[i]=1.0-dx[i];
                fx[i]=fx[i]%dims;
                nx[i]=(fx[i]+1)%dims;
             }

             cont[0]=invrho*tx[0]*tx[1]*tx[2]*wg(n);
	     cont[1]=invrho*dx[0]*tx[1]*tx[2]*wg(n);
	     cont[2]=invrho*tx[0]*dx[1]*tx[2]*wg(n);
	     cont[3]=invrho*dx[0]*dx[1]*tx[2]*wg(n);
	     cont[4]=invrho*tx[0]*tx[1]*dx[2]*wg(n);
	     cont[5]=invrho*dx[0]*tx[1]*dx[2]*wg(n);
	     cont[6]=invrho*tx[0]*dx[1]*dx[2]*wg(n);
	     cont[7]=invrho*dx[0]*dx[1]*dx[2]*wg(n);
	     
             index[0]=dims2*fx[0] + dims*fx[1] + fx[2];
	     index[1]=dims2*nx[0] + dims*fx[1] + fx[2];
	     index[2]=dims2*fx[0] + dims*nx[1] + fx[2];
	     index[3]=dims2*nx[0] + dims*nx[1] + fx[2];
	     index[4]=dims2*fx[0] + dims*fx[1] + nx[2];
	     index[5]=dims2*nx[0] + dims*fx[1] + nx[2];
	     index[6]=dims2*fx[0] + dims*nx[1] + nx[2];
	     index[7]=dims2*nx[0] + dims*nx[1] + nx[2];

             for (j=0;j<8;j++)
                 cic_densities(index[j])+=cont[j];
         }
    """

    #check that the sizes of the positions and the weights are the same
    if weights!=None:
        if total_siz!=weights.shape[0]:
            print 'the sizes of the positions and weights are not the same'
            sys.exit()

    #if the array to be sent is larger than n_max, split it into smaller pieces
    start=0; final=False
    while not(final):

        if start+n_max>total_siz:
            end=total_siz; final=True
        else:
            end=start+n_max

        print start,'--',end
        pos=positions[start:end]
        siz=pos.shape[0]

        if weights==None:
            wv.inline(code,
                      ['pos','units','siz','dims','cic_densities','invrho'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m'],
                      extra_compile_args =['-O3'],)
        else:
            wg=weights[start:end]
            wv.inline(code_w,
                      ['pos','units','siz','dims','cic_densities','invrho','wg'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m'],
                      extra_compile_args =['-O3'],)

        start=end


    return cic_densities
################################################################################

def CIC_openmp(positions,dims,BoxSize,threads,cic_densities):
    n_max=850**3

    units=dims*1.0/BoxSize
    total_length=positions.shape[0]
    invrho=dims**3*1.0/total_length
    
    support = """
         #include<math.h>
         #include<omp.h>
         #define IL 1000
    """       

    code = """
         omp_set_num_threads(threads);
         int dims2=dims*dims;
         float dx[3],tx[3],x[3],cont[IL][8];
         int fx[3],nx[3],index[IL][8];
         int n,i,j,final;

         #pragma omp parallel for private(dx,tx,x,cont,fx,nx,index,n,i,j,final) shared(cic_densities) 
         for (int l=0;l<length;l+=IL){

            final = (l+IL<length) ? IL : length-l;
            for (n=0;n<final;n++)
            {
                for (i=0;i<3;i++){
                   x[i]=pos(n+l,i)*units;
                   fx[i]=(int)floor(x[i]);
                   dx[i]=x[i]-fx[i];
                   tx[i]=1.0-dx[i];
                   fx[i]=fx[i]%dims;
                   nx[i]=(fx[i]+1)%dims;
                }

                cont[n][0]=invrho*tx[0]*tx[1]*tx[2];
       	        cont[n][1]=invrho*dx[0]*tx[1]*tx[2];
	        cont[n][2]=invrho*tx[0]*dx[1]*tx[2];
	        cont[n][3]=invrho*dx[0]*dx[1]*tx[2];
	        cont[n][4]=invrho*tx[0]*tx[1]*dx[2];
	        cont[n][5]=invrho*dx[0]*tx[1]*dx[2];
	        cont[n][6]=invrho*tx[0]*dx[1]*dx[2];
	        cont[n][7]=invrho*dx[0]*dx[1]*dx[2];

                index[n][0]=dims2*fx[0] + dims*fx[1] + fx[2];
  	        index[n][1]=dims2*nx[0] + dims*fx[1] + fx[2];
	        index[n][2]=dims2*fx[0] + dims*nx[1] + fx[2];
	        index[n][3]=dims2*nx[0] + dims*nx[1] + fx[2];
	        index[n][4]=dims2*fx[0] + dims*fx[1] + nx[2];
	        index[n][5]=dims2*nx[0] + dims*fx[1] + nx[2];
	        index[n][6]=dims2*fx[0] + dims*nx[1] + nx[2];
	        index[n][7]=dims2*nx[0] + dims*nx[1] + nx[2];
            }

            #pragma omp critical
            {
            for (n=0;n<final;n++)
               for (j=0;j<8;j++)
                   cic_densities(index[n][j])+=cont[n][j];
            }
         }
    """ 

    start=0; final=False
    while not(final):

        if start+n_max>total_length:
            end=total_length; final=True
        else:
            end=start+n_max

        print start,'--',end
        pos=positions[start:end]
        length=pos.shape[0]
        wv.inline(code,
                  ['threads','pos','length','units','dims','cic_densities','invrho'],extra_compile_args =['-O3 -fopenmp'],
                  extra_link_args=['-lgomp'],
                  type_converters = wv.converters.blitz,
                  support_code = support,libraries = ['m','gomp'])
        start=end

    return cic_densities




########################## EXAMPLE OF USAGE #########################
### CIC_serial without weights ###
"""
n=512**3
BoxSize=500.0 #Mpc/h
dims=512

np.random.seed(seed=1)
pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
print pos

cic_densities=np.zeros(dims**3,dtype=np.float32)
CIC_serial(pos,dims,BoxSize,cic_densities)

print np.sum(cic_densities,dtype=np.float64)
print cic_densities
print np.min(cic_densities),np.max(cic_densities)
"""


### CIC_serial with weights ###
"""
n=512**3
BoxSize=500.0 #Mpc/h
dims=512

#pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
#print pos

weights=np.ones(dims**3,dtype=np.float32)
print weights
cic_densities=np.zeros(dims**3,dtype=np.float32)
CIC_serial(pos,dims,BoxSize,cic_densities,weights)

print np.sum(cic_densities,dtype=np.float64)
print cic_densities
print np.min(cic_densities),np.max(cic_densities)
"""

### CIC_openmp 
"""
n=512**3
BoxSize=500.0 #Mpc/h
dims=512

threads=8

#pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
#print pos

cic_densities=np.zeros(dims**3,dtype=np.float32)
CIC_openmp(pos,dims,BoxSize,threads,cic_densities)

print np.sum(cic_densities,dtype=np.float64)
print cic_densities
print np.min(cic_densities),np.max(cic_densities)
"""
