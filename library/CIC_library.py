#version 2.1

#changes wrt to version 2.0: Added CIC_serial_2D to compute the CIC on 
#2 dimensions. Added the posibility to compile the library by typing:
#python CIC_library.py

#changes wrt to version 1.1: Now the routine computes the densities in each 
#grid cell. Before it computed the overdensity by dividing the densities by the
#mean density. When having more that one particle type, and it is wanted to 
#compute the (over)density of the whole field, this routine is more appropiate
#and clean.

#########################

#Library to compute the density of a point-set distribution using the 
#CIC interpolation technique (there is also a rutine to compute it using the 
#nearest grid point (NGP) technique)

#If the library needs to be compiled type: python CIC_library.py compile

#IMPORTANT!! If the c/c++ functions need to be modified, the code has to be
#compiled by calling those functions within this file, otherwise it gives errors

import numpy as np
import scipy.weave as wv
import sys

################################################################################
#This function computes the contribution of the particle set to each of the 
#grid cells. The grid consists on dims x dims x dims cells.
#The input is positions, which is an array of shape (N,3), with N the number
#of particles, and cic_densities: an array of dimensions dims**3. This array
#can be set to 0 for the first time or can have the values of a previous 
#computation (this is useful when there are several particle types)
#For reasons regarding weave, the size of the positions array, N, has to be 
#smaller than 2^31/3. For that reason, if the array is larger, we split it into
#several pieces to guaranty that requirement

#In order to avoid problems with particles at the edge x==BoxSize or y==BoxSize
#or z==BoxSize, we first compute x[i]=dims*1.0 (note that this coordinate goes
#from 0 to dims-1, i.e. it cant take a value equal to dims). 
#This particle will contribute to the cell 0, not dims nor dims-1. 
#We first compute fx[i]==dims, and thus dx[i]=0.0 and tx[i]=1.0
#Once this is computed we recalculate fx[i] by fx[i]%dims, and thus, fx[i]=0
#and dx[i]=1

def CIC_serial(positions,dims,BoxSize,cic_densities,weights=None):
    n_max=850**3 #maximum number of elements weave can deal with

    units=dims*1.0/BoxSize
    total_siz=positions.shape[0]

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

             cont[0]=tx[0]*tx[1]*tx[2];
	     cont[1]=dx[0]*tx[1]*tx[2];
	     cont[2]=tx[0]*dx[1]*tx[2];
	     cont[3]=dx[0]*dx[1]*tx[2];
	     cont[4]=tx[0]*tx[1]*dx[2];
	     cont[5]=dx[0]*tx[1]*dx[2];
	     cont[6]=tx[0]*dx[1]*dx[2];
	     cont[7]=dx[0]*dx[1]*dx[2];

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

             cont[0]=tx[0]*tx[1]*tx[2]*wg(n);
	     cont[1]=dx[0]*tx[1]*tx[2]*wg(n);
	     cont[2]=tx[0]*dx[1]*tx[2]*wg(n);
	     cont[3]=dx[0]*dx[1]*tx[2]*wg(n);
	     cont[4]=tx[0]*tx[1]*dx[2]*wg(n);
	     cont[5]=dx[0]*tx[1]*dx[2]*wg(n);
	     cont[6]=tx[0]*dx[1]*dx[2]*wg(n);
	     cont[7]=dx[0]*dx[1]*dx[2]*wg(n);
	     
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
            print total_siz,weights.shape[0]
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
            wv.inline(code,['pos','units','siz','dims','cic_densities'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m'],
                      extra_compile_args =['-O3'],)
        else:
            wg=weights[start:end]
            wv.inline(code_w,['pos','units','siz','dims','cic_densities','wg'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m'],
                      extra_compile_args =['-O3'],)

        start=end


    return cic_densities
################################################################################
#This function computes the CIC interpolated values of a distribution
#of points in a plane. The above function is a for 3-dimensional distribution
#of points

def CIC_serial_2D(positions,dims,BoxSize,cic_densities,weights=None):
    n_max=850**3 #maximum number of elements weave can deal with

    units=dims*1.0/BoxSize
    total_siz=positions.shape[0]

    support = """
         #include <math.h>
    """
    code = """
         float dx[2],tx[2],x[2],cont[4];
         int fx[2],nx[2],index[4];
         int i,j;

         for (int n=0;n<siz;n++){

             for (i=0;i<2;i++){
                x[i]=pos(n,i)*units;
                fx[i]=(int)floor(x[i]);
                dx[i]=x[i]-fx[i];
                tx[i]=1.0-dx[i];
                fx[i]=fx[i]%dims;
                nx[i]=(fx[i]+1)%dims;
             } 

             cont[0]=tx[0]*tx[1];
	     cont[1]=dx[0]*tx[1];
	     cont[2]=tx[0]*dx[1];
	     cont[3]=dx[0]*dx[1];

             index[0]=dims*fx[0] + fx[1];
	     index[1]=dims*nx[0] + fx[1];
	     index[2]=dims*fx[0] + nx[1];
	     index[3]=dims*nx[0] + nx[1];

             for (j=0;j<4;j++)
                 cic_densities(index[j])+=cont[j];
             
         }
    """
    code_w = """
         float dx[2],tx[2],x[2],cont[4];
         int fx[2],nx[2],index[4];
         int i,j;

         for (int n=0;n<siz;n++){

             for (i=0;i<2;i++){
                x[i]=pos(n,i)*units;
                fx[i]=(int)floor(x[i]);
                dx[i]=x[i]-fx[i];
                tx[i]=1.0-dx[i];
                fx[i]=fx[i]%dims;
                nx[i]=(fx[i]+1)%dims;
             }

             cont[0]=tx[0]*tx[1]*wg(n);
	     cont[1]=dx[0]*tx[1]*wg(n);
	     cont[2]=tx[0]*dx[1]*wg(n);
	     cont[3]=dx[0]*dx[1]*wg(n);
	     
             index[0]=dims*fx[0] + fx[1];
	     index[1]=dims*nx[0] + fx[1];
	     index[2]=dims*fx[0] + nx[1];
	     index[3]=dims*nx[0] + nx[1];

             for (j=0;j<4;j++)
                 cic_densities(index[j])+=cont[j];
         }
    """

    #check that the sizes of the positions and the weights are the same
    if weights!=None:
        if total_siz!=weights.shape[0]:
            print 'the sizes of the positions and weights are not the same'
            print total_siz,weights.shape[0]
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
            wv.inline(code,['pos','units','siz','dims','cic_densities'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m'],
                      extra_compile_args =['-O3'],)
        else:
            wg=weights[start:end]
            wv.inline(code_w,['pos','units','siz','dims','cic_densities','wg'],
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

                cont[n][0]=tx[0]*tx[1]*tx[2];
       	        cont[n][1]=dx[0]*tx[1]*tx[2];
	        cont[n][2]=tx[0]*dx[1]*tx[2];
	        cont[n][3]=dx[0]*dx[1]*tx[2];
	        cont[n][4]=tx[0]*tx[1]*dx[2];
	        cont[n][5]=dx[0]*tx[1]*dx[2];
	        cont[n][6]=tx[0]*dx[1]*dx[2];
	        cont[n][7]=dx[0]*dx[1]*dx[2];

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
                  ['threads','pos','length','units','dims','cic_densities'],
                  extra_compile_args=['-O3 -fopenmp'],
                  extra_link_args=['-lgomp'],
                  type_converters = wv.converters.blitz,
                  support_code = support,libraries = ['m','gomp'])
        start=end

    return cic_densities
################################################################################

#positions: array with the positions of the particles
#velocity1: array with the velocities of the particles along one direction
#velocity2: array with the velocities of the particles along one direction
#dims: number of mesh points in one direction
#BoxSize: Size of the simulation box in the same units as positions
#cic_sigma: dims^3 array where the result will be stored
#cic_vel1:  dims^3 array containing the cic velocities along one direction
#cic_vel2:  dims^3 array containing the cic velocities along one direction
def CIC_sigma(positions,velocity1,velocity2,dims,BoxSize,
              cic_sigma,cic_vel1,cic_vel2):
    n_max=850**3 #maximum number of elements weave can deal with

    units=dims*1.0/BoxSize
    total_siz=positions.shape[0]
    dims3=dims**3

    support = """
         #include <math.h>
    """
    code = """
         int dims2=dims*dims;
         float dx[3],tx[3],x[3],W[8],V1,V2;
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

             /* compute the weigths of each particle into its neighbours */
             W[0]=tx[0]*tx[1]*tx[2];
	     W[1]=dx[0]*tx[1]*tx[2];
	     W[2]=tx[0]*dx[1]*tx[2];
	     W[3]=dx[0]*dx[1]*tx[2];
	     W[4]=tx[0]*tx[1]*dx[2];
	     W[5]=dx[0]*tx[1]*dx[2];
	     W[6]=tx[0]*dx[1]*dx[2];
	     W[7]=dx[0]*dx[1]*dx[2];
	     
             /* the particle gives contributions to the neighbour mesh point
                the index array contains the positions of those mesh poins */
             index[0]=dims2*fx[0] + dims*fx[1] + fx[2];
	     index[1]=dims2*nx[0] + dims*fx[1] + fx[2];
	     index[2]=dims2*fx[0] + dims*nx[1] + fx[2];
	     index[3]=dims2*nx[0] + dims*nx[1] + fx[2];
	     index[4]=dims2*fx[0] + dims*fx[1] + nx[2];
	     index[5]=dims2*nx[0] + dims*fx[1] + nx[2];
	     index[6]=dims2*fx[0] + dims*nx[1] + nx[2];
	     index[7]=dims2*nx[0] + dims*nx[1] + nx[2];

             /* V1 and V2 are the cic weighted values of the velocity field
                along axis 1 and 2. Those values can be computed through
                the CIC_serial code */
             for (j=0;j<8;j++){
                 V1=cic_vel1(index[j]);
                 V2=cic_vel2(index[j]);
                 cic_sigma(index[j])+=W[j]*(vel1(n)-V1)*(vel2(n)-V2);
             }
         }
    """

    #some checks before starting the computation
    if len(positions)!=len(velocity1) or len(positions)!=len(velocity2) or len(velocity1)!=len(velocity2):
        print 'sizes of positions and/or velocities different'
        print len(positions),len(velocity1),len(velocity2)
        sys.exit()

    if len(cic_vel1)!=dims3 or len(cic_vel2)!=dims3 or len(cic_sigma)!=dims3:
        print 'the sizes of the cic array must be equal to dims^3'
        print len(cic_vel1),len(cic_vel2),len(cic_sigma)
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
        vel1=velocity1[start:end]
        vel2=velocity2[start:end]
        siz=pos.shape[0]

        wv.inline(code,['pos','vel1','vel2','units','siz','dims','cic_sigma',
                        'cic_vel1','cic_vel2'],
                  type_converters = wv.converters.blitz,                  
                  verbose=2,support_code = support,libraries = ['m'],
                  extra_compile_args =['-O3'])

        start=end

    return cic_sigma
################################################################################

def NGP_serial(positions,dims,BoxSize,ngp_densities,weights=None):
    n_max=850**3 #maximum number of elements weave can deal with

    units=dims*1.0/BoxSize
    total_siz=positions.shape[0]

    support = """
         #include <math.h>
    """
    code = """
         int dims2=dims*dims;
         int i,index,coord[3];
         float dumb,inv_units=1.0/units;

         for (int n=0;n<siz;n++){
            for (i=0;i<3;i++){
                dumb=pos(n,i)+0.5*inv_units;
                coord[i]=((int)(dumb*units))%dims;
            }

            index=dims2*coord[0] + dims*coord[1] +coord[2];
            ngp_densities(index)+=1.0;
         }
    """
    code_w = """
         int dims2=dims*dims;
         int i,index,coord[3];
         float dumb,inv_units=1.0/units;

         for (int n=0;n<siz;n++){
            for (i=0;i<3;i++){
                dumb=pos(n,i)+0.5*inv_units;
                coord[i]=((int)(dumb*units))%dims;
            }

            index=dims2*coord[0] + dims*coord[1] +coord[2];
            ngp_densities(index)+=wg(n);
         }
    """

    #check that the sizes of the positions and the weights are the same
    if weights!=None:
        if total_siz!=weights.shape[0]:
            print 'the sizes of the positions and weights are not the same'
            print total_siz,weights.shape[0]
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
            wv.inline(code,['pos','units','siz','dims','ngp_densities'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m'],
                      extra_compile_args =['-O3'],)
        else:
            wg=weights[start:end]
            wv.inline(code_w,['pos','units','siz','dims','ngp_densities','wg'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m'],
                      extra_compile_args =['-O3'],)

        start=end


    return ngp_densities
################################################################################


############################### EXAMPLE OF USAGE ###############################
if len(sys.argv)==2:
    if sys.argv[1]=='compile':

#########################################################################
        ### CIC_serial without weights ###
        #computes the density in each grid cell and later can be computed the
        #overdensity or the deltas

        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        np.random.seed(seed=1)
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        print pos

        cic_overdensities=np.zeros(dims**3,dtype=np.float32) 
        CIC_serial(pos,dims,BoxSize,cic_overdensities) #computes densities
        cic_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(cic_overdensities,dtype=np.float64)
        print cic_overdensities
        print np.min(cic_overdensities),np.max(cic_overdensities)

#########################################################################
        ### CIC_serial with weights ###

        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        #pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        #print pos

        weights=np.ones(n,dtype=np.float32)
        print weights
        cic_overdensities=np.zeros(dims**3,dtype=np.float32)
        CIC_serial(pos,dims,BoxSize,cic_overdensities,weights)#compute densities
        cic_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(cic_overdensities,dtype=np.float64)
        print cic_overdensities
        print np.min(cic_overdensities),np.max(cic_overdensities)

#########################################################################
        ### CIC_openmp 

        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        threads=8

        #pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        #print pos

        cic_overdensities=np.zeros(dims**3,dtype=np.float32)
        CIC_openmp(pos,dims,BoxSize,threads,cic_overdensities)#compute densities
        cic_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(cic_overdensities,dtype=np.float64)
        print cic_overdensities
        print np.min(cic_overdensities),np.max(cic_overdensities)

#########################################################################
        ### CIC_sigma

        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128
        
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        print pos

        weights=np.ones(n,dtype=np.float32)
        print weights
        cic_overdensities=np.zeros(dims**3,dtype=np.float32)
        CIC_serial(pos,dims,BoxSize,cic_overdensities,weights)#compute densities
        cic_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(cic_overdensities,dtype=np.float64)
        print cic_overdensities
        print np.min(cic_overdensities),np.max(cic_overdensities)

        cic_sigma=np.zeros(dims**3,dtype=np.float32)
        CIC_sigma(pos,weights,weights,dims,BoxSize,
                  cic_sigma,cic_overdensities,cic_overdensities)
        print cic_sigma

#########################################################################
        ### NGP_serial without weights ###
        #computes the density in each grid cell and later can be computed the
        #overdensity or the deltas

        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        np.random.seed(seed=1)
        pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        print pos

        ngp_overdensities=np.zeros(dims**3,dtype=np.float32) 
        NGP_serial(pos,dims,BoxSize,ngp_overdensities) #computes densities
        ngp_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(ngp_overdensities,dtype=np.float64)
        print ngp_overdensities
        print np.min(ngp_overdensities),np.max(ngp_overdensities)

#########################################################################
        ### NGP_serial with weights ###

        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        #pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        #print pos

        weights=np.ones(n,dtype=np.float32)
        print weights
        ngp_overdensities=np.zeros(dims**3,dtype=np.float32)
        NGP_serial(pos,dims,BoxSize,ngp_overdensities,weights)#compute densities
        ngp_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(ngp_overdensities,dtype=np.float64)
        print ngp_overdensities
        print np.min(ngp_overdensities),np.max(ngp_overdensities)

#########################################################################
        ### CIC_serial_2D without weights ###
        #computes the density (in 2D) in each grid cell and later can 
        #be computed the overdensity or the deltas

        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        np.random.seed(seed=1)
        pos=(np.random.random((n,2))*BoxSize).astype(np.float32)
        print pos

        cic_overdensities=np.zeros(dims**2,dtype=np.float32) 
        CIC_serial_2D(pos,dims,BoxSize,cic_overdensities) #computes densities
        cic_overdensities*=dims**2*1.0/n #divide by mean to obtain overdensities

        print np.sum(cic_overdensities,dtype=np.float64)
        print cic_overdensities
        print np.min(cic_overdensities),np.max(cic_overdensities)

#########################################################################
        ### CIC_serial_2D with weights ###
        
        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        #pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        #print pos

        weights=np.ones(n,dtype=np.float32)
        print weights
        cic_overdensities=np.zeros(dims**2,dtype=np.float32)
        #compute densities
        CIC_serial_2D(pos,dims,BoxSize,cic_overdensities,weights)
        cic_overdensities*=dims**2*1.0/n #divide by mean to obtain overdensities

        print np.sum(cic_overdensities,dtype=np.float64)
        print cic_overdensities
        print np.min(cic_overdensities),np.max(cic_overdensities)

#########################################################################

    else:
        print 'To compile the code type:'
        print 'python CIC_library.py compile'
