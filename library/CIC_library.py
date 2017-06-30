#version 2.1

#changes wrt to version 2.0: Added CIC_serial_2D to compute the CIC on 
#2 dimensions. Added the posibility to compile the library by typing:
#python CIC_library.py

#changes wrt to version 1.1: Now the routine computes the densities in each 
#grid cell. Before it computed the overdensity by dividing the densities by the
#mean density. When having more that one particle type, and it is wanted to 
#compute the (over)density of the whole field, this routine is more appropiate
#and clean.

############ AVAILABLE ROUTINES ###########
#CIC_serial
#CIC_serial_2D
#CIC_openmp
#CIC_sigma
#NGP_serial
#TSC_serial
#SPH_gas
#   Volw
##########################################

#Library to compute the density of a point-set distribution using the 
#NGP,CIC and TSC interpolation techniques
#It also contains a routine to compute the density field when there are gas
#particles (having SPH smoothing lengths)

######## COMPILATION ##########
#If the library needs to be compiled type: python CIC_library.py compile
###############################

#IMPORTANT!! If the c/c++ functions need to be modified, the code has to be
#compiled by calling those functions within this file, otherwise it gives errors

import numpy as np
#import scipy.weave as wv
import weave as wv
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
#This routine computes the values of the density on each point of a regular
#grid using the TSC interpolation scheme
def TSC_serial(positions,dims,BoxSize,cic_densities,weights=None):
    n_max=850**3 #maximum number of elements weave can deal with

    units=dims*1.0/BoxSize
    total_siz=positions.shape[0]

    support = """
         #include <math.h>
    """
    code = """
         int dims2=dims*dims;
         float x,W[3][4],cont,diff;
         int num[3][4],index,point;
         int xmin,i,j,k;

         for (int n=0;n<siz;n++){

             for (i=0;i<3;i++){
                x=pos(n,i)*units;
                xmin=(int)floor(x-1.5+0.5);

                for (k=0;k<4;k++){
                   point=xmin+k;
                   num[i][k]=(point+dims)%dims;

                   diff=fabs(point-x);
                   if (diff<0.5){
                       W[i][k]=0.75-diff*diff;
                   }
                   else{
                      if (diff<1.5){
                          W[i][k]=(1.5-diff)*(1.5-diff)/2.0;
                      }
                      else{
                          W[i][k]=0.0; 
                      }
                   }
                }
             }

             for (i=0;i<4;i++){
                for (j=0;j<4;j++){
                   for (k=0;k<4;k++){
                       index=dims*(dims*num[0][i]+num[1][j])+num[2][k];
                       cont=W[0][i]*W[1][j]*W[2][k];
                       cic_densities(index)+=cont;
                   }
                }
             }

         }    
    """
    code_w = """
         int dims2=dims*dims;
         float x,W[3][4],cont,diff;
         int num[3][4],index,point;
         int xmin,i,j,k;

         for (int n=0;n<siz;n++){

             for (i=0;i<3;i++){
                x=pos(n,i)*units;
                xmin=(int)floor(x-1.5+0.5);

                for (k=0;k<4;k++){
                   point=xmin+k;
                   num[i][k]=(point+dims)%dims;

                   diff=fabs(point-x);
                   if (diff<0.5){
                       W[i][k]=0.75-diff*diff;
                   }
                   else{
                      if (diff<1.5){
                          W[i][k]=(1.5-diff)*(1.5-diff)/2.0;
                      }
                      else{
                          W[i][k]=0.0;
                      }
                   }
                }
             }

             for (i=0;i<4;i++){
                for (j=0;j<4;j++){
                   for (k=0;k<4;k++){
                       index=dims2*num[0][i]+dims*num[1][j]+num[2][k];
                       cont=W[0][i]*W[1][j]*W[2][k]*wg(n); 
                       cic_densities(index)+=cont;
                   }
                }
             }

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

#This routine is used to compute the values of the density in a cubic regular
#grid when the field is made up by gas particles, which have an associated
#SPH radius. We generate a distribution of points sampling uniformiuosly the
#interior volume of a sphere of radius 1 and then for each particle the code
#computes in which grid cell each point (for each particle) lies.
#positions -------> positions of the particles
#radii -----------> SPH smoothing lengths of the particles (same units as pos)
#divisions -------> a sphere of radius 1 will be divided into divisions^3 points
#dims ------------> number of grid cells per axis
#BoxSize ---------> size of the simulation box (in the same units as pos)
#threads ---------> number of openmp threads to use
#densities -------> array to be filled with the values of the density
#weights ---------> weights associated to the gas particles
#If the density field of the neutral hydrogen is wanted then use the positions
#and radii of the gas particles but use as weights the HI masses of the gas
def SPH_gas(positions,radii,divisions,dims,BoxSize,threads,
            densities,weights=None):
                       
    n_max=850**3 #maximum number of elements weave can deal with
    units=np.array([dims*1.0/BoxSize]); total_siz=positions.shape[0]; pi=np.pi

    #check that the number of elements in the positions and radii are the same
    if len(positions)!=len(radii):
        print 'number of elements in the positions and radii are different!!!'
        sys.exit()

    ######### select volume_divisions x theta_divisions x phi_divisions #######
    ####### points equally spaced (in volume) within a sphere of radius 1 #####
    volume_divisions=theta_divisions=phi_divisions=divisions
    sphere_points=volume_divisions*theta_divisions*phi_divisions

    #Divide a sphere of radius 1 into spherical shells of the same volume,
    #taking into account the SPH density profile
    V_shell=1.0/volume_divisions; R0=0.0; Radii_sphere=[R0]
    for i in xrange(volume_divisions):
        #compute the radii of the spherical shell: 
        #R1=(3.0*V_shell/(4.0*pi)+R0**3)**(1.0/3.0); Radii.append(R1); R0=R1
        Final=False; Rmin=R0; Rmax=1.0; Vol0=Volw(R0); tol=1e-5
        while not(Final):
            test_R=0.5*(Rmin+Rmax)
            Vol1=Volw(test_R); rel_diff=((Vol1-Vol0) - V_shell)/V_shell

            if np.absolute(rel_diff)<tol:
                print 'R = [%f,%f] ---> Vol = %f'%(test_R,R0,Vol1-Vol0)
                Final=True; R1=test_R; R0=R1; Radii_sphere.append(R1)
            else:
                if rel_diff<0.0:
                    Rmin=test_R
                else:
                    Rmax=test_R
    Radii_sphere=np.array(Radii_sphere)
    Radii_sphere=0.5*(Radii_sphere[1:]+Radii_sphere[:-1])

    #Divide a sphere of radius 1 into slices in theta of the same area
    A_slice=4.0*pi/theta_divisions; theta0=0.0; thetas=[theta0]
    for i in xrange(theta_divisions):
        #compute the theta using 2*pi*(cos(theta0)-cos(theta1)) = A_slice
        argument=np.max([-1.0,np.cos(theta0)-A_slice/(2.0*pi)])
        theta1=np.arccos(argument); thetas.append(theta1); theta0=theta1
    thetas=np.array(thetas); thetas=0.5*(thetas[1:]+thetas[:-1])

    #Divide a circle into phi_divisions of the same length
    phis=np.linspace(0.0,2.0*pi,phi_divisions+1); phis=0.5*(phis[1:]+phis[:-1])

    #compute the positions of the selected points
    sphere_pos=[]
    for R in Radii_sphere:
        for theta in thetas:
            for phi in phis:
                sphere_pos.append([R*np.sin(theta)*np.cos(phi),
                                   R*np.sin(theta)*np.sin(phi),
                                   R*np.cos(theta)])
    sphere_pos=np.array(sphere_pos); sphere_pos=sphere_pos.astype(np.float32)

    ###########################################################################

    support = """
         #include <math.h>
         #include<omp.h>
    """
    code = """
         omp_set_num_threads(threads);
         int dims2=dims*dims;
         float x[3];
         int fx[3],index,i,j;

         #pragma omp parallel for private(x,fx,index,i,j) shared(densities)
         for (int n=0;n<siz;n++){
             for (j=0;j<sphere_points;j++){
                 for (i=0;i<3;i++){
                    x[i]=(pos(n,i)+R(n)*sphere_pos(j,i))*units(0);
                    fx[i]=(int)floor(x[i]+0.5);
                    fx[i]=(fx[i]+dims)%dims;
                 }
                 index=dims2*fx[0] + dims*fx[1] + fx[2];
                 #pragma omp atomic
                     densities(index)+=1.0; 
             } 
         }
    """
    code_w = """
         omp_set_num_threads(threads);
         int dims2=dims*dims;
         float x[3];
         int fx[3],index,i,j;

         #pragma omp parallel for private(x,fx,index,i,j) shared(densities) 
         for (int n=0;n<siz;n++){
             for (j=0;j<sphere_points;j++){
                 for (i=0;i<3;i++){
                    x[i]=(pos(n,i)+R(n)*sphere_pos(j,i))*units(0);
                    fx[i]=(int)floor(x[i]+0.5);
                    fx[i]=(fx[i]+dims)%dims;
                 }
                 index=dims2*fx[0] + dims*fx[1] + fx[2];
                 #pragma omp atomic
                     densities(index)+=wg(n); 
             } 
         }
    """

    #check that the sizes of the positions and the weights are the same
    if weights!=None:
        if total_siz!=weights.shape[0]:
            print 'the sizes of the positions and weights are not the same'
            print total_siz,weights.shape[0]; sys.exit()

    #if the array to be sent is larger than n_max, split it into smaller pieces
    start=0; final=False
    while not(final):

        if start+n_max>total_siz:
            end=total_siz; final=True
        else:
            end=start+n_max

        print start,'--',end
        pos=positions[start:end]; R=radii[start:end]; siz=pos.shape[0]

        if weights==None:
            wv.inline(code,['pos','R','sphere_pos','units','siz','threads',
                            'sphere_points','dims','densities'],
                      extra_compile_args=['-O3 -fopenmp'],
                      extra_link_args=['-lgomp'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m','gomp'])
        else:
            wg=weights[start:end]
            wv.inline(code_w,['pos','R','sphere_pos','units','siz','threads',
                            'sphere_points','dims','densities','wg'],
                      extra_compile_args=['-O3 -fopenmp'],
                      extra_link_args=['-lgomp'],
                      type_converters = wv.converters.blitz,
                      verbose=2,support_code = support,libraries = ['m','gomp'])

        start=end

    #to take into account that each SPH gas particle is divided into 
    #divisions^3 points
    densities/=divisions**3

    return densities

#This function returns \int_0^r w(r,h) d^3r: u=r/h
def Volw(u):
    if u<0.5:
        return 32.0*(1.0/3.0*u**3-6.0/5.0*u**5+u**6)
    elif u<=1.0:
        return 16.0/15.0*u**3*(36.0*u**2-10.0*u**3-45.0*u+20.0)-1.0/15.0
    else:
        print 'error: u=r/h can not be larger than 1'
################################################################################





############################### EXAMPLE OF USAGE ###############################
if len(sys.argv)==2:
    if sys.argv[0]=='CIC_library.py' and sys.argv[1]=='compile':

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
        ### TSC_serial ###
        
        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        #pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        #print pos

        tsc_overdensities=np.zeros(dims**3,dtype=np.float32)
        #compute densities
        TSC_serial(pos,dims,BoxSize,tsc_overdensities)
        tsc_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(tsc_overdensities,dtype=np.float64)
        print tsc_overdensities
        print np.min(tsc_overdensities),np.max(tsc_overdensities)

#########################################################################
        ### TSC_serial with weights ###
        
        n=100**3
        BoxSize=500.0 #Mpc/h
        dims=128

        #pos=(np.random.random((n,3))*BoxSize).astype(np.float32)
        #print pos
        
        weights=np.ones(n,dtype=np.float32)
        print weights
        tsc_overdensities=np.zeros(dims**3,dtype=np.float32)
        #compute densities
        TSC_serial(pos,dims,BoxSize,tsc_overdensities,weights)
        tsc_overdensities*=dims**3*1.0/n #divide by mean to obtain overdensities

        print np.sum(tsc_overdensities,dtype=np.float64)
        print tsc_overdensities
        print np.min(tsc_overdensities),np.max(tsc_overdensities)

#########################################################################
        ### SPH_gas without weights ###
        n=100**3 #number of particles
        BoxSize=100.0 #Mpc/h
        dims=128

        divisions=2
        threads=1

        pos=(np.random.random((n,3))*BoxSize).astype(np.float32) #positions
        R=np.ones(dims**3,dtype=np.float32)*BoxSize/100.0 #SPH radii
        print pos
        print R

        overdensities=np.zeros(dims**3,dtype=np.float32)
        SPH_gas(pos,R,divisions,dims,BoxSize,threads,
                overdensities,weights=None)
        print np.sum(overdensities,dtype=np.float64); print n
        overdensities*=(dims**3*1.0/n)

        print np.min(overdensities),'< density / <density> <',\
            np.max(overdensities)
        print overdensities

#########################################################################
        ### SPH_gas with weights ###
        n=100**3 #number of particles
        BoxSize=100.0 #Mpc/h
        dims=128

        divisions=2
        threads=1

        #pos=(np.random.random((n,3))*BoxSize).astype(np.float32) #positions
        #R=np.ones(n,dtype=np.float32)*BoxSize/100.0 #SPH radii
        #print pos
        #print R

        weights=np.ones(n,dtype=np.float32)

        overdensities=np.zeros(dims**3,dtype=np.float32)
        SPH_gas(pos,R,divisions,dims,BoxSize,threads,
                overdensities,weights)
        print np.sum(overdensities,dtype=np.float64); print n
        overdensities*=(dims**3*1.0/n)

        print np.min(overdensities),'< density / <density> <',\
            np.max(overdensities)
        print overdensities
        


    else:
        print 'To compile the code type:'
        print 'python CIC_library.py compile'


