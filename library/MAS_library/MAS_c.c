#include <stdio.h>
#include "MAS_c.h"
#include <omp.h>
#include <math.h>

// ###################### CIC #################### //
// This function carries out the standard CIC in 3D
void CIC3D(FLOAT *pos, FLOAT *number, long particles, int dims, FLOAT BoxSize,
	  int threads)
{

  long i, dims2;
  int axis, index_d[3], index_u[3];
  FLOAT inv_cell_size, dist, d[3], u[3];

  inv_cell_size = dims*1.0/BoxSize; 
  dims2 = dims*dims;
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,u,d,index_d,index_u) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<3; axis++)
	{
	  dist          = pos[i*3+axis]*inv_cell_size;
	  u[axis]       = dist - (int)dist;
	  d[axis]       = 1.0 - u[axis];
	  index_d[axis] = ((int)dist)%dims; 
	  index_u[axis] = (index_d[axis] + 1)%dims;
	  //index_d and index_u are always be positive, no need to add dims
	}
      
#pragma omp atomic
      number[index_d[0]*dims2 + index_d[1]*dims + index_d[2]] += d[0]*d[1]*d[2];

#pragma omp atomic
      number[index_d[0]*dims2 + index_d[1]*dims + index_u[2]] += d[0]*d[1]*u[2];

#pragma omp atomic
      number[index_d[0]*dims2 + index_u[1]*dims + index_d[2]] += d[0]*u[1]*d[2];

#pragma omp atomic
      number[index_d[0]*dims2 + index_u[1]*dims + index_u[2]] += d[0]*u[1]*u[2];

#pragma omp atomic
      number[index_u[0]*dims2 + index_d[1]*dims + index_d[2]] += u[0]*d[1]*d[2];

#pragma omp atomic
      number[index_u[0]*dims2 + index_d[1]*dims + index_u[2]] += u[0]*d[1]*u[2];

#pragma omp atomic
      number[index_u[0]*dims2 + index_u[1]*dims + index_d[2]] += u[0]*u[1]*d[2];

#pragma omp atomic
      number[index_u[0]*dims2 + index_u[1]*dims + index_u[2]] += u[0]*u[1]*u[2];
    
    }
  
}

// This function carries out the standard CIC with weights in 3D
void CICW3D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	    FLOAT BoxSize, int threads)
{

  long i, dims2;
  int axis, index_d[3], index_u[3];
  FLOAT inv_cell_size, dist, d[3], u[3], WW;

  inv_cell_size = dims*1.0/BoxSize; 
  dims2 = dims*dims;
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,u,d,index_d,index_u) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<3; axis++)
	{
	  dist          = pos[i*3+axis]*inv_cell_size;
	  u[axis]       = dist - (int)dist;
	  d[axis]       = 1.0 - u[axis];
	  index_d[axis] = ((int)dist)%dims; 
	  index_u[axis] = (index_d[axis] + 1)%dims;
	  //index_d and index_u are always be positive, no need to add dims
	}

      WW = W[i];
      
#pragma omp atomic
      number[index_d[0]*dims2 + index_d[1]*dims + index_d[2]] += d[0]*d[1]*d[2]*WW;

#pragma omp atomic
      number[index_d[0]*dims2 + index_d[1]*dims + index_u[2]] += d[0]*d[1]*u[2]*WW;

#pragma omp atomic
      number[index_d[0]*dims2 + index_u[1]*dims + index_d[2]] += d[0]*u[1]*d[2]*WW;

#pragma omp atomic
      number[index_d[0]*dims2 + index_u[1]*dims + index_u[2]] += d[0]*u[1]*u[2]*WW;

#pragma omp atomic
      number[index_u[0]*dims2 + index_d[1]*dims + index_d[2]] += u[0]*d[1]*d[2]*WW;

#pragma omp atomic
      number[index_u[0]*dims2 + index_d[1]*dims + index_u[2]] += u[0]*d[1]*u[2]*WW;

#pragma omp atomic
      number[index_u[0]*dims2 + index_u[1]*dims + index_d[2]] += u[0]*u[1]*d[2]*WW;

#pragma omp atomic
      number[index_u[0]*dims2 + index_u[1]*dims + index_u[2]] += u[0]*u[1]*u[2]*WW;
    
    }
  
}


// This function carries out the standard CIC with weights in 2D
void CIC2D(FLOAT *pos, FLOAT *number, long particles, int dims, FLOAT BoxSize,
	  int threads)
{

  long i;
  int axis, index_d[2], index_u[2];
  FLOAT inv_cell_size, dist, d[2], u[2];

  inv_cell_size = dims*1.0/BoxSize; 
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,u,d,index_d,index_u) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<2; axis++)
	{
	  dist          = pos[i*2+axis]*inv_cell_size;
	  u[axis]       = dist - (int)dist;
	  d[axis]       = 1.0 - u[axis];
	  index_d[axis] = ((int)dist)%dims; 
	  index_u[axis] = (index_d[axis] + 1)%dims;
	  //index_d and index_u are always be positive, no need to add dims
	}
      
#pragma omp atomic
      number[index_d[0]*dims + index_d[1]] += d[0]*d[1];

#pragma omp atomic
      number[index_d[0]*dims + index_u[1]] += d[0]*u[1];

#pragma omp atomic
      number[index_u[0]*dims + index_d[1]] += u[0]*d[1];

#pragma omp atomic
      number[index_u[0]*dims + index_u[1]] += u[0]*u[1];

    }
  
}


// This function carries out the standard CIC in 2D
void CICW2D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	    FLOAT BoxSize, int threads)
{

  long i;
  int axis, index_d[2], index_u[2];
  FLOAT inv_cell_size, dist, d[2], u[2], WW;

  inv_cell_size = dims*1.0/BoxSize; 
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,u,d,index_d,index_u) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<2; axis++)
	{
	  dist          = pos[i*2+axis]*inv_cell_size;
	  u[axis]       = dist - (int)dist;
	  d[axis]       = 1.0 - u[axis];
	  index_d[axis] = ((int)dist)%dims; 
	  index_u[axis] = (index_d[axis] + 1)%dims;
	  //index_d and index_u are always be positive, no need to add dims
	}

      WW = W[i];
      
#pragma omp atomic
      number[index_d[0]*dims + index_d[1]] += d[0]*d[1]*WW;

#pragma omp atomic
      number[index_d[0]*dims + index_u[1]] += d[0]*u[1]*WW;

#pragma omp atomic
      number[index_u[0]*dims + index_d[1]] += u[0]*d[1]*WW;

#pragma omp atomic
      number[index_u[0]*dims + index_u[1]] += u[0]*u[1]*WW;

    }
  
}



// ###################### NGP #################### //
// This function carries out the standard NGP in 3D
void NGP3D(FLOAT *pos, FLOAT *number, long particles, int dims, FLOAT BoxSize,
	   int threads)
{

  long i, dims2;
  int axis, index[3];
  FLOAT inv_cell_size;

  inv_cell_size = dims*1.0/BoxSize; 
  dims2 = dims*dims;
  
#pragma omp parallel for num_threads(threads) private(i,axis,index) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<3; axis++)
	{
	  index[axis] = (int)(pos[3*i+axis]*inv_cell_size + 0.5);
	  index[axis] = (index[axis])%dims; //Always positive. No need to add +dims
	}
#pragma omp atomic
      number[index[0]*dims2 + index[1]*dims + index[2]] += 1.0;
      
    }
}


// This function carries out the standard NGP with weights in 3D
void NGPW3D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	    FLOAT BoxSize, int threads)
{

  long i, dims2;
  int axis, index[3];
  FLOAT inv_cell_size;

  inv_cell_size = dims*1.0/BoxSize; 
  dims2 = dims*dims;
  
#pragma omp parallel for num_threads(threads) private(i,axis,index) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<3; axis++)
	{
	  index[axis] = (int)(pos[3*i+axis]*inv_cell_size + 0.5);
	  index[axis] = (index[axis])%dims; //Always positive. No need to add +dims
	}
#pragma omp atomic
      number[index[0]*dims2 + index[1]*dims + index[2]] += W[i];
      
    }
}

// This function carries out the standard NGP in 2D
void NGP2D(FLOAT *pos, FLOAT *number, long particles, int dims, FLOAT BoxSize,
	   int threads)
{

  long i;
  int axis, index[2];
  FLOAT inv_cell_size;

  inv_cell_size = dims*1.0/BoxSize; 
  
#pragma omp parallel for num_threads(threads) private(i,axis,index) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<2; axis++)
	{
	  index[axis] = (int)(pos[2*i+axis]*inv_cell_size + 0.5);
	  index[axis] = (index[axis])%dims; //Always positive. No need to add +dims
	}
#pragma omp atomic
      number[index[0]*dims + index[1]] += 1.0;
      
    }
}

// This function carries out the standard NGP with weights in 2D
void NGPW2D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	    FLOAT BoxSize, int threads)
{

  long i;
  int axis, index[2];
  FLOAT inv_cell_size;

  inv_cell_size = dims*1.0/BoxSize; 
  
#pragma omp parallel for num_threads(threads) private(i,axis,index) shared(number)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<2; axis++)
	{
	  index[axis] = (int)(pos[2*i+axis]*inv_cell_size + 0.5);
	  index[axis] = (index[axis])%dims; //Always positive. No need to add +dims
	}
#pragma omp atomic
      number[index[0]*dims + index[1]] += W[i];
      
    }
}


// ###################### TSC #################### //
// This function carries out the standard TSC in 3D
void TSC3D(FLOAT *pos, FLOAT *number, long particles, int dims, FLOAT BoxSize,
	   int threads)
{

  long i, dims2;
  int j, l, m, n, axis, index[3][3], minimum;
  FLOAT inv_cell_size, dist, diff, C[3][3];

  inv_cell_size = dims*1.0/BoxSize; 
  dims2 = dims*dims;
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,minimum,j,index,diff,C,l,m,n) shared(number,pos)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<3; axis++)
	{
	  dist    = pos[3*i+axis]*inv_cell_size;
	  minimum = (int)(floor(dist-1.5));
	  
	  for (j=0; j<3; j++)
	    {
	      index[axis][j] = (minimum+j+1+dims)%dims;
	      diff = fabs(minimum + j+1 - dist);
	      if (diff<0.5)
		C[axis][j] = 0.75-diff*diff;
	      else if (diff<1.5)
		C[axis][j] = 0.5*(1.5-diff)*(1.5-diff);
	      else
		C[axis][j] = 0.0;
	    }
	}
      for (l=0; l<3; l++)
	for (m=0; m<3; m++)
	  for (n=0; n<3; n++)
	    {
#pragma omp atomic
	      number[index[0][l]*dims2 + index[1][m]*dims + index[2][n]] += C[0][l]*C[1][m]*C[2][n];
	    }
    }
}

// This function carries out the standard TSC with weights in 3D
void TSCW3D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	    FLOAT BoxSize, int threads)
{

  long i, dims2;
  int j, l, m, n, axis, index[3][3], minimum;
  FLOAT inv_cell_size, dist, diff, C[3][3];

  inv_cell_size = dims*1.0/BoxSize; 
  dims2 = dims*dims;
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,minimum,j,index,diff,C,l,m,n) shared(number,pos)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<3; axis++)
	{
	  dist    = pos[3*i+axis]*inv_cell_size;
	  minimum = (int)(floor(dist-1.5));
	  
	  for (j=0; j<3; j++)
	    {
	      index[axis][j] = (minimum+j+1+dims)%dims;
	      diff = fabs(minimum + j+1 - dist);
	      if (diff<0.5)
		C[axis][j] = 0.75-diff*diff;
	      else if (diff<1.5)
		C[axis][j] = 0.5*(1.5-diff)*(1.5-diff);
	      else
		C[axis][j] = 0.0;
	    }
	}
      for (l=0; l<3; l++)
	for (m=0; m<3; m++)
	  for (n=0; n<3; n++)
	    {
#pragma omp atomic
	      number[index[0][l]*dims2 + index[1][m]*dims + index[2][n]] += C[0][l]*C[1][m]*C[2][n]*W[i];
	    }
    }
} 
                
// This function carries out the standard TSC in 2D
void TSC2D(FLOAT *pos, FLOAT *number, long particles, int dims, FLOAT BoxSize,
	   int threads)
{

  long i;
  int j, l, m, axis, index[2][3], minimum;
  FLOAT inv_cell_size, dist, diff, C[2][3];

  inv_cell_size = dims*1.0/BoxSize; 
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,minimum,j,index,diff,C,l,m) shared(number,pos)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<2; axis++)
	{
	  dist    = pos[2*i+axis]*inv_cell_size;
	  minimum = (int)(floor(dist-1.5));
	  
	  for (j=0; j<3; j++)
	    {
	      index[axis][j] = (minimum+j+1+dims)%dims;
	      diff = fabs(minimum + j+1 - dist);
	      if (diff<0.5)
		C[axis][j] = 0.75-diff*diff;
	      else if (diff<1.5)
		C[axis][j] = 0.5*(1.5-diff)*(1.5-diff);
	      else
		C[axis][j] = 0.0;
	    }
	}
      for (l=0; l<3; l++)
	for (m=0; m<3; m++)
	    {
#pragma omp atomic
	      number[index[0][l]*dims + index[1][m]] += C[0][l]*C[1][m];
	    }
    }
}

// This function carries out the standard TSC with weights in 2D
void TSCW2D(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	    FLOAT BoxSize, int threads)
{

  long i;
  int j, l, m, axis, index[2][3], minimum;
  FLOAT inv_cell_size, dist, diff, C[2][3];

  inv_cell_size = dims*1.0/BoxSize; 
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,minimum,j,index,diff,C,l,m) shared(number,pos)
  for (i=0; i<particles; i++)
    {
      for (axis=0; axis<2; axis++)
	{
	  dist    = pos[2*i+axis]*inv_cell_size;
	  minimum = (int)(floor(dist-1.5));
	  
	  for (j=0; j<3; j++)
	    {
	      index[axis][j] = (minimum+j+1+dims)%dims;
	      diff = fabs(minimum + j+1 - dist);
	      if (diff<0.5)
		C[axis][j] = 0.75-diff*diff;
	      else if (diff<1.5)
		C[axis][j] = 0.5*(1.5-diff)*(1.5-diff);
	      else
		C[axis][j] = 0.0;
	    }
	}
      for (l=0; l<3; l++)
	for (m=0; m<3; m++)
	    {
#pragma omp atomic
	      number[index[0][l]*dims + index[1][m]] += C[0][l]*C[1][m]*W[i];
	    }
    }
}
