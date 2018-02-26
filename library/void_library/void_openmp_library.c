#include <stdio.h>
#include "void_openmp_library.h"
#include <omp.h>
#include <math.h>


void mark_void_region(int *in_void, int Ncells, int dims, float R_grid2,
		      int i, int j, int k, int threads)
{
  int l, m, n, i1, j1, k1;
  long number;
  float dist2;

#pragma omp parallel for num_threads(threads) private(l,m,n,i1,j1,k1,dist2,number) shared(in_void,i,j,k,Ncells,R_grid2,dims)
  for (l=-Ncells; l<=Ncells; l++)
    {
      i1 = (i+l+dims)%dims;
      for (m=-Ncells; m<=Ncells; m++)
	{
	  j1 = (j+m+dims)%dims;
	  for (n=-Ncells; n<=Ncells; n++)
	    {
	      k1 = (k+n+dims)%dims;

	      dist2 = l*l + m*m + n*n;
	      if (dist2<R_grid2)
		{
		  number = i1*dims*dims + j1*dims + k1;
		  in_void[number] = 1;
		}
	    }
	}
    } 
}

// This routine computes the distance between a cell and voids already identified
// if that distance is smaller than the sum of their radii then the cell can not
// host a void as it will overlap with the other void
int num_voids_around(long total_voids_found, long *IDs, int dims, float middle,
		     int i, int j, int k, float *void_radius, float *void_pos,
		     float R_grid, int threads)
{

  int l, nearby_voids=0;
  float dx, dy, dz, dist2;

#pragma omp parallel for num_threads(threads) private(l,dx,dy,dz,dist2) shared(void_pos,void_radius,nearby_voids,R_grid,i,j,k,dims,middle)
  for (l=0; l<total_voids_found; l++)
    {
      if (nearby_voids>0)  continue;

      dx = i - void_pos[3*l+0];
      if (dx>middle)   dx = dx - dims;
      if (dx<-middle)  dx = dx + dims;

      dy = j - void_pos[3*l+1];
      if (dy>middle)   dy = dy - dims;
      if (dy<-middle)  dy = dy + dims;

      dz = k - void_pos[3*l+2];
      if (dz>middle)   dz = dz - dims;
      if (dz<-middle)  dz = dz + dims;

      dist2 = dx*dx + dy*dy + dz*dz;

      if (dist2<((void_radius[l]+R_grid)*(void_radius[l]+R_grid)))
	{
#pragma omp atomic
	  nearby_voids += 1;
	}
    }
  
  return nearby_voids;
}


// This routine looks at the cells around a given cell to see if those belong
// to other voids
int num_voids_around2(int Ncells, int i, int j, int k, int dims,
		      float R_grid2, int *in_void, int threads)
{
  int l, m, n, i1, j1, k1, nearby_voids=0;
  long num;
  float dist2;

#pragma omp parallel for num_threads(threads) private(l, m, n, i1, j1, k1, num, dist2) shared(Ncells, i, j, k, dims, nearby_voids, R_grid2, in_void)
  for (l=-Ncells; l<=Ncells; l++)
    {
      if (nearby_voids>0)  continue;

      i1 = (i+l+dims)%dims;
      for (m=-Ncells; m<=Ncells; m++)
	{
	  j1 = (j+m+dims)%dims;
	  for (n=-Ncells; n<=Ncells; n++)
	    {
	      k1 = (k+n+dims)%dims;

	      num = i1*dims*dims + j1*dims + k1;
	      dist2 = l*l + m*m + n*n;
	      if ((dist2<R_grid2) && (in_void[num]==1))
		{
#pragma omp atomic
		  nearby_voids += 1;
		}
	    }
	}
    }
	  
  return nearby_voids;
}

