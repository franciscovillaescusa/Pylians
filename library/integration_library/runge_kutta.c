#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "integration.h"


//Given values for the variables y[0..n-1] and their derivatives dydx[0..n-1] known at x, use the fourth-order Runge-Kutta method to advance the solution over an interval h and return the incremented variables as yout[0..n-1], which need not be a distinct array from y. The user supplies the routine derivs(x,y,dydx), which returns derivatives dydx at x.
void rk4(double y[], double dydx[], int n, double x, double h, double yout[],
	 double a[], double b[], long elements,
	 //void (*derivs)(double, double [], double []))
	 void (*derivs)(double, double [], double [], double [], double [], long))
{
  int i;
  double xh, hh, h6, *dym, *dyt, *yt;

  dym = (double *)malloc(n*sizeof(double));
  dyt = (double *)malloc(n*sizeof(double));
  yt  = (double *)malloc(n*sizeof(double));

  hh = h*0.5;
  h6 = h/6.0;
  xh = x+hh;

  for (i=0;i<n;i++) yt[i]=y[i]+hh*dydx[i]; 
  //(*derivs)(xh,yt,dyt);
  (*derivs)(xh,yt,dyt,a,b,elements);
  for (i=0;i<n;i++) yt[i]=y[i]+hh*dyt[i];
  //(*derivs)(xh,yt,dym);
  (*derivs)(xh,yt,dym,a,b,elements);
  for (i=0;i<n;i++) {
    yt[i]=y[i]+h*dym[i];
    dym[i] += dyt[i];
  }
  //(*derivs)(x+h,yt,dyt);
  (*derivs)(x+h,yt,dyt,a,b,elements);
  for (i=0;i<n;i++)
    yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);

  free(yt);  free(dyt);  free(dym);
}


//float **y,*xx; //For communication back to main.

//Starting from initial values vstart[0..nvar-1] known at x1 use fourth-order Runge-Kutta to advance nstep equal increments to x2. The user-supplied routine derivs(x,v,dvdx) evaluates derivatives. Results are stored in the global variables y[0..nvar-1][1..nstep+1] and xx[1..nstep+1].
double *rkdumb(double vstart[], int nvar, double x1, double x2, long nstep,
	       double a[], double b[], long elements,
	       //void (*derivs)(double, double [], double []))
	       void (*derivs)(double, double [], double [], double [],
			      double [], long))
{

  int i;
  long k;
  double x, h, *v, *vout, *dv;

  v    = (double *)malloc(nvar*sizeof(double));
  vout = (double *)malloc(nvar*sizeof(double));
  dv   = (double *)malloc(nvar*sizeof(double));
  
  for (i=0;i<nvar;i++) {
    v[i]=vstart[i];
  }

  x=x1;
  h=(x2-x1)/nstep;

  for (k=0;k<nstep;k++) {
    //(*derivs)(x,v,dv);
    (*derivs)(x,v,dv,a,b,elements);
    rk4(v,dv,nvar,x,h,vout,a,b,elements,derivs);
    if ((double)(x+h) == x){
      printf("Step size too small in routine rkdumb\n");
      exit(1);
    }
    x += h;
    
    for (i=0;i<nvar;i++)
      v[i]=vout[i];  
  }
  
  free(dv);  free(vout);
  return v;

}
