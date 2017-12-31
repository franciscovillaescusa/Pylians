double rectangular(double x_min, double x_max, double *x, double *y,
		   long elements);
double simpson(double x_min, double x_max, double *x, double *y,
	       long elements, long steps);

void rk4(double y[], double dydx[], int n, double x, double h, double yout[],
	 double a[], double b[], long elements,
	 //void (*derivs)(double, double [], double []));
	 void (*derivs)(double, double [], double [],
			double [], double [], long));


double *rkdumb(double vstart[], int nvar, double x1, double x2, long nstep,
	       double a[], double b[], long elements,
	       //void (*derivs)(double, double [], double []));
	       void (*derivs)(double, double [], double [],
			      double [], double [], long));

void derivs(double, double [], double [], double [], double [], long);
