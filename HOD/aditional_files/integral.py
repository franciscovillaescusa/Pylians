import numpy as np
import scipy.integrate as si
import sys

def xi_interp(r_interp,r,xi):
    return np.interp(r_interp,r,xi)

def deriv(y,x,r,xi,rp):
    value=2.0*x*np.interp(x,r,xi)/np.sqrt(x**2-rp**2)
    return np.array([value])


r,xi=[],[]
f=open('borrar2.dat','r')
for line in f.readlines():
    a=line.split()
    r.append(float(a[0]))
    xi.append(float(a[1]))
f.close()
r=np.array(r)
xi=np.array(xi)

intervals=1000
r_min=np.min(r); r_max=np.max(r)
f=open('borrar.dat','w')
for i in range(intervals):
    ri=10**(np.log10(r_min)+np.log10(r_max/r_min)*i/intervals)
    f.write(str(ri)+' '+str(xi_interp(ri,r,xi))+'\n')
f.close()

rp=[0.17, 0.27, 0.42, 0.67, 1.1, 1.7, 2.7, 4.2, 6.7, 10.6, 16.9, 26.8, 42.3]

#print si.quadrature(xi_interp,0.17,r_max,args=(r,xi),maxiter=500)



steps=10000000

index=1
r_p=rp[index]+1e-10


### simple rule
delta=(r_max-r_p)*1.0/steps
r_aux=np.linspace(r_p,r_max,steps)
r_int=0.5*(r_aux[1:]+r_aux[:-1])

xi_r=np.interp(r_int,r,xi)

integral=delta*np.sum(xi_r*2.0*r_int/np.sqrt(r_int**2-r_p**2))
print 'integral=',integral


### trapezoidal rule
delta=(r_max-r_p)*1.0/steps
r_aux=np.linspace(r_p,r_max,steps)
f=np.interp(r_aux,r,xi)*2.0*r_aux/np.sqrt(r_aux**2-rp[index]**2)
integral=delta*(np.sum(f[1:-1])+f[0]+f[steps-1])
print 'integral=',integral

### odeint
time=np.array([r_p,r_max])
yinit=np.array([0.0])
y=si.odeint(deriv,yinit,time,args=(r,xi,rp[index]),mxstep=1000)
print 'integral=',y[1][0]


# wp(rp)
h=1e-13
yinit=np.array([0.0])

for Rp in rp:
    time=np.array([Rp+h,r_max])
    y=si.odeint(deriv,yinit,time,args=(r,xi,Rp),mxstep=1000)
    print 'integral=',y[1][0]



sys.exit()
