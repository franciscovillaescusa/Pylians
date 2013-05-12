import numpy as np
import sys

r,xi=[],[]
f=open('borrar2.dat','r')
for line in f.readlines():
    a=line.split()
    r.append(float(a[0]))
    xi.append(float(a[1]))
f.close()
r=np.array(r)
xi=np.array(xi)

print xi


steps=1000000

r_min=np.min(r);   r_max=np.max(r)

r_p=16.4687

delta=(r_max-r_p)*1.0/steps
r_aux=np.linspace(r_p,r_max,steps)
r_int=0.5*(r_aux[1:]+r_aux[:-1])
print r_aux
print r_int

xi_r=np.interp(r_int,r,xi)

integral=delta*np.sum(xi_r*2.0*r_int/np.sqrt(r_int**2-r_p**2))
print 'integral=',integral
sys.exit()



f=open('borrar1.dat','w')
for i in range(steps):
    f.write(str(r_aux[i])+' '+str(xi_aux[i])+'\n')
f.close()
