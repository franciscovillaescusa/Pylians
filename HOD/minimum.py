import numpy as np

file='results_0.0.dat'

f=open(file,'r')
data=[]
for line in f.readlines():
    a=line.split()
    data.append([float(a[0]),float(a[1]),float(a[2])])
f.close()
data=np.array(data)

X2min=np.min(data[:,2])
print data[np.where(data[:,2]==X2min)]

