import numpy as np

f_in='new_results/results_0.0_mean200_YB.datB_12'

f=open(f_in,'r')
M1,alpha,X2=[],[],[]
for line in f.readlines():
    a=line.split()
    if a!=[]:
        M1.append(float(a[0]))
        alpha.append(float(a[1]))
        X2.append(float(a[2]))
f.close; M1=np.array(M1); alpha=np.array(alpha); X2=np.array(X2)

index=np.where(X2==np.min(X2))[0]

print 'X2 minima=',X2[index]
print 'M_1 minima=',M1[index]
print 'alpha minima=',alpha[index]
