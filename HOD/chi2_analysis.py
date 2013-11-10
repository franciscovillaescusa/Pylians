import numpy as np



################################## INPUT #####################################
f_in='500Mpc_512_0.0_mean200.dat'

start=2
end=9

f_out='borrar.dat'
##############################################################################
bins=end-start

f=open(f_in,'r')
M1,alpha,chi2_bins=[],[],[]
for line in f.readlines():
    a=line.split()
    M1.append(float(a[0])); alpha.append(float(a[1]))
    chi2_bins.append([float(a[2]),float(a[3]),float(a[4]),float(a[5]),
                      float(a[6]),float(a[6]),float(a[7]),float(a[8]),
                      float(a[9]),float(a[10]),float(a[11]),float(a[11]),
                      float(a[12]),float(a[13]),float(a[14])])
f.close()
chi2_bins=np.array(chi2_bins); M1=np.array(M1); alpha=np.array(alpha)

chi2=chi2_bins[:,start:end]
chi2=np.sum(chi2,axis=1)

index=np.where(chi2==np.min(chi2))[0]

print 'M1=',M1[index][0]
print 'alpha=',alpha[index][0]
print 'X2=',chi2[index][0]
print 'X2/dof=',chi2[index][0]/bins

f=open(f_out,'w')
for i in range(len(chi2)):
    f.write(str(M1[i])+' '+str(alpha[i])+' '+str(chi2[i]/bins)+'\n')
f.close()
