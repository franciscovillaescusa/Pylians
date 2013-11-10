import numpy as np
import sys

Mnu=0.0
z=[0, 0.5, 1, 2]

folder='Paco_1000Mpc_z=99/'

for Z in z:
    f_halos=folder+'Pk_halos_'+str(Mnu)+'_z='+str(Z)+'.dat'
    f_DM=folder+'Pk_DM_'+str(Mnu)+'_z='+str(Z)+'.dat'
    f_DM_halos=folder+'Pk_DM-halos_'+str(Mnu)+'_z='+str(Z)+'.dat'
    f_out=folder+'bias_'+str(Mnu)+'_z='+str(Z)+'.dat'
    f_out2=folder+'bias_cross_'+str(Mnu)+'_z='+str(Z)+'.dat'

    #read halos P(k) file
    f=open(f_halos,'r')
    k_halos,Pk_halos,dPk_halos=[],[],[]
    for line in f.readlines():
        a=line.split()
        k_halos.append(float(a[0]))
        Pk_halos.append(float(a[1]))
        dPk_halos.append(float(a[2]))
    f.close()
    k_halos=np.array(k_halos); Pk_halos=np.array(Pk_halos)
    dPk_halos=np.array(dPk_halos)

    #read DM P(k) file
    f=open(f_DM,'r')
    k_DM,Pk_DM=[],[]
    for line in f.readlines():
        a=line.split()
        k_DM.append(float(a[0]))
        Pk_DM.append(float(a[1]))
    f.close()
    k_DM=np.array(k_DM); Pk_DM=np.array(Pk_DM)

    #read DM-halo P(k) file
    f=open(f_DM_halos,'r')
    k_DM_halos,Pk_DM_halos=[],[]
    for line in f.readlines():
        a=line.split()
        k_DM_halos.append(float(a[0]))
        Pk_DM_halos.append(float(a[1]))
    f.close()
    k_DM_halos=np.array(k_DM_halos); Pk_DM_halos=np.array(Pk_DM_halos)

    #check that k-arrays are equal
    if np.any(k_halos!=k_DM):
        print 'k-arrays are different!'
        sys.exit()
    if np.any(k_halos!=k_DM_halos):
        print 'k-arrays are different!'
        sys.exit()

    #to avoid divisions by 0
    inside=np.where(Pk_DM>0.0)[0]
    k_halos=k_halos[inside]; Pk_halos=Pk_halos[inside]
    dPk_halos=dPk_halos[inside]; Pk_DM=Pk_DM[inside]
    Pk_DM_halos=Pk_DM_halos[inside]

    #compute bias
    b=np.sqrt(Pk_halos/Pk_DM)
    db=(dPk_halos/Pk_DM)/(2.0*b)

    #write file
    f=open(f_out,'w')
    for i in range(len(k_halos)):
        f.write(str(k_halos[i])+' '+str(b[i])+' '+str(db[i])+'\n')
    f.close()

    #comute bias from cross-P(k)
    b=Pk_DM_halos/Pk_DM

    #write file
    f=open(f_out2,'w')
    for i in range(len(k_halos)):
        f.write(str(k_halos[i])+' '+str(b[i])+'\n')
    f.close()
