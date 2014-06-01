import numpy as np
import readsnap
import readsubf
import HI_library as HIL
import sys

################################# UNITS #######################################
rho_crit=2.77536627e11 #h^2 Msun/Mpc^3

Mpc=3.0856e24 #cm
Msun=1.989e33 #g
Ymass=0.24 #helium mass fraction
mH=1.6726e-24 #proton mass in grams

pi=np.pi
###############################################################################

################################ INPUT ########################################
if len(sys.argv)>1:
    sa=sys.argv

    snapshot_fname=sa[1]; groups_fname=sa[2]; groups_number=int(sa[3])
    method=sa[4]

    fac=float(sa[5]); HI_frac=float(sa[6]); Omega_HI_ref=float(sa[7])
    method_Bagla=int(sa[8]); long_ids_flag=bool(int(sa[9]))
    SFR_flag=bool(int(sa[10])); f_MF=sa[11]
    
    threads=int(sa[12]); num_los=int(sa[13]) 
    f_out=sa[14]

    print '################# INFO ##############'
    for element in sa:
        element

else:
    #snapshot and halo catalogue
    snapshot_fname='../Efective_model_15Mpc/snapdir_013/snap_013'
    groups_fname='../Efective_model_15Mpc/FoF_0.2'
    groups_number=13

    #'Dave','method_1','Bagla','Barnes'
    method='Dave'

    #1.362889 (60 Mpc/h z=3) 1.436037 (30 Mpc/h z=3) 1.440990 (15 Mpc/h z=3)
    fac=1.436037 #factor to obtain <F> = <F>_obs from the Lya : only for Dave
    HI_frac=0.95 #HI/H for self-shielded regions : for method_1
    Omega_HI_ref=1e-3 #for method_1 and Bagla
    method_Bagla=3 #only for Bagla
    long_ids_flag=False; SFR_flag=True #flags for reading the FoF file
    f_MF='../mass_function/ST_MF_z=2.4.dat' #file containing the mass function

    threads=15
    num_los=5000

    f_out='cross_section_Dave_15Mpc_z=2.4.dat'
###############################################################################

#read snapshot head and obtain BoxSize, Omega_m and Omega_L
print '\nREADING SNAPSHOTS PROPERTIES'
head=readsnap.snapshot_header(snapshot_fname)
BoxSize=head.boxsize/1e3 #Mpc/h
Nall=head.nall
Masses=head.massarr*1e10 #Msun/h
Omega_m=head.omega_m
Omega_l=head.omega_l
redshift=head.redshift
Hubble=100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)  #h*km/s/Mpc
h=head.hubble

#find the total number of particles in the simulation
Ntotal=np.sum(Nall,dtype=np.uint64)
print 'Total number of particles in the simulation:',Ntotal

#sort the pos array
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)-1
pos_unsort=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
pos=np.empty((Ntotal,3),dtype=np.float32); pos[ID_unsort]=pos_unsort
del pos_unsort, ID_unsort

#sort the R array
ID_unsort=readsnap.read_block(snapshot_fname,"ID  ",parttype=0)-1
R_unsort=readsnap.read_block(snapshot_fname,"HSML",parttype=0)/1e3 #Mpc/h
R=np.zeros(Ntotal,dtype=np.float32); R[ID_unsort]=R_unsort
del R_unsort, ID_unsort

#find the IDs and HI masses of the particles to which HI has been assigned
if method=='Dave':
    [IDs,M_HI]=HIL.Dave_HI_assignment(snapshot_fname,HI_frac,fac)
elif method=='method_1':
    [IDs,M_HI]=HIL.method_1_HI_assignment(snapshot_fname,HI_frac,Omega_HI_ref)
elif method=='Barnes':
    [IDs,M_HI]=HIL.Barnes_Haehnelt(snapshot_fname,groups_fname,
                                   groups_number,long_ids_flag,SFR_flag)
elif method=='Paco':
    [IDs,M_HI]=HIL.Paco_HI_assignment(snapshot_fname,groups_fname,
                                      groups_number,long_ids_flag,SFR_flag)
elif method=='Bagla':
    [IDs,M_HI]=HIL.Bagla_HI_assignment(snapshot_fname,groups_fname,
                                       groups_number,Omega_HI_ref,method_Bagla,
                                       f_MF,long_ids_flag,SFR_flag)
else:
    print 'Incorrect method selected!!!'; sys.exit()

#just keep with the particles having HI masses
M_HI=M_HI[IDs]; pos=pos[IDs]; R=R[IDs]; del IDs

#compute the value of Omega_HI
print 'Omega_HI = %e'%(np.sum(M_HI,dtype=np.float64)/BoxSize**3/rho_crit)


#read FoF/Subfind halos information
halos=readsubf.subfind_catalog(groups_fname,groups_number,group_veldisp=True,
                               masstab=True,long_ids=long_ids_flag,swap=False)
pos_SO =halos.group_pos/1e3        #Mpc/h
M_SO   =halos.group_m_mean200*1e10 #Msun/h. SO mass
R_SO   =halos.group_r_mean200/1e3  #Mpc/h          
M_FoF  =halos.group_mass*1e10      #Msun/h. FoF mass
del halos

"""#write X-Y positions and R of the halos
f=open('borrar.dat','w')
for i in range(len(R_SO)):
    print i
    if M_FoF[i]>8.75e8:
        f.write(str(pos_SO[i,0])+' '+str(pos_SO[i,1])+' '+str(pos_SO[i,2])+\
                    ' '+str(R_SO[i])+' '+str(M_FoF[i])+'\n')
f.close()"""

#some verbose
print 'Number of FoF halos:',len(pos_SO),len(M_SO)
print '%f < X [Mpc/h] < %f'%(np.min(pos_SO[:,0]),np.max(pos_SO[:,0]))
print '%f < Y [Mpc/h] < %f'%(np.min(pos_SO[:,1]),np.max(pos_SO[:,1]))
print '%f < Z [Mpc/h] < %f'%(np.min(pos_SO[:,2]),np.max(pos_SO[:,2]))
print '%e < M [Msun/h] < %e'%(np.min(M_SO),np.max(M_SO))

#find the number of cells per dimension: the grid will have cells x cells points
cells=int(BoxSize/np.max(R))
print '%d x %d grid created\n'%(cells,cells)

#sort particles: to each particle we associate a (index_x,index_y) coordinate
index_x=(pos[:,0]/BoxSize*cells).astype(np.int32)
index_y=(pos[:,1]/BoxSize*cells).astype(np.int32)

index_x[np.where(index_x==cells)[0]]=0
index_y[np.where(index_y==cells)[0]]=0

print '%d < index_x < %d'%(np.min(index_x),np.max(index_x))
print '%d < index_y < %d'%(np.min(index_y),np.max(index_y))

print 'Sorting particles...'
indexes=[]
for i in range(cells):
    for j in range(cells):
        indexes.append([])

number=cells*index_y+index_x; del index_x,index_y
for i in range(len(number)):
    indexes[number[i]].append(i)
indexes=np.array(indexes); print 'Done!'




#do a loop over all the halos
for l in xrange(0,50000): #(len(pos_SO)):  

    """x_halo=10.2 #Mpc/h
    y_halo=8.3  #Mpc/h
    r_halo=1.0  #Mpc/h"""

    x_halo=pos_SO[l,0]; y_halo=pos_SO[l,1]; z_halo=pos_SO[l,2]; r_halo=R_SO[l]

    print '\nl=',l
    print 'halo pos =',pos_SO[l]
    print 'halo mass = %e'%M_FoF[l]
    print 'halo radius = %f'%r_halo

    index_x_min=int((x_halo-r_halo)/BoxSize*cells)
    index_x_max=int((x_halo+r_halo)/BoxSize*cells)
    if (x_halo+r_halo)>((index_x_max+1)*BoxSize/cells):
        index_x_max+=1

    index_y_min=int((y_halo-r_halo)/BoxSize*cells)
    index_y_max=int((y_halo+r_halo)/BoxSize*cells)
    if (y_halo+r_halo)>((index_y_max+1)*BoxSize/cells):
        index_y_max+=1

    print index_x_min,index_x_max
    print index_y_min,index_y_max

    #identify the IDs of the particles that can contribute to that region
    length=0
    for i in xrange(index_x_min,index_x_max+1):
        number_x=(i+cells)%cells
        for j in xrange(index_y_min,index_y_max+1):
            number_y=(j+cells)%cells
            num=cells*number_y+number_x
            length+=len(indexes[num])

    IDs=np.empty(length,dtype=np.int32); offset=0
    for i in xrange(index_x_min,index_x_max+1):
        number_x=(i+cells)%cells
        for j in xrange(index_y_min,index_y_max+1):
            number_y=(j+cells)%cells
            num=cells*number_y+number_x
            length=len(indexes[num])
            IDs[offset:offset+length]=indexes[num]
            offset+=length

    pos_gas=pos[IDs]; R_gas=R[IDs]; M_HI_gas=M_HI[IDs]

    #compute the cross section only if there are particles!!!
    if len(pos_gas)>0:

        #keep only with particles with z-coordinates within the virial radius
        z=pos_gas[:,2]
        indexes_z=np.where((z>(z_halo-r_halo)) & (z<(z_halo+r_halo)))[0]
        pos_gas=pos_gas[indexes_z]; R_gas=R_gas[indexes_z]
        M_HI_gas=M_HI_gas[indexes_z]

        #f=open('borrar2.dat','w')
        #for i in xrange(len(pos_gas)):
        #f.write(str(pos_gas[i,0])+' '+str(pos_gas[i,1])+' '+str(R_gas[i])+'\n')
        #f.close()

        cross_section=HIL.cross_section_halo(x_halo,y_halo,r_halo,num_los,
                                             redshift,h,pos_gas[:,0],
                                             pos_gas[:,1],R_gas,M_HI_gas,
                                             threads)
                                             
        #note that the cross_section returned it is comoving (Mpc/h)^2 units
        print 'cross section=',cross_section
        f=open(f_out,'a')
        f.write(str(M_FoF[l])+' '+str(cross_section*1e6)+'\n')
        f.close()
