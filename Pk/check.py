import numpy as np
import readsubf
import readfof
import readsnap
import sys

snapshot_fname='/home/villa/disksom2/ICTP/CDM/0/snapdir_008/snap_008'
groups_number=8
mass1=4e13
mass2=4e13

#read DM positions and IDS
Pos=readsnap.read_block(snapshot_fname,"POS ",parttype=-1)/1e3 #Mpc/h
Vel=readsnap.read_block(snapshot_fname,"VEL ",parttype=-1)     #km/s
IDS=readsnap.read_block(snapshot_fname,"ID  ",parttype=-1)
print len(IDS),np.min(IDS),np.max(IDS)
sorted_ids=IDS.argsort(axis=0); del IDS
#the particle whose ID is N is located in the position sorted_ids[N]
#i.e. DM_ids[sorted_ids[N]]=N
#the position of the particle whose ID is N would be:
#DM_pos[sorted_ids[N]]

############################### CDM+NU #######################################
groups_fname='/home/villa/disksom2/ICTP/CDM/0/'

#read FoF halos information
fof=readfof.FoF_catalog(groups_fname,groups_number,long_ids=True,swap=False)
F_pos1=fof.GroupPos/1e3        #positions in Mpc/h
F_mass1=fof.GroupMass*1e10     #masses in Msun/h
F_vel1=fof.GroupVel            #velocity of the halo
F_len1=fof.GroupLen            #number of particles in the FoF halos
F_offset1=fof.GroupOffset      #offset of the particles belonging to the group
F_ID1=fof.GroupIDs             #ID of the particles in the FoF halo
del fof

#read CDM halos/subhalos information
halos=readsubf.subfind_catalog(groups_fname,groups_number,
                               group_veldisp=True,masstab=True,
                               long_ids=True,swap=False)
#F_pos1=halos.group_pos/1e3             #positions in Mpc/h
#F_mass1=halos.group_m_mean200*1e10     #masses in Msun/h

F_pos1=halos.sub_pos/1e3                 #positions in Mpc/h
F_mass1=halos.sub_mass*1e10              #masses in Msun/h
F_vel1=halos.sub_vel                     #velocities in km/s
F_len1=halos.sub_len
F_offset1=halos.sub_offset
F_ID1=readsubf.subf_ids(groups_fname,groups_number,0,0,long_ids=True,
                        verbose=False,read_all=True).SubIDs
del halos


num=25 #number of particular halo to study

#find the ID of the particles belonging to halo[number]
partial_ids=F_ID1[F_offset1[num]:F_offset1[num]+F_len1[num]]-1
partial_pos=Pos[sorted_ids[partial_ids]]
partial_vel=Vel[sorted_ids[partial_ids]]
print F_pos1[num]
print np.sum(partial_pos,axis=0)/len(partial_pos)

print F_vel1[num]
print np.sum(partial_vel,axis=0)/len(partial_vel)


#inside=np.where(F_mass1>mass1)[0]
#F_pos1=F_pos1[inside]



print len(F_pos1)

#print F_pos1[0]
#print F_num1[0]

#diff=Pos-F_pos1[100000]
#diff=np.sqrt(diff[:,0]**2+diff[:,1]**2+diff[:,2]**2)
#print 'distance=',np.min(diff)



