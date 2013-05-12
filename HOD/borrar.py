import numpy as np
import time

BoxSize=500.0 #Mpc/h
dims=10
points=200000
pos_g=np.random.random((points,3))*BoxSize

 
start=time.clock()
dims2=dims**2; dims3=dims**3
Ng=len(pos_g)*1.0
coord=np.floor(dims*pos_g/BoxSize).astype(np.int32)
index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]

indexes_g=[]
for i in range(dims3):
    ids=np.where(index==i)[0]
    indexes_g.append(ids)
indexes_g=np.array(indexes_g)
end=time.clock()
print 'time:',end-start
print pos_g[indexes_g[7]]
