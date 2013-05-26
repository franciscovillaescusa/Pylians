#This code writes a binary file containing the positions of particles within
#a box of size 1.0
#The format of the file is:
# x1y1z1x2y2z2x3y3z3.....xNyNzN
#each coordinate is a float of 32 bits
#to read the file it is enough to do:
#dt=np.dtype((np.float32,3)); b=np.fromfile(file,dtype=dt)

import numpy as np

############################ INPUT ###################################
file='random_catalogue_1e7.dat'
points=10000000
######################################################################

a=np.random.random((points,3)).astype(np.float32)
f=open(file,'wb')
for i in range(len(a)):
    f.write(a[i])
f.close()

