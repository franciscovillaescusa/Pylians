#!/usr/bin/env python

#This code creates a grid of a given distribution of points
import numpy as np
import sys

if len(sys.argv)==1 or len(sys.argv)!=6:
    print 'USAGE:'
    print 'python map.py filename axis1 axis2 points_axis1 points_axis2'
else:
    fname=str(sys.argv[1])
    axis1=int(sys.argv[2])
    axis2=int(sys.argv[3])
    points1=int(sys.argv[4])
    points2=int(sys.argv[5])

    X,Y=[],[]
    f=open(fname,'r')
    for line in f.readlines():
        a=line.split()
        X.append(float(a[axis1]))
        Y.append(float(a[axis2]))
    f.close()
    X=np.array(X)
    Y=np.array(Y)

    min_X=np.min(X)
    max_X=np.max(X)
    min_Y=np.min(Y)
    max_Y=np.max(Y)

    grid=np.zeros((points1,points2),dtype=np.float32)

    bin1=((X-min_X)*points1/(max_X-min_X)).astype(np.int32)
    beyond=np.where(bin1==points1)
    bin1[beyond]=points1-1

    bin2=((Y-min_Y)*points1/(max_Y-min_Y)).astype(np.int32)
    beyond=np.where(bin2==points2)
    bin2[beyond]=points2-1
    
    for i in range(points1):
        for j in range(points2):
            a=bin1==i
            b=bin2==j
            c=a*b
            grid[i,j]=len(np.where(c==True)[0])
            
    if (int(np.sum(grid))!=len(X)):
        print 'not all points counted',np.sum(grid),len(X)
        sys.exit()

    f=open(fname+'_map','w')
    for i in range(points1):
        X_middle=min_X+(max_X-min_X)*(i+0.5)/points1
        for j in range(points2):
            Y_middle=min_Y+(max_Y-min_Y)*(j+0.5)/points2
            f.write(str(X_middle)+' '+str(Y_middle)+' '+str(grid[i,j])+'\n')
    f.close()



