#!/usr/bin/env python

import numpy as np
import sys

if len(sys.argv)==1 or (len(sys.argv)!=8 and len(sys.argv)!=9):
    print 'USAGE:'
    print 'cic_map.py filename axis1 axis2 points_axis1 points_axis2 log_x log_y'
    print 'or, if overdensity plots want to be made:'
    print 'cic_map.py filename axis1 axis2 points_axis1 points_axis2 log_x log_y background density'
    print 'set log_x to 1 for results in log scale in x, otherwise set to 0'
    print 'set log_y to 1 for results in log scale in y, otherwise set to 0'
    sys.exit()

fname=str(sys.argv[1])
axis_x=int(sys.argv[2])
axis_y=int(sys.argv[3])
points_x=int(sys.argv[4])
points_y=int(sys.argv[5])
log_x=bool(sys.argv[6])
log_y=bool(sys.argv[7])
if len(sys.argv)==9:
    background_density=float(sys.argv[8])


print 'Reading the data file'
f=open(fname,'r'); x,y=[],[]
for line in f.readlines():
    a=line.split()
    x.append(float(a[axis_x]))
    y.append(float(a[axis_y]))
f.close(); x=np.array(x); y=np.array(y)
if log_x:
    x=np.log10(x)
if log_y:
    y=np.log10(y)

min_x=np.min(x); max_x=np.max(x)
min_y=np.min(y); max_y=np.max(y)

delta_x=(max_x-min_x)*1.0/points_x; delta_y=(max_y-min_y)*1.0/points_y
area=delta_x*delta_y

bin_x=((x+delta_x/2.0-min_x)/delta_x).astype(np.int64)
bin_y=((y+delta_y/2.0-min_y)/delta_y).astype(np.int64)

dx=x+delta_x/2.0-(bin_x*delta_x+min_x)
tx=delta_x-dx

dy=y+delta_y/2.0-(bin_y*delta_y+min_y)
ty=delta_y-dy

grid=np.zeros((points_x,points_y),dtype=np.float64)

print 'Counting the number of points in each grid cell'
for i in range(len(x)):
    if bin_x[i]==0:
        if bin_y[i]==0:
            grid[0,0]+=1.0
        elif bin_y[i]==points_y:
            grid[0,points_y-1]+=1.0
        else:
            grid[0,bin_y[i]]+=dx[i]*dy[i]/(delta_y*dx[i])
            grid[0,bin_y[i]-1]+=dx[i]*ty[i]/(delta_y*dx[i])
    elif bin_y[i]==0:
        if bin_x[i]==points_x:
            grid[points_x-1,0]+=1.0
        else:
            grid[bin_x[i],0]+=dx[i]*dy[i]/(delta_x*dy[i])
            grid[bin_x[i]-1,0]+=tx[i]*dy[i]/(delta_x*dy[i])
    elif bin_y[i]==points_y:
        if bin_x[i]==points_x:
            grid[points_x-1,points_y-1]+=1.0
        else:
            grid[bin_x[i],points_y-1]+=dx[i]*ty[i]/(delta_x*ty[i])
            grid[bin_x[i]-1,points_y-1]+=tx[i]*ty[i]/(delta_x*ty[i])
    elif bin_x[i]==points_x:
        grid[points_x-1,bin_y[i]]+=tx[i]*dy[i]/(delta_y*tx[i])
        grid[points_x-1,bin_y[i]-1]+=tx[i]*ty[i]/(delta_y*tx[i])
    else:
        grid[bin_x[i],bin_y[i]]+=dx[i]*dy[i]/area
        grid[bin_x[i],bin_y[i]-1]+=dx[i]*ty[i]/area
        grid[bin_x[i]-1,bin_y[i]]+=tx[i]*dy[i]/area
        grid[bin_x[i]-1,bin_y[i]-1]+=tx[i]*ty[i]/area

if len(sys.argv)==9:
    background_density=background_density*area

print 'write the output file'
f=open(fname+'_cic_map','w')
for i in range(points_x):
    x_middle=min_x+(max_x-min_x)*(i+0.5)/points_x
    if log_x:
        x_middle=10**(x_middle)
    for j in range(points_y):
        y_middle=min_y+(max_y-min_y)*(j+0.5)/points_y
        if log_y:
            y_middle=10**(y_middle)
        if len(sys.argv)==8:
            f.write(str(x_middle)+' '+str(y_middle)+' '+str(grid[i,j])+'\n')
        else:
            f.write(str(x_middle)+' '+str(y_middle)+' '+str(grid[i,j]/background_density)+'\n')
f.close()



