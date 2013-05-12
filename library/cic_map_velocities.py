#!/usr/bin/env python

import numpy as np
import sys

if len(sys.argv)==1 or (len(sys.argv)!=4 and len(sys.argv)!=5):
    print 'format of the files should be:'
    print 'x1 y1 Vx1 Vy1'
    print 'x2 y2 Vx2 Vy2'
    print '......'
    print 'xN yN VxN VyN'
    print 'USAGE:'
    print 'cic_map_velocities.py filename points_x points_y'
    print 'or, if overdensity plots want to be made:'
    print 'cic_map_velocities.py filename points_x points_y background density'
    sys.exit()

fname=str(sys.argv[1])
points_x=int(sys.argv[2])
points_y=int(sys.argv[3])
if len(sys.argv)==5:
    background_density=float(sys.argv[4])

x,y,Vx,Vy=[],[],[],[]
f=open(fname,'r')
for line in f.readlines():
    a=line.split()
    x.append(float(a[0]))
    y.append(float(a[1]))
    Vx.append(float(a[2]))
    Vy.append(float(a[3]))
f.close()
x=np.array(x)
y=np.array(y)
Vx=np.array(Vx)
Vy=np.array(Vy)

min_x=np.min(x)
max_x=np.max(x)
min_y=np.min(y)
max_y=np.max(y)

delta_x=(max_x-min_x)*1.0/points_x
delta_y=(max_y-min_y)*1.0/points_y
area=delta_x*delta_y

bin_x=((x+delta_x/2.0-min_x)/delta_x).astype(np.int64)
bin_y=((y+delta_y/2.0-min_y)/delta_y).astype(np.int64)

dx=x+delta_x/2.0-(bin_x*delta_x+min_x)
tx=delta_x-dx

dy=y+delta_y/2.0-(bin_y*delta_y+min_y)
ty=delta_y-dy

grid=np.zeros((points_x,points_y),dtype=np.float64)
grid_Vx=np.zeros((points_x,points_y),dtype=np.float64)
grid_Vy=np.zeros((points_x,points_y),dtype=np.float64)

for i in range(len(x)):
    if bin_x[i]==0:
        if bin_y[i]==0:
            grid[0,0]+=1.0
            grid_Vx[0,0]+=Vx[i]
            grid_Vy[0,0]+=Vy[i]
        elif bin_y[i]==points_y:
            grid[0,points_y-1]+=1.0
            grid_Vx[0,points_y-1]+=Vx[i]
            grid_Vy[0,points_y-1]+=Vy[i]
        else:
            grid[0,bin_y[i]]+=dx[i]*dy[i]/(delta_y*dx[i])
            grid_Vx[0,bin_y[i]]+=dx[i]*dy[i]/(delta_y*dx[i])*Vx[i]
            grid_Vy[0,bin_y[i]]+=dx[i]*dy[i]/(delta_y*dx[i])*Vy[i]
            grid[0,bin_y[i]-1]+=dx[i]*ty[i]/(delta_y*dx[i])
            grid_Vx[0,bin_y[i]-1]+=dx[i]*ty[i]/(delta_y*dx[i])*Vx[i]
            grid_Vy[0,bin_y[i]-1]+=dx[i]*ty[i]/(delta_y*dx[i])*Vy[i]
    elif bin_y[i]==0:
        if bin_x[i]==points_x:
            grid[points_x-1,0]+=1.0
            grid_Vx[points_x-1,0]+=Vx[i]
            grid_Vy[points_x-1,0]+=Vy[i]
        else:
            grid[bin_x[i],0]+=dx[i]*dy[i]/(delta_x*dy[i])
            grid_Vx[bin_x[i],0]+=dx[i]*dy[i]/(delta_x*dy[i])*Vx[i]
            grid_Vy[bin_x[i],0]+=dx[i]*dy[i]/(delta_x*dy[i])*Vy[i]
            grid[bin_x[i]-1,0]+=tx[i]*dy[i]/(delta_x*dy[i])
            grid_Vx[bin_x[i]-1,0]+=tx[i]*dy[i]/(delta_x*dy[i])*Vx[i]
            grid_Vy[bin_x[i]-1,0]+=tx[i]*dy[i]/(delta_x*dy[i])*Vy[i]
    elif bin_y[i]==points_y:
        if bin_x[i]==points_x:
            grid[points_x-1,points_y-1]+=1.0
            grid_Vx[points_x-1,points_y-1]+=Vx[i]
            grid_Vx[points_x-1,points_y-1]+=Vy[i]
        else:
            grid[bin_x[i],points_y-1]+=dx[i]*ty[i]/(delta_x*ty[i])
            grid_Vx[bin_x[i],points_y-1]+=dx[i]*ty[i]/(delta_x*ty[i])*Vx[i]
            grid_Vy[bin_x[i],points_y-1]+=dx[i]*ty[i]/(delta_x*ty[i])*Vy[i]
            grid[bin_x[i]-1,points_y-1]+=tx[i]*ty[i]/(delta_x*ty[i])
            grid_Vx[bin_x[i]-1,points_y-1]+=tx[i]*ty[i]/(delta_x*ty[i])*Vx[i]
            grid_Vy[bin_x[i]-1,points_y-1]+=tx[i]*ty[i]/(delta_x*ty[i])*Vy[i]
    elif bin_x[i]==points_x:
        grid[points_x-1,bin_y[i]]+=tx[i]*dy[i]/(delta_y*tx[i])
        grid_Vx[points_x-1,bin_y[i]]+=tx[i]*dy[i]/(delta_y*tx[i])*Vx[i]
        grid_Vy[points_x-1,bin_y[i]]+=tx[i]*dy[i]/(delta_y*tx[i])*Vy[i]
        grid[points_x-1,bin_y[i]-1]+=tx[i]*ty[i]/(delta_y*tx[i])
        grid_Vx[points_x-1,bin_y[i]-1]+=tx[i]*ty[i]/(delta_y*tx[i])*Vx[i]
        grid_Vy[points_x-1,bin_y[i]-1]+=tx[i]*ty[i]/(delta_y*tx[i])*Vy[i]
    else:
        grid[bin_x[i],bin_y[i]]+=dx[i]*dy[i]/area
        grid_Vx[bin_x[i],bin_y[i]]+=dx[i]*dy[i]/area*Vx[i]
        grid_Vy[bin_x[i],bin_y[i]]+=dx[i]*dy[i]/area*Vy[i]
        grid[bin_x[i],bin_y[i]-1]+=dx[i]*ty[i]/area
        grid_Vx[bin_x[i],bin_y[i]-1]+=dx[i]*ty[i]/area*Vx[i]
        grid_Vy[bin_x[i],bin_y[i]-1]+=dx[i]*ty[i]/area*Vy[i]
        grid[bin_x[i]-1,bin_y[i]]+=tx[i]*dy[i]/area
        grid_Vx[bin_x[i]-1,bin_y[i]]+=tx[i]*dy[i]/area*Vx[i]
        grid_Vy[bin_x[i]-1,bin_y[i]]+=tx[i]*dy[i]/area*Vy[i]
        grid[bin_x[i]-1,bin_y[i]-1]+=tx[i]*ty[i]/area
        grid_Vx[bin_x[i]-1,bin_y[i]-1]+=tx[i]*ty[i]/area*Vx[i]
        grid_Vy[bin_x[i]-1,bin_y[i]-1]+=tx[i]*ty[i]/area*Vy[i]

if len(sys.argv)==7:
    background_density=background_density*area

inside=np.where(grid==0.0)
grid[inside]=1e-12

grid_Vx=grid_Vx/grid
grid_Vy=grid_Vy/grid

f=open(fname+'_cic_velocities_map','w')
for i in range(points_x):
    x_middle=min_x+(max_x-min_x)*(i+0.5)/points_x
    for j in range(points_y):
        y_middle=min_y+(max_y-min_y)*(j+0.5)/points_y
        if len(sys.argv)==4:
            f.write(str(x_middle)+' '+str(y_middle)+' '+str(grid[i,j])+' '+str(grid_Vx[i,j])+' '+str(grid_Vy[i,j])+'\n')
        else:
            f.write(str(x_middle)+' '+str(y_middle)+' '+str(grid[i,j]/background_density)+'\n')
f.close()



