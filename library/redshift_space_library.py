#This library contains the routines used to compute quantities in redshift-space

############## AVAILABLE ROUTINES ##############
#pos_redshift_space
################################################



###############################################################################

#This routine receives the positions of the particles in configuration-space
#and return them in redshift-space
#pos --------------------> array with the positions of the particles
#vel --------------------> array with the velocities of the particles
#BoxSize ----------------> size of the simulation box
#Hubble -----------------> value of H(z) in (km/s)/(Mpc/h)
#redshift ---------------> redshift
#axis -------------------> axis along which perform the RSD
#The routines just uses: s = r + (1+z)/H(z)*v
def pos_redshift_space(pos,vel,BoxSize,Hubble,redshift,axis):
    #transform coordinates to redshift space
    delta_y=(vel[:,axis]/Hubble)*(1.0+redshift)  #displacement in Mpc/h
    pos[:,axis]+=delta_y #add distorsion to position of particle in real-space
    del delta_y

    #take care of the boundary conditions
    beyond=np.where(pos[:,axis]>BoxSize)[0]; pos[beyond,axis]-=BoxSize
    beyond=np.where(pos[:,axis]<0.0)[0];     pos[beyond,axis]+=BoxSize
    del beyond
###############################################################################
