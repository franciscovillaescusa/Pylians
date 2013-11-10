#This code is used to compute the mean number of galaxies with luminosities
#above a certain threshold. It uses the Press-Schechter luminosity function
#with parameters obtained from Blanton et al 2003. For details see:
# http://www.astro.virginia.edu/class/whittle/astr553/Topic04/Lecture_4.html

import numpy as np
import mpmath #used to compute the incomplete gamma function

########################### INPUT ##############################
M = -20.5 #This is maximum magnitude of the galaxies in the sample
################################################################

#### Press-Schechter parameter: taken form Blanton et al. 2003
M_star = -20.44 
alpha = -1.05   
n_star = 1.49e-2 #galaxies h^3 / Mpc^3


ratio_L = 10**(0.4*(M_star-M))

density = n_star * mpmath.gammainc(1.0+alpha,ratio_L) #galaxies h^3 / Mpc^3

print density,'galaxies / (Mpc/h)^3 with M - 5*log10(h) <',M
