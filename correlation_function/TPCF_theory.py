import numpy as np
import scipy.integrate as si
import Power_spectrum_library as PSL
import sys,os


def deriv1(y,x,k,Pk,R):
    Pkp = 10**(np.interp(np.log10(x),np.log10(k),np.log10(Pk)))
    kR=x*R;  return [x**2*Pkp*np.sin(kR)/kR]
    #kR=x*R;  return [x**2*Pkp*np.sin(kR)/kR*np.exp(-x**2/5.0**2)]
    
def deriv2(y,x,k,Pk,R,A,ns):
    if x<k[0]:  Pkp = A*x**ns
    else:       Pkp = 10**(np.interp(np.log10(x),np.log10(k),np.log10(Pk)))
    kR=x*R;  return [x**2*Pkp*np.sin(kR)/kR]


def correlation_function_from_Pk(k,Pk,R,A,ns):
    yinit = [0.0]
    if A==None:    #if no extrapolation of P(k) on large scales
        k_limits = [k[0], k[-1]]
        I = si.odeint(deriv1,yinit,k_limits,args=(k,Pk,R),
                      rtol=1e-9,atol=1e-12,mxstep=10000000)[1][0]/(2.0*np.pi**2)
    else:          #if extrapolation of P(k) on large scales
        k_limits = [1e-12, k[-1]]
        I = si.odeint(deriv2,yinit,k_limits,args=(k,Pk,R,A,ns),
                      rtol=1e-9,atol=1e-12,mxstep=10000000)[1][0]/(2.0*np.pi**2)
    return I


################################# INPUT ######################################
f_Pk       = '../CAMB_TABLES/ics_matterpow_0.dat'
f_transfer = '../CAMB_TABLES/ics_transfer_0.dat'

Rmin    = 1e-3   #Mpc/h
Rmax    = 500.0  #Mpc/h 
bins    = 1000   
binning = 'log'  #choose between 'linear' and 'log'

Omega_CDM = None    #set only to compute the CDM+B xi(r)
Omega_B   = None    #set only to compute the CDM+B xi(r)
ns        = None    #set only to extrapole the P(k) on very large scales

#choose one or several from: 'm','c','b','n','cb','c-b','c-n','b-n','cb-n'
object_type = ['m']

fout = ['CF_m_z=0.dat']
##############################################################################

#compute the different power spectra from the CAMB files
CAMB = PSL.CAMB_Pk(f_Pk,f_transfer,Omega_CDM,Omega_B)

#dictionary to read the different power spectra
k = CAMB.k
Pk_dict = {'m':CAMB.Pk_m,        'c':CAMB.Pk_c,
           'b':CAMB.Pk_b,        'n':CAMB.Pk_n,
           'c-b':CAMB.Pk_x_c_b,  'c-n':CAMB.Pk_x_c_n,  
           'b-n':CAMB.Pk_x_b_n}

if Omega_CDM!=None and Omega_B!=None:
    Pk_dict.update({'cb':CAMB.Pk_cb,  'cb-n':CAMB.Pk_x_cb_n})

#find the bins where to compute the correlation function
if binning == 'log':    R = np.logspace(np.log10(Rmin),np.log10(Rmax),bins)
else:                   R = np.linspace(Rmin,Rmax,bins)


#make a loop over all the different objects
for i,obj in enumerate(object_type):
    print 'Computing the correlation function of the object:',obj
    Pk = Pk_dict[obj]

    #Extrapolate the P(k) on very large scales: P(k) = A*k^ns
    if ns!=None:   A = Pk[0]/k[0]**ns
    else:          A = None

    #if the file exists delete it
    if os.path.exists(fout[i]):  os.system('rm '+fout[i])

    #compute the correlation function
    for r in R:
        xi = correlation_function_from_Pk(k,Pk,r,A,ns);  print r,xi
        f=open(fout[i],'a');  f.write(str(r)+' '+str(xi)+'\n');  f.close()



