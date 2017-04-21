# usage: python nu_masses_hierarchy.py Mnu
import numpy as np
import argparse
from math import sqrt
import scipy.optimize as SO 

def normal(x,Mnu,delta21,delta31):
    m2, m3 = sqrt(delta21+x**2), sqrt(delta31+x**2)
    return Mnu-x-m2-m3

def inverted(x,Mnu,delta21,delta31):
    m1 = sqrt(delta31+x**2);  m2 = sqrt(delta21+m1**2)
    return Mnu-x-m1-m2

parser = argparse.ArgumentParser()
parser.add_argument("Mnu", help="computes m1 m2 and m3 from Mnu")
args = parser.parse_args()

Mnu = float(args.Mnu)

delta21 = 7.5e-5
delta31 = 2.45e-3

M1 = sqrt(delta21)+sqrt(delta31)         #minimum mass for NH
M2 = sqrt(delta31)+sqrt(delta21+delta31) #minimum mass for IH

############ Normal hierarchy #############
if Mnu<M1: print '\nNormal hierarchy non allowed for Mnu = %.4f eV'%Mnu
else:
    m1 = SO.brentq(normal, 0.0, Mnu, args=(Mnu,delta21,delta31), xtol=2e-12, 
                   rtol=8.8817841970012523e-16, maxiter=100, 
                   full_output=False, disp=True)

    m2, m3 = sqrt(delta21+m1**2), sqrt(delta31+m1**2)

    print '\n#### Normal hierarchy ####'
    print 'm1  = %.4f eV'%m1
    print 'm2  = %.4f eV'%m2
    print 'm3  = %.4f eV'%m3
    print 'Mnu = %.4f eV'%(m1+m2+m3)


############ Inverted hierarchy #############
if Mnu<M2: print '\nInverted hierarchy non allowed for Mnu = %.4f eV'%Mnu
else:
    m3 = SO.brentq(inverted, 0.0, Mnu, args=(Mnu,delta21,delta31), xtol=2e-12, 
                   rtol=8.8817841970012523e-16, maxiter=100, 
                   full_output=False, disp=True)

    m1 = sqrt(delta31+m3**2)
    m2 = sqrt(delta21+m1**2)

    print '\n#### Inverted hierarchy ####'
    print 'm1  = %.4f eV'%m1
    print 'm2  = %.4f eV'%m2
    print 'm3  = %.4f eV'%m3
    print 'Mnu = %.4f eV'%(m1+m2+m3)
