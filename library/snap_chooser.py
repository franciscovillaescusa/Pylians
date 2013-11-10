import sys


# VARIABLES:
# Mnu--total neutrino mass: 0.0, 0.3, 0.6, ....
# z--redshift: 0.0, 0.1, 0.2, 1.0, 3.0, ...
# Box--size of the simulation box in Mpc/h: 500, 1000, 100, .... 
# location: file contained in som1/som2/cosmos ...
class snap_chooser:
    def __init__(self,Mnu,z,Box,location):

        RI1={0.0:'17', 0.1:'16', 0.2:'15', 0.3:'14', 0.4:'13',
             0.5:'12', 0.6:'11', 0.7:'10', 0.8:'09', 0.9:'08',
             1.0:'07', 1.2:'06', 1.4:'05', 1.6:'04', 2.0:'03', 
             3.0:'02', 4.0:'01', 5.0:'00'}

        RI2={0.0:'22', 0.1:'21', 0.2:'20', 0.3:'19', 0.4:'18',
             0.5:'17', 0.6:'16', 0.7:'15', 0.8:'14', 0.9:'13',
             1.0:'12', 1.2:'11', 1.4:'10', 1.6:'09', 2.0:'08', 
             3.0:'07', 4.0:'06', 5.0:'05', 8.0:'04', 10.0:'03',
             20.0:'02', 50.0:'01', 90.0:'00'}

        
        ########## 0.0 eV : 500 Mpc/h #########
        if Mnu==0.0 and Box==500:
            if location=='som1':
                folder='/data1/villa/b500p512nu0z99tree'
            elif location=='som2':
                folder='/disk/disksom2/villa/b500p512nu0z99tree'
            elif location=='cosmos':
                folder='/home/cosmos/users/mv249/RUNSG2/Paco/simulations/500Mpc_z=99/b500p512nu0z99'
            else:
                print 'no files in that location'
                sys.exit()

            self.group=folder
            self.snap=folder+'/snapdir_0'+RI1[z]+'/snap_0'+RI1[z]
            self.group_number=int(RI1[z])
        ########## 0.3 eV : 500 Mpc/h #########
        elif Mnu==0.3 and Box==500:
            if location=='som1':
                folder='/data1/villa/b500p512nu0.3z99'
            elif location=='som2':
                folder='/disk/disksom2/villa/b500p512nu0.3z99'
            elif location=='cosmos':
                folder='/home/cosmos/users/mv249/RUNSG2/Paco/simulations/500Mpc_z=99/b500p512nu0.3z99'
            else:
                print 'no files in that location'
                sys.exit()

            self.group=folder            
            self.snap=folder+'/snapdir_0'+RI1[z]+'/snap_0'+RI1[z]
            self.group_number=int(RI1[z])
        ########## 0.6 eV : 500 Mpc/h #########
        elif Mnu==0.6 and Box==500:
            if location=='som1':
                folder='/data1/villa/b500p512nu0.6z99np1024tree'
            elif location=='som2':
                folder='/disk/disksom2/villa/b500p512nu0.6z99np1024tree'
            elif location=='cosmos':
                folder='/home/cosmos/users/mv249/RUNSG2/Paco/simulations/500Mpc_z=99/b500p512nu0.6z99'
            else:
                print 'no files in that location'
                sys.exit()

            self.group=folder
            self.snap=folder+'/snapdir_0'+RI1[z]+'/snap_0'+RI1[z]
            self.group_number=int(RI1[z])



        ########## 0.0 eV : 1000 Mpc/h #########
        elif Mnu==0.0 and Box==1000:
            if location=='som1':
                folder='/data1/villa/b500p512nu0z99tree'
            elif location=='som2':
                folder='/disk/disksom2/villa/b500p512nu0z99tree'
            elif location=='cosmos':
                folder='/home/cosmos/users/mv249/RUNSG2/Paco/simulations/1000Mpc_z=99/CDM'
            else:
                print 'no files in that location'
                sys.exit()

            self.group=folder
            self.snap=folder+'/snapdir_0'+RI2[z]+'/snap_0'+RI2[z]
            self.group_number=int(RI2[z])
        ########## 0.3 eV : 1000 Mpc/h #########
        elif Mnu==0.3 and Box==1000:
            if location=='som1':
                folder='/data1/villa/b500p512nu0z99tree'
            elif location=='som2':
                folder='/disk/disksom2/villa/b500p512nu0z99tree'
            elif location=='cosmos':
                folder='/home/cosmos/users/mv249/RUNSG2/Paco/simulations/1000Mpc_z=99/NU0.3'
            else:
                print 'no files in that location'
                sys.exit()

            self.group=folder
            self.snap=folder+'/snapdir_0'+RI2[z]+'/snap_0'+RI2[z]
            self.group_number=int(RI2[z])
        ########## 0.6 eV : 1000 Mpc/h #########
        elif Mnu==0.6 and Box==1000:
            if location=='som1':
                folder='/data1/villa/b500p512nu0z99tree'
            elif location=='som2':
                folder='/disk/disksom2/villa/b500p512nu0z99tree'
            elif location=='cosmos':
                folder='/home/cosmos/users/mv249/RUNSG2/Paco/simulations/1000Mpc_z=99/NU0.6'
            else:
                print 'no files in that location'
                sys.exit()

            self.group=folder
            self.snap=folder+'/snapdir_0'+RI2[z]+'/snap_0'+RI2[z]
            self.group_number=int(RI2[z])


        ##### IF NO FILES WITH THAT CHARACTERISTICS #####
        else:
            'no files with that neutrino masses'
            sys.exit()
            
        
##### EXAMPLE OF USAGE #####
"""
Mnu=0.0; 
z=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
   0.9, 1.0, 1.2, 1.4, 1.6, 2.0, 3.0, 4.0, 5.0]
location='cosmos'
Box=1000

for Z in z:
    print Z
    F=snap_chooser(Mnu,Z,Box,location)

    snapshot_fname=F.snap
    groups_fname=F.group
    groups_number=F.group_number

    print snapshot_fname
    print groups_fname
    print groups_number
    print ' '
"""
