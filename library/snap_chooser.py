import sys


# VARIABLES:
# Mnu--total neutrino mass: 0.0, 0.3, 0.6, ....
# z--redshift: 0.0, 0.1, 0.2, 1.0, 3.0, ...
# som: file contained in som1/som2/cosmos ...
class snap_chooser:
    def __init__(self,Mnu,z,som):

        ########## 0.0 eV #########
        if Mnu==0.0:
            if som=='som1':
                folder='/data1/villa/b500p512nu0z99tree'
            elif som=='som2':
                folder='/disk/disksom2/villa/b500p512nu0z99tree'
            else:
                print 'no files in that som'
                sys.exit()

            self.group=folder

            if z==0.0:
                self.snap=folder+'/snapdir_017/snap_017'
                self.group_number=17
            elif z==0.1:
                self.snap=folder+'/snapdir_016/snap_016'
                self.group_number=16
            elif z==0.2:
                self.snap=folder+'/snapdir_015/snap_015'
                self.group_number=15
            elif z==0.3:
                self.snap=folder+'/snapdir_014/snap_014'
                self.group_number=14
            elif z==0.4:
                self.snap=folder+'/snapdir_013/snap_013'
                self.group_number=13
            elif z==0.5:
                self.snap=folder+'/snapdir_012/snap_012'
                self.group_number=12
            elif z==0.6:
                self.snap=folder+'/snapdir_011/snap_011'
                self.group_number=11
            elif z==0.7:
                self.snap=folder+'/snapdir_010/snap_010'
                self.group_number=10
            elif z==0.8:
                self.snap=folder+'/snapdir_009/snap_009'
                self.group_number=9
            elif z==0.9:
                self.snap=folder+'/snapdir_008/snap_008'
                self.group_number=8
            elif z==1.0:
                self.snap=folder+'/snapdir_007/snap_007'
                self.group_number=7
            elif z==1.2:
                self.snap=folder+'/snapdir_006/snap_006'
                self.group_number=6
            elif z==1.4:
                self.snap=folder+'/snapdir_005/snap_005'
                self.group_number=5
            elif z==1.6:
                self.snap=folder+'/snapdir_004/snap_004'
                self.group_number=4
            elif z==2.0:
                self.snap=folder+'/snapdir_003/snap_003'
                self.group_number=3
            elif z==3.0:
                self.snap=folder+'/snapdir_002/snap_002'
                self.group_number=2
            elif z==4.0:
                self.snap=folder+'/snapdir_001/snap_001'
                self.group_number=1
            elif z==5.0:
                self.snap=folder+'/snapdir_000/snap_000'
                self.group_number=0
            else:
                print 'no files with at that redshift'



        ########## 0.3 eV #########
        elif Mnu==0.3:
            if som=='som1':
                folder='/data1/villa/b500p512nu0.3z99'
            elif som=='som2':
                folder='/disk/disksom2/villa/b500p512nu0.3z99'
            else:
                print 'no files in that som'
                sys.exit()

            self.group=folder            
            
            if z==0.0:
                self.snap=folder+'/snapdir_017/snap_017'
                self.group_number=17
            elif z==0.1:
                self.snap=folder+'/snapdir_016/snap_016'
                self.group_number=16
            elif z==0.2:
                self.snap=folder+'/snapdir_015/snap_015'
                self.group_number=15
            elif z==0.3:
                self.snap=folder+'/snapdir_014/snap_014'
                self.group_number=14
            elif z==0.4:
                self.snap=folder+'/snapdir_013/snap_013'
                self.group_number=13
            elif z==0.5:
                self.snap=folder+'/snapdir_012/snap_012'
                self.group_number=12
            elif z==0.6:
                self.snap=folder+'/snapdir_011/snap_011'
                self.group_number=11
            elif z==0.7:
                self.snap=folder+'/snapdir_010/snap_010'
                self.group_number=10
            elif z==0.8:
                self.snap=folder+'/snapdir_009/snap_009'
                self.group_number=9
            elif z==0.9:
                self.snap=folder+'/snapdir_008/snap_008'
                self.group_number=8
            elif z==1.0:
                self.snap=folder+'/snapdir_007/snap_007'
                self.group_number=7
            elif z==1.2:
                self.snap=folder+'/snapdir_006/snap_006'
                self.group_number=6
            elif z==1.4:
                self.snap=folder+'/snapdir_005/snap_005'
                self.group_number=5
            elif z==1.6:
                self.snap=folder+'/snapdir_004/snap_004'
                self.group_number=4
            elif z==2.0:
                self.snap=folder+'/snapdir_003/snap_003'
                self.group_number=3
            elif z==3.0:
                self.snap=folder+'/snapdir_002/snap_002'
                self.group_number=2
            elif z==4.0:
                self.snap=folder+'/snapdir_001/snap_001'
                self.group_number=1
            elif z==5.0:
                self.snap=folder+'/snapdir_000/snap_000'
                self.group_number=0
            else:
                print 'no files with at that redshift'



        ########## 0.6 eV #########
        elif Mnu==0.6:
            if som=='som1':
                folder='/data1/villa/b500p512nu0.6z99np1024tree'
            elif som=='som2':
                folder='/disk/disksom2/villa/b500p512nu0.6z99np1024tree'
            else:
                print 'no files in that som'
                sys.exit()

            self.group=folder            
            
            if z==0.0:
                self.snap=folder+'/snapdir_017/snap_017'
                self.group_number=17
            elif z==0.1:
                self.snap=folder+'/snapdir_016/snap_016'
                self.group_number=16
            elif z==0.2:
                self.snap=folder+'/snapdir_015/snap_015'
                self.group_number=15
            elif z==0.3:
                self.snap=folder+'/snapdir_014/snap_014'
                self.group_number=14
            elif z==0.4:
                self.snap=folder+'/snapdir_013/snap_013'
                self.group_number=13
            elif z==0.5:
                self.snap=folder+'/snapdir_012/snap_012'
                self.group_number=12
            elif z==0.6:
                self.snap=folder+'/snapdir_011/snap_011'
                self.group_number=11
            elif z==0.7:
                self.snap=folder+'/snapdir_010/snap_010'
                self.group_number=10
            elif z==0.8:
                self.snap=folder+'/snapdir_009/snap_009'
                self.group_number=9
            elif z==0.9:
                self.snap=folder+'/snapdir_008/snap_008'
                self.group_number=8
            elif z==1.0:
                self.snap=folder+'/snapdir_007/snap_007'
                self.group_number=7
            elif z==1.2:
                self.snap=folder+'/snapdir_006/snap_006'
                self.group_number=6
            elif z==1.4:
                self.snap=folder+'/snapdir_005/snap_005'
                self.group_number=5
            elif z==1.6:
                self.snap=folder+'/snapdir_004/snap_004'
                self.group_number=4
            elif z==2.0:
                self.snap=folder+'/snapdir_003/snap_003'
                self.group_number=3
            elif z==3.0:
                self.snap=folder+'/snapdir_002/snap_002'
                self.group_number=2
            elif z==4.0:
                self.snap=folder+'/snapdir_001/snap_001'
                self.group_number=1
            elif z==5.0:
                self.snap=folder+'/snapdir_000/snap_000'
                self.group_number=0
            else:
                print 'no files with at that redshift'


        ##### IF NO FILES WITH THAT NEUTRINO MASSES #####
        else:
            'no files with that neutrino masses'
            sys.exit()
            
        
##### EXAMPLE OF USAGE #####
"""
Mnu=0.0; 
z=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
   0.9, 1.0, 1.2, 1.4, 1.6, 2.0, 3.0, 4.0, 5.0]
som='som2'

for Z in z:
    F=snap_chooser(Mnu,Z,som)

    snapshot_fname=F.snap
    groups_fname=F.group
    groups_number=F.group_number

    print snapshot_fname
    print groups_fname
    print groups_number
    print ' '
"""
