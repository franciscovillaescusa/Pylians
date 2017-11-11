# This script can be used to merge subfiles in FoF catalogues. It will allow to
# read the catalogue much faster and will ocupy less space
# USAGE: python FoF_file_merger.py snapdir snapnums
# optional arguments: long_ids, swap, SFR. Type FoF_file_merger.py -h for help
# e.g. python FoF_file_merger.py ./ -snapnums 0 1 2 3 4 5 --long_ids
import argparse
import numpy as np
import readfof
import sys,os

parser = argparse.ArgumentParser()
parser.add_argument("snapdir", help="folder where the groups_XXX folder is")
parser.add_argument("-snapnums", nargs='+', type=int, 
                    help="groups number; python list")
parser.add_argument("--swap", dest="swap", action="store_true", default=False, 
                    help="False by default. Set --swap for True")
parser.add_argument("--SFR", dest="SFR", action="store_true", default=False, 
                    help="False by default. Set --SFR for True")
parser.add_argument("--long_ids", dest="long_ids", action="store_true", 
                    default=False, help="False by default. Set --long_ids for True")
args = parser.parse_args()


snapdir = args.snapdir

for snapnum in args.snapnums:

    # find the names of the old and new group folders
    FoF_folder     = snapdir+'groups_%03d'%snapnum
    old_FoF_folder = snapdir+'original_groups/'

    # create original FoF folder
    if os.path.exists(old_FoF_folder+'groups_%03d'%snapnum):
        raise Exception('files already merged!')

    # create new FoF file
    f_tab = '%s/group_tab_%03d.0'%(snapdir,snapnum)
    f_ids = '%s/group_ids_%03d.0'%(snapdir,snapnum)
    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=args.long_ids,
                              swap=args.swap, SFR=args.SFR)
    Ngroups_in = FoF.TotNgroups
    Nids_in    = FoF.TotNids
    readfof.writeFoFCatalog(FoF, f_tab, idsFile=f_ids);  del FoF
           
    # rename FoF folder, create new FoF folder and move files to it
    if not(os.path.exists(old_FoF_folder)):
        os.system('mkdir %s'%old_FoF_folder)
    os.system('mv '+FoF_folder+' '+old_FoF_folder)
    os.system('mkdir '+FoF_folder)
    os.system('mv '+f_tab+' '+f_ids+' '+FoF_folder)

    # check that number of halos are the same in both folders
    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=args.long_ids, 
                              swap=args.swap, SFR=args.SFR)
    Ngroups_out = FoF.TotNgroups
    Nids_out    = FoF.TotNids;  del FoF

    if Ngroups_in!=Ngroups_out:
        raise Exception('Number of FoF halos is different in new/old files!!!')
    if Nids_in!=Nids_out:
        raise Exception('Number of FoF halos is different in new/old files!!!')
        
    print '\nFiles correctly merged!!!'
    print '# of halos in original file = %ld'%Ngroups_in
    print '# of halos in new file      = %ld'%Ngroups_out
    print '# of ids   in original file = %ld'%Nids_in
    print '# of ids   in new file      = %ld'%Nids_out
