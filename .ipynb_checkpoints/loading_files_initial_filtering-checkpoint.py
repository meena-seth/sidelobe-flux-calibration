import numpy as np
import os
import sys 
import pdb
from iautils import cascade


## Path to folder with all the files
path = '/arc/projects/chime_frb/adamdong/for_meena/old_cascades'

## Separate out rescaled vs. norescaled files 
norescale = 'norescale'
frb = 'frb' 

rescaled_filepaths = []
for (root, dirs, file) in os.walk(path):
    for f in file: 
        if norescale in f:
            continue 
        if frb in f:
            continue
        else:
            if f.endswith('.npz'):
                rescaled_filepaths.append(os.path.join(root, f))
            else:
                continue
            
norescaled_filepaths = []
for (root, dirs, file) in os.walk(path):
    for f in file: 
        if norescale in f:
            if f.endswith('.npz'):
                norescaled_filepaths.append(os.path.join(root, f)) 
            else:
                continue
        else:
            continue
            
pdb.set_trace()

            
## Do initial DM filtering on the rescaled files 
crab_rescaled_filepaths = []
dms = []
for filepath in rescaled_filepaths:
    try:
        cascade_obj = cascade.load_cascade_from_file(filepath) 
        dm = cascade_obj.best_snr_dm
        
        if 55 <= dm <= 58:
            dms.append(dm)
            crab_rescaled_filepaths.append(filepath)
            print(f"{filepath} good")
            del cascade_obj
            continue
        else:
            print(f"{filepath} bad")
            continue 
        
    except Exception as e:
        print(e)
        continue
        
        
## Now use that to get list of crab norescaled files 
crabfiles_mjd = []
for crab_filepath in crab_rescaled_filepaths:  #Get mjd from each file name
    file = crab_filepath.split("/")
    mjd = file[7].split("_")[1].split(".")[0]
    crabfiles_mjd.append(mjd)
    
crab_norescaled_filepaths = []
for mjd in crabfiles_mjd:
    for filepath in norescaled_filepaths:     #Get norescaled with same mjd
        if mjd in filepath:
            crab_norescaled_filepaths.append(filepath)
            
        else:
            continue
            
            
pdb.set_trace()

            
## Saving 

#np.savez("New_Crab_rescaled_filepaths.npz", filepath = crab_rescaled_filepaths)
np.savez("Old_Crab_norescaled_filepaths.npz", filepath = crab_norescaled_filepaths)