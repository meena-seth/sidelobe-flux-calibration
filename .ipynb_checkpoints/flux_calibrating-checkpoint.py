import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pdb
import copy
import datetime
from iautils import cascade
from scipy.stats import iqr 
from datetime import datetime
from astropy.coordinates import SkyCoord

sys.path.insert(0, os.path.abspath('beam-model'))
from beam_model import utils, formed
from beam_model import config


#### Load in files to observations ####
path = "/arc/projects/chime_frb/adamdong/for_meena/no_dedisperse"
norescale = 'norescale'
crab_norescaled_filepaths = []
crab_rescaled_filepaths = []
for (root, dirs, file) in os.walk(path):
    for f in file: 
        if f.endswith('.npz'):
            if norescale in f:
                crab_norescaled_filepaths.append(os.path.join(root, f)) 
            else:
                crab_rescaled_filepaths.append(os.path.join(root, f)) 
        else:
            continue

#### Predefined stuff ####

source_name = "TAU_A"
coords = SkyCoord.from_name(source_name)
source_ra = coords.ra.deg
source_dec = coords.dec.deg

freqs = np.linspace(400.390625, 800, 1024) #1024 frequencies
has_list = np.linspace(-105, 104.90278, 2160)   #2160 HAs in holography data

def bf_to_jy(bf_spectrum, f_good):
    factor = (np.square(1024) * 128) / (np.square(4) * 0.806745 * 400)
    result = bf_spectrum / ( factor * np.square(f_good) ) 
    return result

def flux_to_luminosity(peak_flux):
    result = 4 * np.pi * np.square(6.171 * 10**19) * peak_flux * 10**(-19)
    return result 

def normalise(spectrum):
    spectrum -=np.nanmedian(spectrum,axis=1)[:,np.newaxis]
    spectrum /=np.nanstd(spectrum,axis=1)[:,np.newaxis]
    return spectrum

#### Load in beam response #####

holography_data = np.load('/arc/projects/chime_frb/mseth/Holography_Data.npz')
intensity_norm = holography_data['intensity_norm']  #Primary beam response (1024 frequencies, 2160 HAs)

#### Getting initial parameters #### 
fluences = []
scaled_fluxes = []
peak_luminosities = []

event_timestamps = []
beam_ids = []
has = []
y_at_peak = []
mjds =[]
peak_idxs = []

for file in crab_norescaled_filepaths[31:32]:
    # Get file name and index 
    filename = file.split("/")
    mjd = filename[7].split("_")[1].split(".")[0]
    i = crab_norescaled_filepaths.index(file)
    ##
    
    cascade_obj = cascade.load_cascade_from_file(file)    
    beam = cascade_obj.beams[0]
    
    cascade_obj.dm = 56.7
    beam.subband(1024,56.7,apply_weights=False)  #Downsample to 1024 frequency
    
    cascade_obj.dm = 0
    
    ## RFI MASKING (at DM=0) ##
    ds_before = cascade_obj.beams[0].intensity
    ds_copy = copy.deepcopy(ds_before)
    ds_copy = normalise(ds_copy)
    ts_before = np.nanmean(ds_copy, axis=0)

    ts_median = np.nanmedian(ts_before)
    ts_iqr = iqr(ts_before)
    ts_difference = np.abs(ts_before - ts_median)
    ts_limit = ts_iqr * 2
    ts_mask = np.where(ts_difference >= ts_limit)
    
    ds_masked = copy.deepcopy(ds_before)
    ds_masked[:, ts_mask] = np.nan
    
    ## DEDISPERSE TO CRAB DM ##
    cascade_obj.beams[0].intensity = ds_masked
    cascade_obj.dm = 56.7 
    #beam.subband(1024,56.7,apply_weights=False)  #Downsample to 1024 frequency
    
    # Subtract off-pulse mean from dynamic spectrum
    offpulse_ds = ds_masked[:, 1500:1550]
    offpulse_mean = np.nanmean(offpulse_ds, axis=1)
    ds_masked = ds_masked - offpulse_mean[:, np.newaxis]
    
    # Check masking     
    plt.figure()
    plt.imshow(ds_masked, aspect='auto')
    #plt.savefig(f"{i}_{mjd}_mask.png") #DS after masking & dedispersing
        
    # Get parameters for later
    event_timestamp = cascade_obj.event_time 
    beam_id = int(beam.beam_no)
    ha, y = utils.get_position_from_equatorial(source_ra, source_dec, event_timestamp)
    
    pdb.set_trace()
    ## PRIMARY BEAM ##
    ha_idx = np.abs(has_list - ha).argmin()
    beam_response = intensity_norm[:, ha_idx] #[1024,]
    beam_response[beam_response==0] = np.nan
    
    ## CORRECTING ##
    #only calibrating lower half of the band 
    
    beam_response[0:20] = beam_response[80:100]
    ds_corrected = ds_masked[0:512] / beam_response[0:512, np.newaxis] 
    ds_calibrated = bf_to_jy(ds_corrected, 1)
    ts_calibrated = np.nanmean(ds_calibrated, axis=0)
        
    ## PLOTTING ##
    peak_idx = np.nanargmax(ts_calibrated)
    
    #DS after masking, dedispersing, and calibrating. (Normalised & zoomed in)
    
    plt.figure()
    im = plt.imshow(normalise(ds_calibrated[:, peak_idx-100:peak_idx+100]), aspect='auto',cmap="YlGnBu")
    cbar = plt.colorbar(im)
    cbar.set_label("Flux (Jy)")
    plt.ylabel("Frequency Bins")
    plt.xlabel("Time sample")
    plt.title(f"{i}_{mjd}, centered on t={peak_idx}") 
    #plt.savefig(f"{i}_{mjd}_ds_calibrated.png")
    
    plt.figure()
    plt.plot(beam_response[0:512])
    plt.yscale('log')
    #plt.savefig("beam_response")
        
    #Time series 
    plt.figure()
    plt.plot(ts_calibrated / 1000 *5)
    plt.ylabel("Flux (kJy)")
    plt.xlabel("Time sample")
    plt.title(f"{i}_{mjd}")
    plt.savefig(f"{i}_{mjd}_ts.png")
    
    ## CALCULATING STUFF 
    
    ind_max = peak_idx+2*cascade_obj.best_width
    ind_min = peak_idx-2*cascade_obj.best_width
    integral = np.trapz(ts_calibrated[ind_min:ind_max])
    fluence = integral * 0.9830400000000001 /1000  * 5 #Jy-s
    
    flux = np.nanmax(ts_calibrated) * 5
    peak_luminosity = flux_to_luminosity(flux)
    
    ## SAVING 
    fluences.append(fluence)
    scaled_fluxes.append(flux)
    peak_luminosities.append(peak_luminosity)
    
    event_timestamps.append(event_timestamp)
    beam_ids.append(beam_id)
    has.append(ha)
    y_at_peak.append(y)
    mjds.append(mjd)
    peak_idxs.append(peak_idx)
    
    print(f"Spectra for {file} calibrated!")
    print(f'''
    Observation: {i}_{mjd}
    Flux = {flux}
    Fluence = {fluence}
    Luminosity = {peak_luminosity}
    ''')
    del cascade_obj 
    del ds_corrected
    del ds_masked 
    del ds_before
    continue 
    
np.savez("rfi_corrected_calibration.npz", mjds=mjds, has=has, y_at_peak=y_at_peak, event_timestamps=event_timestamps, peak_idxs=peak_idxs, beam_ids=beam_ids, scaled_fluxes=scaled_fluxes, peak_luminosity=peak_luminosities, fluence=fluences)
