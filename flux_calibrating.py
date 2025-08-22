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
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

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
chime_location = EarthLocation.of_address("Dominion Radio Astrophysical Observatory, British Columbia")

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

    offpulse_ds = ds_masked[:, 1500:1550] #Subtract off-pulse mean
    offpulse_mean = np.nanmean(offpulse_ds, axis=1)
    ds_masked = ds_masked - offpulse_mean[:, np.newaxis]
 
    # Get parameters for later
    event_timestamp = cascade_obj.event_time 
    event_time = Time(event_timestamp, scale='utc', location=chime_location)
    sidereal_time = event_time.sidereal_time('apparent').deg
    ha = sidereal_time - source_ra
    beam_id = int(beam.beam_no)
    
    ## PRIMARY BEAM  & CORRECTING ##
    ha_idx = np.abs(has_list - ha).argmin()

    ha_idxs = np.arange(ha_idx-120, ha_idx+121) #For a variety of HAs
    fluxes = []
    for i in ha_idxs:  
        beam_response = intensity_norm[:, i] #[1024,]
        beam_response[0:20] = beam_response[80:100]

        beam_response[beam_response==0] = np.nan
    
        ds_corrected = ds_masked[0:512] / beam_response[0:512, np.newaxis] 
        ds_calibrated = bf_to_jy(ds_corrected, 1)
        ts_calibrated = np.nanmean(ds_calibrated, axis=0)
        
        flux = np.nanmax(ts_calibrated) * 5 /1000
        fluxes.append(flux)
    fluxes[42]=np.nan    
    beam_response_center = intensity_norm[:, ha_idx]
    beam_response_center[beam_response_center==0] = np.nan
    beam_response_center[0:20] = beam_response_center[80:100]
    ds_corrected_center = ds_masked[0:512] / beam_response_center[0:512, np.newaxis]
    
    ds_calibrated_center = bf_to_jy(ds_corrected_center, 1)
    ts_calibrated_center = np.nanmean(ds_calibrated_center, axis=0)
    peak_idx = np.argmax(ts_calibrated_center)
    
    ## PLOTTING ##
    
    # How flux changes vs. what HA we use in holography    
    plt.figure()
    plt.scatter(has_list[ha_idxs], fluxes)
    plt.scatter(has_list[ha_idx], fluxes[120], color='r')
    plt.ylabel("Flux (kJy)")
    plt.xlabel("HA used to calibrate")
    plt.savefig("/arc/projects/chime_frb/mseth/plots/HA_vs_flux.png")
    
    pdb.set_trace()
    
    # Comparing spectrum at peak time to holography at peak HA 
    
    fig, ax = plt.subplot_mosaic(
        '''
        A
        B
        ''',
        constrained_layout = True,
        figsize = (10, 8), 
        sharex = True)
    
    ax['A'].plot(ds_calibrated_center[:, peak_idx] / np.nanmax(ds_calibrated_center[:, peak_idx], axis=0))
    ax['A'].set_ylabel('Normalised flux')
    

    ax['B'].plot(beam_response_center[0:512], color='r', label="HA=-53")
    ax['B'].set_yscale('log')
    ax['B'].set_ylabel('Normalised sensitivity')
    ax['B'].set_xlabel('Frequency bins')
    ax['B'].legend
    
    plt.suptitle(f"""{i}_{mjd} at t={peak_idx}, HA={ha_idx}
    Normalised spectrum & beam response""")
    plt.savefig(f"{i}_{mjd}_spectrum")
    plt.close()
        
    pdb.set_trace()
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
    
pdb.set_trace()

np.savez("rfi_corrected_calibration_maxfreqs.npz", scaled_fluxes=scaled_fluxes)

#np.savez("rfi_corrected_calibration.npz", mjds=mjds, has=has, y_at_peak=y_at_peak, event_timestamps=event_timestamps, peak_idxs=peak_idxs, beam_ids=beam_ids, scaled_fluxes=scaled_fluxes, peak_luminosity=peak_luminosities, fluence=fluences)
