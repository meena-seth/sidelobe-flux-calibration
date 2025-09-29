import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pdb
import copy
import datetime
import scipy.signal 
from iautils import cascade
from scipy.stats import iqr 
from datetime import datetime
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages


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
    #result = 4 * np.pi * np.square(6.171 * 10**19) * peak_flux * 10**(-19)
    result = 4 * np.pi * np.square(6.788 * 10**19) * peak_flux * 10**(-19)
    return result 

def normalise(spectrum):
    spectrum -=np.nanmedian(spectrum,axis=1)[:,np.newaxis]
    spectrum /=np.nanstd(spectrum,axis=1)[:,np.newaxis]
    return spectrum

def save_image(filename):
    p = PdfPages(filename)
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs: 
        fig.savefig(p, format='pdf') 
    p.close()  
    
        
#### Load in beam response #####

holography_data = np.load('/arc/projects/chime_frb/mseth/Holography_Data.npz')
intensity_norm = holography_data['intensity_norm']  #Primary beam response (1024 frequencies, 2160 HAs)

#### Getting initial parameters #### 
fluences = []
scaled_fluxes = []
peak_luminosities = []
peakfreq_fluxes = []
peakfreq_luminosities = []

flux_errors = []
fluence_errors = []
lum_errors = []

Nimmo_x = []


event_timestamps = []
beam_ids = []
has = []
y_at_peak = []
mjds =[]
peak_idxs = []

f'''
for file in crab_norescaled_filepaths:
    # Get file name and index 
    filename = file.split("/")
    mjd = filename[7].split("_")[1].split(".")[0]
    i = crab_norescaled_filepaths.index(file)
    ##
     
    cascade_obj = cascade.load_cascade_from_file(file)    
    beam = cascade_obj.beams[0]
    
    cascade_obj.dm = 56.7
    beam.subband(1024,56.7,apply_weights=False)  #Downsample to 1024 frequency
        
    ### RFI MASKING DATA ###
    cascade_obj.dm = 0  # Using DM 0 
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
    for mask_idx in ts_mask[0]:
        ds_masked[:, mask_idx] = np.nan
        
    ## DEDISPERSE TO CRAB DM ##
    cascade_obj.beams[0].intensity = ds_masked
    cascade_obj.dm = 56.7 
    
    # Subtract median around pulse from dynamic spectrum
    initial_peak_idx = np.nanargmax((np.nansum(ds_masked, axis=1))) 
    offpulse_median = np.nanmedian(ds_masked[:, initial_peak_idx-200:initial_peak_idx+200], axis=1)
    ds_masked = ds_masked - offpulse_median[:, np.newaxis]
        
    width = cascade_obj.best_width * 0.9830400000000001 / 1000 * 0.4
    Nimmo_x.append(width)
    
    del cascade_obj
    del ds_masked 
    continue 

    
    
np.savez("/arc/projects/chime_frb/mseth/widths.npz", Nimmo_x=Nimmo_x)

pdb.set_trace()
'''

for file in crab_norescaled_filepaths[34:35]:
    # Get file name and index 
    filename = file.split("/")
    mjd = filename[7].split("_")[1].split(".")[0]
    i = crab_norescaled_filepaths.index(file)
    ##
     
    cascade_obj = cascade.load_cascade_from_file(file)    
    beam = cascade_obj.beams[0]
    
    cascade_obj.dm = 56.7
    beam.subband(1024,56.7,apply_weights=False)  #Downsample to 1024 frequency
        
    ### RFI MASKING DATA ###
    cascade_obj.dm = 0  # Using DM 0 
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
    for mask_idx in ts_mask[0]:
        ds_masked[:, mask_idx] = np.nan
        
    ## DEDISPERSE TO CRAB DM ##
    cascade_obj.beams[0].intensity = ds_masked
    cascade_obj.dm = 56.7 
    
    # Subtract median around pulse from dynamic spectrum
    initial_peak_idx = np.nanargmax((np.nansum(ds_masked, axis=1))) 
    offpulse_median = np.nanmedian(ds_masked[:, initial_peak_idx-200:initial_peak_idx+200], axis=1)
    ds_masked = ds_masked - offpulse_median[:, np.newaxis]
    
    #DS after masking & dedispersing     
    plt.figure()
    plt.title("DS after masking")
    plt.imshow(ds_masked, aspect='auto')
    
    # Get parameters for later
    event_timestamp = cascade_obj.event_time 
    event_time = Time(event_timestamp, scale='utc', location=chime_location)
    sidereal_time = event_time.sidereal_time('apparent').deg
    ha = sidereal_time - source_ra
    if ha > 180:
        ha = -1 * np.abs(360 - ha)
        
    ### PRIMARY BEAM ##
    if ha > 90 or ha < -90:
        print(f'''
                HA of {ha} out of bounds.
                Averaging holography from 80-90 degrees to get lower limit.
                ''')
        
    ha_idx = np.abs(has_list - ha).argmin()
    center_has = np.arange(ha_idx-6, ha_idx+7)
    beam_id = int(beam.beam_no)
        
        
    intensity_norm[np.log10(intensity_norm)>1] = np.nan
    
    #Removing RFI from holography
    fluxes = []
    timeseries = []
    for center_ha in center_has:
        masked_beams = []
        
        # AVERAGING HOLOGRAPHY
        if ha > 90 or ha < -90:
            #Average holography between 80-90 degrees
            ha_idxs =  np.arange(1900, 2000) 
        else:
            #Average holography 1 degree around each HA 
            ha_idxs = np.arange(center_ha - 6, center_ha + 6)
        
        for index in ha_idxs:
            beam_response = intensity_norm[0:512, index]
            beam_response[beam_response==0] = np.nan
            beam_copy = copy.deepcopy(beam_response)
            peaks, properties = scipy.signal.find_peaks(beam_copy, prominence=0.0004, width=0.001)
            widths = properties['widths']

            for peak, width in zip(peaks, widths):
                beam_slice = beam_copy[peak-20:peak-10]
                median = np.nanmedian(beam_slice)

                lower_ind = np.round(peak - 7* width).astype(int)
                upper_ind = np.round(peak + 7* width).astype(int)

                beam_copy[lower_ind:upper_ind] = median

            peaks2 = scipy.signal.find_peaks(beam_copy)

            beam_copy2 = copy.deepcopy(beam_copy)
            difference = np.abs(beam_copy2 - np.nanmedian(beam_copy2))
            std = np.nanstd(beam_copy2)

            idxs = np.where(difference >= std)

            for idx in idxs[0]:
                beam_copy2[idx-5:idx+5] = np.nan

            nanidxs = np.where(np.isnan(beam_copy2))
            for nanidx in nanidxs[0]:
                beam_slice = beam_copy2[nanidx-20:nanidx+20]
                median = np.nanmedian(beam_slice)
                beam_copy2[nanidx] = median
    
            masked_beams.append(beam_copy2)
    
        stack = np.vstack(masked_beams)
        averaged_beam = np.nanmean(stack, axis=0)
        
        ### CORRECTING ###
        ds_corrected = ds_masked[0:512] / averaged_beam[:, np.newaxis] 
        ds_calibrated = bf_to_jy(ds_corrected, 1)
        ts_calibrated = np.nanmean(ds_calibrated, axis=0)
        peak_idx = np.nanargmax(ts_calibrated)

        offpulse_ts = np.nanmedian(ts_calibrated[peak_idx-50:peak_idx+50])
        ts_masked = ts_calibrated - offpulse_ts
        
        flux = np.nanmax(ts_masked) * 5 
        
        timeseries.append(ts_masked)
        fluxes.append(flux)
        
        
    #Plotting center HA used vs. Flux
    plt.figure()
    plt.scatter(has_list[center_has], fluxes)
    plt.scatter(has_list[center_has[6]], fluxes[6], color='r', label='HA of observation=-79.6')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Center HA used', fontsize=15)
    plt.ylabel('Flux Density (kJy)', fontsize=15)
    #plt.savefig("/arc/projects/chime_frb/mseth/plots/fixed_axes_labels/6_center_ha_flux.pdf", bbox_inches='tight')
    
    ## CALCULATING STUFF 
    actual_flux = fluxes[6]
    
    # Systematic error
    uppersys_error = np.abs(np.max(fluxes) - actual_flux)
    lowersys_error = np.abs(np.min(fluxes) - actual_flux)
    # Random error 
    actual_ts = timeseries[6]
    noise_ts = actual_ts[peak_idx-1000:peak_idx+1000]
    random_error = np.nanstd(noise_ts) * 5
    # Combined error
    upper_error = np.sqrt(np.square(uppersys_error) + np.square(random_error))
    lower_error = np.sqrt(np.square(lowersys_error) + np.square(random_error))
            
    # Fluence 
    ind_max = peak_idx+2*cascade_obj.best_width
    ind_min = peak_idx-2*cascade_obj.best_width
    
    integral = np.trapz(actual_ts[ind_min:ind_max])
    factor =  0.9830400000000001 / 1000  * 5 # Jy-s
    
    fluence = integral * factor #Jy-s
    
    uppersys_fluence = np.trapz(actual_ts[ind_min:ind_max] + uppersys_error) * factor
    lowersys_fluence = np.trapz(actual_ts[ind_min:ind_max] - lowersys_error) * factor
    upperran_fluence = np.trapz(actual_ts[ind_min:ind_max] + random_error/5) * factor
    lowerran_fluence = np.trapz(actual_ts[ind_min:ind_max] - random_error/5) * factor
    
    upper_flu_error = np.sqrt(np.square(uppersys_fluence - fluence) + np.square(upperran_fluence - fluence))
    lower_flu_error = np.sqrt(np.square(fluence - lowersys_fluence) + np.square(fluence - lowerran_fluence))
    
    #Luminosity
    peak_luminosity = flux_to_luminosity(actual_flux)
    
    uppersys_luminosity = flux_to_luminosity(actual_flux + uppersys_error)
    lowersys_luminosity = flux_to_luminosity(actual_flux - lowersys_error)
    
    upperran_luminosity = flux_to_luminosity(actual_flux + random_error)
    lowerran_luminosity = flux_to_luminosity(actual_flux - random_error)
    
    upper_lum_error = np.sqrt(np.square(uppersys_luminosity - peak_luminosity) + np.square(upperran_luminosity - peak_luminosity)
    lower_lum_error = np.sqrt(np.square(peak_luminosity - lowersys_luminosity) + np.square(peak_luminosity - lowerran_luminosity))
    
    pdb.set_trace()
    
    ## SAVING 
    fluences.append(fluence)
    scaled_fluxes.append(actual_flux)
    peak_luminosities.append(peak_luminosity)
    
    flux_errors.append([lower_error, upper_error])
    fluence_errors.append([lower_flu_error, upper_flu_error])
    lum_errors.append([lower_lum_error, upper_lum_error])
    
    event_timestamps.append(event_timestamp)
    has.append(ha)
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
    del timeseries
    del fluxes
    continue 
    

np.savez("/arc/projects/chime_frb/mseth/error_rfi_corrected_calibration.npz", mjds=mjds, has=has, event_timestamps=event_timestamps, peak_idxs=peak_idxs, scaled_fluxes=scaled_fluxes, peak_luminosity=peak_luminosities, fluence=fluences, flux_errors=flux_errors, fluence_errors=fluence_errors, lum_errors=lum_errors)