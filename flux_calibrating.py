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
    result = 4 * np.pi * np.square(6.171 * 10**19) * peak_flux * 10**(-19)
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

event_timestamps = []
beam_ids = []
has = []
y_at_peak = []
mjds =[]
peak_idxs = []

for file in crab_norescaled_filepaths[26:27]:
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
    else:
        ha_idx = np.abs(has_list - ha).argmin()
        center_has = np.arange(ha_idx-12, ha_idx+13)

        beam_response = intensity_norm[0:512, ha_idx]

        beam_id = int(beam.beam_no)
        
        
    intensity_norm[np.log10(intensity_norm)>1] = np.nan
    
    #Removing RFI from holography
    
    fluxes = []
    for center_ha in center_has:        
        beam_response = intensity_norm[0:512, center_ha]
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
    
        ### CORRECTING ###
        ds_corrected = ds_masked[0:512] / beam_copy2[:, np.newaxis] 
        ds_calibrated = bf_to_jy(ds_corrected, 1)
    
        ts_calibrated = np.nanmean(ds_calibrated, axis=0)

        peak_idx = np.nanargmax(ts_calibrated)

        offpulse_ts = np.nanmedian(ts_calibrated[peak_idx-50:peak_idx+50])
        ts_masked = ts_calibrated - offpulse_ts
        
        flux = np.nanmax(ts_masked) * 5 /1000
        fluxes.append(flux)

    plt.figure()
    plt.scatter(has_list[center_has], fluxes)
    plt.scatter(has_list[center_has[12]], fluxes[12], color='r', label='HA of observation=-79.6')
    plt.xlabel('Center HA used')
    plt.ylabel('Flux Density (kJy)')
    plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/zoomed_noaveraging_centerHA_vs_flux")
    
    pdb.set_trace()

    f'''
    ### PLOTTING ###   
    #DS after masking, dedispersing, and calibrating. (Normalised & zoomed in)
    plt.figure()
    im = plt.imshow(normalise(ds_calibrated[:, peak_idx-100:peak_idx+100]), aspect='auto',cmap="YlGnBu")
    cbar = plt.colorbar(im)
    cbar.set_label("Flux (Jy)")
    plt.ylabel("Frequency Bins")
    plt.xlabel("Time sample")
    plt.title(f"DS centered on t={peak_idx}") 
    
    #Time series 
    plt.figure()
    plt.plot(ts_masked / 1000 *5)
    plt.ylabel("Flux (kJy)")
    plt.xlabel("Time sample")
    plt.title(f"Time series, all samples")
    
    #Time series (Zoomed in)
    plt.figure()
    plt.plot(ts_masked[peak_idx-100:peak_idx+100] / 1000 *5)
    plt.ylabel("Flux (kJy)")
    plt.xlabel("Time sample")
    plt.title(f"Time series, centered on t={peak_idx}")
    #plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/tests/23_ts")
            
    #Spectrum vs. holography response
    fig, ax = plt.subplot_mosaic(
    '''

    ''',
    constrained_layout = True,
    figsize = (10, 8), 
    sharex = True)
    
    ax['A'].plot(ds_calibrated[:, peak_idx] / np.nanmax(ds_calibrated[:, peak_idx], axis=0))
    ax['A'].set_ylabel('Normalised flux')
    
    ax['B'].plot(beam_response)
    ax['B'].plot(averaged_beam, color='r')
    ax['B'].set_yscale('log')
    ax['B'].set_ylabel('Normalised sensitivity')
    ax['B'].set_xlabel('Frequency bins')
    
    plt.suptitle(f"""Normalised spectrum & beam response
                     t={peak_idx}, HA={ha_idx}""")
    
    
    #(Peak frequency only) Time series
    plt.figure()
    plt.plot(ts_peak_freq / 1000 *5)
    plt.ylabel("Flux (kJy)")
    plt.xlabel("Time sample")
    plt.title(f"Time series, only at {freqs[peak_freq]} MHz")
    '''
        
    ## CALCULATING STUFF 
    ind_max = peak_idx+2*cascade_obj.best_width
    ind_min = peak_idx-2*cascade_obj.best_width
    integral = np.trapz(ts_masked[ind_min:ind_max])
    fluence = integral * 0.9830400000000001 /1000  * 5 #Jy-s
    
    flux = np.nanmax(ts_masked) * 5
    peak_luminosity = flux_to_luminosity(flux)
    
    #At peak frequency only 
    flux_peakfreq = np.nanmax(ts_peak_freq) * 5 
    luminosity_peakfreq = flux_to_luminosity(flux_peakfreq)
    holography_peakfreq = averaged_beam[peak_freq]
    
    ## SAVING 
    fluences.append(fluence)
    scaled_fluxes.append(flux)
    peak_luminosities.append(peak_luminosity)
    
    peakfreq_fluxes.append(flux_peakfreq)
    peakfreq_luminosities.append(luminosity_peakfreq)
    
    event_timestamps.append(event_timestamp)
    beam_ids.append(beam_id)
    has.append(ha)
    mjds.append(mjd)
    peak_idxs.append(peak_idx)
    
    plt.figure()
    plt.title(f"{i}_{mjd}")
    plt.text(x=0.5, y=0.5, ha='center', va='center', s=f"""
                  HA = {ha}
                  Flux = {flux:.2f}
                  Fluence = {fluence:.2f}
                  Luminosity = {peak_luminosity}
                  
                  Peak frequency: {freqs[peak_freq]:.2f} MHz, idx={peak_freq}
                  Flux = {flux_peakfreq:.2f}
                  Luminosity = {luminosity_peakfreq}
                  Holography value used = {holography_peakfreq}
                  """)
    plt.axis('off')
    #filename = f"/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/tests/{i}_{mjd}.pdf"
    #save_image(filename)
    plt.close('all')

    print(f"Spectra for {file} calibrated!")
    print(f'''
    Observation: {i}_{mjd}
    Flux = {flux}
    Fluence = {fluence}
    Luminosity = {peak_luminosity}
    ''')
    pdb.set_trace()
    
    del cascade_obj 
    del ds_corrected
    del ds_masked 
    del ds_before
    continue 
    
pdb.set_trace()
np.savez("rfi_corrected_calibration_1.npz", mjds=mjds, has=has, y_at_peak=y_at_peak, event_timestamps=event_timestamps, peak_idxs=peak_idxs, beam_ids=beam_ids, scaled_fluxes=scaled_fluxes, peak_luminosity=peak_luminosities, fluence=fluences, peakfreq_fluxes=peakfreq_fluxes, peakfreq_luminosities=peakfreq_luminosities)
