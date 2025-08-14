import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pdb
import copy
import datetime
from iautils import cascade

from datetime import datetime
from astropy.coordinates import SkyCoord

sys.path.insert(0, os.path.abspath('beam-model'))
from beam_model import utils, formed
from beam_model import config

'''
Flux calibrates sidelobe intensity data with holography. 

Takes: 
    Holography intensity data (.npz file)
    List of filepaths to cascades we want to flux calibrate 
Gives:
    Plots of calibrated timeseries & dynamic spectra 
    .npz file with MJDs, HAs, peak flux, peak luminosity, and fluence for each pulse. 
'''

#### Load in files to observations ####

files = np.load("/arc/projects/chime_frb/mseth/nrao/flux_calibration/New_Crab_norescaled_filepaths.npz")
crab_norescaled_filepaths = files['filepath']

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

max_tidxs = []
max_timestamps = []
beam_ids = []
timeseries = []


has = []
y_at_peak = []
scaled_fluxes = []
mjds =[]
fluences = []
peak_luminosities = []


for file in crab_norescaled_filepaths[0:2]:
    cascade_obj = cascade.load_cascade_from_file(file)
    cascade_obj.dm = 56.7    #Dedisperse 
    
    peaks = []
    
    ## OBSERVATIONS ##
    for beam in cascade_obj.beams:   #Find index of max beam 
        beam.subband(1024,56.7,apply_weights=False)  #Downsample to 1024 frequency
        mean_ts = np.nanmean(beam.intensity, axis=0)
        mean_ts_masked = mean_ts[400:500]
        offpeak_mean = np.nanmean(mean_ts_masked)
    
        subtracted_ts = mean_ts - offpeak_mean
    
        peak = np.max(subtracted_ts)
        peaks.append(peak)
        del mean_ts
    
    max_beam_idx = np.argmax(peaks)
    max_beam = cascade_obj.beams[max_beam_idx]
    ts = np.nansum(max_beam.intensity, axis=0)

    # Load in parameters for later 
    max_tidx = np.argmax(ts)
    max_timestamp = cascade_obj.event_time 
    beam_id = int(beam.beam_no)
    ha, y = utils.get_position_from_equatorial(source_ra, source_dec, max_timestamp)
    
    # Get dynamic spectrum & subtract off-pulse mean
    dyn_spectrum = max_beam.intensity
    offpulse_spectrum = dyn_spectrum[:, 0:100]
    offpulse_mean = np.nanmean(offpulse_spectrum, axis=1)

    dynamic_spectrum = dyn_spectrum - offpulse_mean[:, np.newaxis]
    #dynamic_spectrum = dynamic_spectrum[0:512, :]
    #spectrum = dynamic_spectrum[0:512, max_tidx] # [1024,] at time of Tau A's peak
    
    filename = file.split("/")
    mjd = filename[7].split("_")[1].split(".")[0]
    filepath_list = crab_norescaled_filepaths.tolist()
    i = filepath_list.index(file)
    
    spectrum_copy = copy.deepcopy(dynamic_spectrum)
    spectrum_copy = normalise(spectrum_copy)
    spectrum_copy[np.isnan(spectrum_copy)]=np.nanmedian(spectrum_copy)
    ind_max = cascade_obj.peak_position+5*cascade_obj.best_width
    ind_min = cascade_obj.peak_position-5*cascade_obj.best_width
    spectrum_copy = spectrum_copy[:,ind_min:ind_max]
    
    plt.figure()
    plt.pcolormesh( np.arange(10*cascade_obj.best_width),freqs, spectrum_copy)
    plt.ylabel("Frequency bins")
    plt.xlabel("Time samples")
    plt.savefig(f"a_{i}_{mjd}_ds.png")
    plt.close()
    

    ## PRIMARY BEAM ##
    
    ha_idx = np.abs(has_list - ha).argmin()
    beam_response = intensity_norm[:, ha_idx] #[1024,] at HA of Tau A's peak
    beam_response[beam_response==0] = np.nan
        
    ## CORRECTING ##
    spectrum_corrected = dynamic_spectrum[0:178] / beam_response[0:178, np.newaxis]
    
    plt.figure()
    plt.imshow(spectrum_corrected, aspect="auto")
    plt.savefig(f"a_{i}_{mjd}_sc.png")
    
    spectrum_calibrated = bf_to_jy(spectrum_corrected, 1)
    ts_calibrated = np.nanmean(spectrum_calibrated, axis=0)
    
    #Plot calibrated ts
    plt.figure()
    plt.plot(ts_calibrated/1000 * 5)
    plt.ylabel("Flux (kJy)")
    plt.xlabel("Time sample")
    plt.title(f"{mjd}_{i}")
    plt.savefig(f"a_{i}_{mjd}_ts.png")
    
    ## CALCULATING STUFF ##
    integral = np.trapz(ts_calibrated[ind_min:ind_max])
    fluence = integral * 0.9830400000000001  #Fluence
        
    max_flux = np.max(ts_calibrated)
    scaled_flux = max_flux * 5   # Maximum flux density
    
    peak_luminosity = flux_to_luminosity(scaled_flux)

    ## SAVING ##
    fluences.append(fluence)
    scaled_fluxes.append(scaled_flux)
    peak_luminosities.append(peak_luminosity)
    has.append(ha)
    mjds.append(mjd)
    
    timeseries_calibrated = ts_calibrated.tolist()
    timeseries.append(timeseries_calibrated)
    
    print(f"Spectra for {file} calibrated!")
    print(f'''
    Observation: {i}_{mjd}
    Flux = {scaled_flux}
    Fluence = {fluence}
    Luminosity = {peak_luminosity}
    ''')
    
    
    del cascade_obj 
    del dynamic_spectrum 
    del beam_response 
    del spectrum_corrected
    continue 
    
#timeseries_all = np.vstack(timeseries)
#np.savez("Calibrated_timeseries.npz", ts=timeseries_all)

np.savez("Calibration_values_first.npz", mjds=mjds, has=has, scaled_fluxes=scaled_fluxes, peak_luminosity=peak_luminosities, fluence=fluences)
