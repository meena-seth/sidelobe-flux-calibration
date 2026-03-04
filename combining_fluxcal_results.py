import numpy as np 
import matplotlib.pyplot as plt
import pdb
from iautils import cascade
import pickle
import pdb
import copy
import os
import sys
from uncertainties import ufloat
from uncertainties import unumpy as unp
import re
from glob import glob

sys.path.append("/home/mseth2/scratch/sidelobe-flux-calibration")
from utils import load_cascade_any, flux_to_luminosity, get_peak_flux, get_HA


'''
After running frb_intensity_analysis/main_side_flux_cal.py, you get a directory full of .pkl files with flux calibrated cascade objects
This script will combine the results from those files into a single .npz file to make plotting results quicker.

Optionally: 
- Make a separate .npz file containing pulse widths (for Nimmo 2022 plot)
- Make a separate .txt file with event IDs 
- Calculate fluences 
- Calculate uncertainties for flux and propagate for luminosities 
- Only do this for files that have a corresponding image of the waterfall 
    (i.e. if you've gone thru and deleted images for RFI events)

Can be run with sbatch script 
'''

from_images = False
get_uncertainty=True
get_widths=True
get_eventids = True
get_fluences = True
calc_fluxha = True

path = '/project/rpp-chime/adamdong/rfi_filtered' #Path to directory with flux calibrated .pkl files
outdir = '/home/mseth2/scratch/02_23_fluxcal_results' #Path to save .npz file with combined results

has = []
event_ids = []
event_times = []
fluxes = []
rand_uncs = []
sys_uncs = []
total_uncs = []
lums = []
lum_uncs = []
widths = []
event_ids = []
fluences = []

if from_images:
    #### Make a list of filepaths to load data for. Either only do for those that also have a corresponding png, 
    ####  or just do for all .pkl files in directory.
    images = []
    files = []
    for (root, dirs, file) in os.walk(path):
        for f in file: 
            if f.endswith('.png'):
                images.append(os.path.join(root, f))
                files.append(os.path.join(root, f.replace("_all_beams.png", "_flux_calibrated.pkl")))
            else:
                continue
else: 
    files = []
    for (root, dirs, file) in os.walk(path):
        for f in file: 
            if f.endswith('.pkl'):
                files.append(os.path.join(root, f))
            else:
                continue


if get_eventids:
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.search(r'cascade_(\d+)_norescale', filename)
        
        if match:
            event_ids.append(match.group(1))

    # Save to txt
    with open("event_ids.txt", "w") as f:
        for eid in event_ids:
            f.write(eid + "\n")

    print(f"Saved {len(event_ids)} event IDs to event_ids.txt")


for file in files:
    try:
        print(f"Loading {file}....")
        cascade_data = load_cascade_any(file)

        try:
            ha = cascade_data.ha_deg
            flux = cascade_data.peak_flux
        except Exception as e:
            cascade_data = get_HA(cascade_data)
            cascade_data = get_peak_flux(cascade_data)

        if get_widths:
            #### Get pulse widths 
            width = cascade_data.best_width * cascade_data.dt[0]  # in ms
            if width==0:
                width = cascade_data.dt[0]
            widths.append(width)

        if get_uncertainty:
            ##### Get random noise for each beam 
            ts = np.nanmean(cascade_data.beams[cascade_data._max_beam_idx].intensity, axis=0)
            convolve_ts = np.convolve(ts, np.ones(5) / 5, mode="same")
            # cut out the peak +- 2*best width
            off_peak_noise = copy.deepcopy(convolve_ts)
            off_peak_noise[
                cascade_data.peak_position
                - 2 * cascade_data.best_width : cascade_data.peak_position
                + 2 * cascade_data.best_width
            ] = np.nan
            rand_unc = np.nanstd(off_peak_noise)

            #### Define systematic uncertainty (set to 40% for now)
            sys_unc = 0.4 * cascade_data.peak_flux

            #### Calculate total uncertainty 
            total_unc = np.sqrt(rand_unc**2 + sys_unc**2)

            #### Calculate luminosity 
            flux_with_unc = ufloat(cascade_data.peak_flux, total_unc)
            lum_with_unc = flux_to_luminosity(flux_with_unc)
            lum = unp.nominal_values(lum_with_unc)
            lum_err = unp.std_devs(lum_with_unc)

            rand_uncs.append(rand_unc)
            sys_uncs.append(sys_unc)
            total_uncs.append(total_unc)
            lums.append(lum)
            lum_uncs.append(lum_err)

        if get_fluences:
            fluence = cascade_data.peak_flux * cascade_data.best_width * cascade_data.dt[0]  # in Jy ms


        #### Save relevant data
        has.append(cascade_data.ha_deg)
        event_ids.append(cascade_data.eventid)
        event_times.append(cascade_data.event_time)
        fluxes.append(cascade_data.peak_flux)
        fluences.append(fluence)
        
        
    except Exception as e:
        print(f"Could not load {file} due to {e}")
        continue

if outdir is not None:
    if get_widths:
        np.savez(f"{outdir}/pulse_widths.npz",
            widths=np.array(widths))
    if get_uncertainty:
        np.savez(
            f"{outdir}/fluxcal_results.npz",
            has=np.array(has),
            event_ids=np.array(event_ids),
            event_times=np.array(event_times),
            fluxes=np.array(fluxes),
            rand_uncs=np.array(rand_uncs),
            sys_uncs=np.array(sys_uncs),
            total_uncs=np.array(total_uncs),
            lums=np.array(lums),
            lum_uncs=np.array(lum_uncs),
            fluences=np.array(fluences)
        )


