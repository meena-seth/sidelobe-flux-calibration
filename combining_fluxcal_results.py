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
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.coordinates import FK5, ICRS
import my_utils
from my_utils import load_cascade_any, flux_to_luminosity

sys.path.append("/home/mseth2/scratch/frb_intensity_analysis")
import utils


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
gains_folder = '/home/mseth2/scratch/gains_hdf5_files'

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

def get_peak_flux(cascade_data):
    try:
            peak_flux = np.nanmax(
                np.nanmean(
                    cascade_data.beams[cascade_data._max_beam_idx].intensity,
                    axis=0,
                )
            )
            cascade_data.peak_flux = float(peak_flux)
    except Exception:
        cascade_data.peak_flux = np.nan
    return cascade_data

def get_HA(cascade_data):
    '''
    Just from Adam's fluxcal script 03/03. Quick fix.
    '''
    source_ra = float(83.6330565)
    source_dec = float(22.0144980)
    coord = SkyCoord(source_ra, source_dec, unit="deg")
   
    event_time, event_time_mjd, width = utils.get_cascade_time(cascade_data)
    cascade.event_time = event_time
    cascade.event_time_mjd = event_time_mjd

    if event_time is None:
        # try to get it from the l2_header
        event_time = cascade_data.event_time

    # precess coord to epoch of observation
    #print(f"Event time: {event_time} MJD: {event_time_mjd}")
    #print(f"Source coord: {coord.ra.deg}, {coord.dec.deg}")

    coord = coord.transform_to(FK5(equinox=Time(event_time)))
    #print(f"Precessed coord: {coord.ra.deg}, {coord.dec.deg}")
    # work out the ha of the observation
    # convert event time to lst

    location = EarthLocation.of_site("chime")
    # Get the datetime from the event
    # gain coverter changed on 2020-04-23, after this date just set input fraction to 1
    # This change is made by Kiyo, the gains already have the fgood factored in after this date
    if event_time_mjd > 58962:
        input_fraction = 1.0
    else:
        input_fraction = utils.return_good_inputs(gains_folder, event_time)
    #print(f"Input fraction: {input_fraction}")

    event_time_astropy = Time(event_time, scale="utc", format="datetime")
    lst = event_time_astropy.sidereal_time("mean", longitude=location.lon)
    # convert lst to degrees
    deg_lst = lst.deg
    ha_deg = deg_lst - coord.ra.deg 
    if ha_deg > 180:
        ha_deg -= 360
    elif ha_deg < -180:
        ha_deg += 360
    cascade_data.second_transit = False
    # see if there are two transits
    if (coord.dec.deg > 41) & (ha_deg > 90):
        second_transit_ra = coord.ra.deg - 180
        if second_transit_ra < 0:
            # fix negatives
            second_transit_ra += 360
        ha_deg_second_transit = deg_lst - second_transit_ra
        if ha_deg_second_transit > 180:
            ha_deg_second_transit -= 360
        elif ha_deg_second_transit < -180:
            ha_deg_second_transit += 360
        #print("Using second transit HA")
        #print(
            #f"First transit HA: {ha_deg}, Second transit HA: {ha_deg_second_transit}"
        #)
        ha_deg = ha_deg_second_transit
        # set lower limit to true
        cascade_data.flux_lower_limit = True
        cascade_data.second_transit = True

    #print(f"LST: {lst}, HA: {ha_deg}")
    # find out which transit it's on

    # store extra metadata
    cascade_data.ha_deg = ha_deg
    return cascade_data

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
            if f.endswith('_flux_calibrated.pkl'):
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

for i, file in enumerate(files, 1):
    try:
        print(f"\rProcessing {i}/{len(files)}", end="", flush=True)

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
            fluence = cascade_data.peak_flux * cascade_data.best_width * cascade_data.dt[0]  # in Jy s


        #### Save relevant data
        has.append(cascade_data.ha_deg)
        event_ids.append(cascade_data.eventid)
        event_times.append(cascade_data.event_time)
        fluxes.append(cascade_data.peak_flux)
        fluences.append(fluence)
        
        
    except Exception as e:
        print(f"Could not load {file} due to {e}")
        continue

print()

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


