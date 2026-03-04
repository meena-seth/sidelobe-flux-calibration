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




def load_cascade_any(path: str):
    # Allow loading for pkl files
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            cascade_data = pickle.load(f)
        return cascade_data
    else:
        return cascade.load_cascade_from_file(path)

def flux_to_luminosity(peak_flux):
    result = 4 * np.pi * np.square(6.171 * 10**19) * peak_flux * 10**(-19)
    return result 


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
    print(f"Event time: {event_time} MJD: {event_time_mjd}")
    print(f"Source coord: {coord.ra.deg}, {coord.dec.deg}")

    coord = coord.transform_to(FK5(equinox=Time(event_time)))
    print(f"Precessed coord: {coord.ra.deg}, {coord.dec.deg}")
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
    print(f"Input fraction: {input_fraction}")

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
        print("Using second transit HA")
        print(
            f"First transit HA: {ha_deg}, Second transit HA: {ha_deg_second_transit}"
        )
        ha_deg = ha_deg_second_transit
        # set lower limit to true
        cascade_data.flux_lower_limit = True
        cascade_data.second_transit = True
        
    print(f"LST: {lst}, HA: {ha_deg}")
    # find out which transit it's on

    # store extra metadata
    cascade_data.ha_deg = ha_deg
    return cascade_data