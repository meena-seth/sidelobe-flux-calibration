import numpy as np
from matplotlib import pyplot as plt
from iautils import cascade
import matplotlib
# matplotlib.use('TkAgg')
import copy
from numba import njit, prange, boolean
import os
import pickle
from astropy.time import Time
def normalise(spectrum):
    spectrum -=np.nanmedian(spectrum,axis=1)[:,np.newaxis]
    spectrum /=np.nanstd(spectrum,axis=1)[:,np.newaxis]
    return spectrum


def bf_to_jy(bf_spectrum, f_good):
    factor = (np.square(1024) * 128) / (np.square(4) * 0.806745 * 400)
    result = bf_spectrum / ( factor * np.square(f_good) )
    return result

def bf_holo_correction(cascade_file, holo_has, holo_freqs, holo_spectrum, ha):
    #interpolate the holo_spectrum to the bf_spectrum freqs
    cascade_freqs = cascade_file.beams[0].fbottom + np.arange(cascade_file.beams[0].nchan) * cascade_file.beams[0].df #MHz
    #make sure holo freqs are increasing
    #normalise holo spectrum to has 0
    holo_spectrum /= holo_spectrum[:,np.argmin(np.abs(holo_has))][:,np.newaxis]
    arg_sort = np.argsort(holo_freqs)
    holo_freqs = holo_freqs[arg_sort]
    holo_spectrum = holo_spectrum[arg_sort,:]
    #find the closest ha in the holo data
    holo_idx = np.argmin(np.abs(holo_has - ha))
    holo_correction = holo_spectrum[:,holo_idx]
    holo_correction_interp = np.interp(cascade_freqs, holo_freqs, holo_correction)
    cascade_file.beams[0].intensity /= holo_correction_interp[:,np.newaxis]
    cascade_file.beams[0].holo_correction = holo_correction_interp
    return cascade_file


# @njit(parallel=True)
def mask_bad_freq_nowindow(intensity,thresh=2):
    # mask_freq = np.zeros((intensity.shape[0]), dtype=boolean)
    mask_freq = np.zeros((intensity.shape[0]), dtype=bool)
    #do a moving window median filter to smooth out the freq axis
    window=intensity
    iqr_val_freq = np.quantile(window[~np.isnan(window)], 0.75) - np.quantile(window[~np.isnan(window)], 0.25)
    med = np.nanmedian(window)
    mask_freq |= np.abs(window-med) > (thresh * iqr_val_freq)
    return mask_freq

def mask_bad_freq(intensity,thresh=2):
    #cycle through all ha and mask
    mask = np.zeros(intensity.shape, dtype=bool)
    intensity[intensity==0] = np.nan
    # intensity_norm = intensity / np.nanmax(intensity, axis=0)[np.newaxis, :]
    for i in range(intensity.shape[1]):
        print(f"Masking time {i}/{intensity.shape[1]}")
        my_mask = mask_bad_freq_nowindow(intensity[:,i], thresh=thresh)
        mask[:,i] = my_mask

    intensity[mask] = np.nan
    return intensity, mask

def get_cascade_time(cascade):
    from datetime import datetime, timedelta
    import pytz
    cascade_data = np.load(cascade,allow_pickle=1)
    metadata = cascade_data['metadata'].tolist()
    frame0_ctime = metadata['frame0_ctime']
    frame_0_dt = datetime.fromtimestamp(frame0_ctime, pytz.utc)

    fpga0_seconds = cascade_data['fpga0s'][0] * 2.56e-6
    start_time = frame_0_dt + timedelta(seconds=fpga0_seconds[0])
    time_bins = metadata['dt'][0]
    start_time_mjd = Time(start_time).mjd

    bins_from_start_time = metadata['peak_position']
    width = metadata['width']*time_bins
    seconds_from_start_time = time_bins * bins_from_start_time

    event_time_mjd = start_time_mjd + seconds_from_start_time/86400
    event_time = start_time + timedelta(seconds = seconds_from_start_time)
    if event_time.year < 2000:
        #instead of returning a None, lets query the L2 header for the time
        print(f"Invalid event time: {cascade} - {event_time}")
        return None, None, None
    print(event_time,metadata['event_time'])
    return event_time, event_time_mjd,width


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux Calibration Script")
    parser.add_argument('intensity_files', nargs='+', type=str, help='Path to the intensity data file(s)')
    parser.add_argument('--holo_file', type=str, required=True, help='Path to the holography data file')
    parser.add_argument('-ra', type=str, required=True, help='RA of the source in HH:MM:SS or decimal degrees')
    parser.add_argument('-dec', type=str, required=True, help='Dec of the source in DD:MM:SS or decimal degrees')
    parser.add_argument('-f', '--force', action='store_true', help='Force recalibration even if output file exists')
    parser.add_argument('-l2_header', type=str, default=None, help='Path to the L2 header file (if needed to get time)')
    args = parser.parse_args()
    # Load in beam response
    holography_data = np.load(args.holo_file)
    intenisty_files = args.intensity_files
    for no_rescale_cascade in intenisty_files:
        try:
            cascade_data = cascade.load_cascade_from_file(no_rescale_cascade)
        except Exception as e:
            print(f"Error loading cascade file {no_rescale_cascade}: {e}, skipping")
            sys.exit(1)
        flux_calibrated_cascade = no_rescale_cascade.replace('.npz','_flux_calibrated.pkl')
        if os.path.exists(flux_calibrated_cascade) and not args.force:
            #try to load it
            try:
                with open(flux_calibrated_cascade, 'rb') as f:
                    cascade_data = pickle.load(f)
                print(f"Flux calibrated file already exists and is loadable: {flux_calibrated_cascade}, skipping")
                continue
            except:
                print(f"Flux calibrated file already exists but is not loadable: {flux_calibrated_cascade}, recalibrating")
        from astropy.coordinates import SkyCoord
        try:
            source_ra = float(args.ra)
            source_dec = float(args.dec)
            coord = SkyCoord(source_ra, source_dec, unit='deg')
        except:
            coord = SkyCoord(args.ra, args.dec, unit=('hourangle', 'deg'))
            source_ra = coord.ra.deg
            source_dec = coord.dec.deg


        #now fill in the nans with the median values
        cascade_data.beams[0].intensity, mask = mask_bad_freq(cascade_data.beams[0].intensity)
        cascade_data.beams[0].intensity[np.isnan(cascade_data.beams[0].intensity)] = np.nanmedian(cascade_data.beams[0].intensity)
        #plot the timeseries

        intensity_norm = holography_data['intensity_norm']
        freqs = holography_data['freqs']
        has = holography_data['has']
        event_time, event_time_mjd, width = get_cascade_time(no_rescale_cascade)

        if event_time is None:
            #try to get it from the l2_header
            l2_header_file = args.l2_header
            data = np.load(l2_header_file, allow_pickle=True)
            event_numbers = data[0]
            events = data[1]
            event_number = no_rescale_cascade.split('/')[-1].split('_')[1]
            event_idx = np.where(event_numbers == int(event_number))[0]
            my_event = events[event_idx[0]]
            event_time = my_event.timestamp_utc


        #precess coord to epoch of observation
        print(f"Event time: {event_time} MJD: {event_time_mjd}")
        print(f"Source coord: {coord.ra.deg}, {coord.dec.deg}")
        from astropy.coordinates import FK5, ICRS
        coord = coord.transform_to(FK5(equinox=Time(event_time)))
        print(f"Precessed coord: {coord.ra.deg}, {coord.dec.deg}")
        #work out the ha of the observation
        #convert event time to lst
        from astropy.coordinates import EarthLocation
        location = EarthLocation.of_site('chime')
        event_time_astropy = Time(event_time, scale='utc', format='datetime')
        lst = event_time_astropy.sidereal_time('mean', longitude=location.lon)
        #convert lst to degrees
        deg_lst = lst.deg
        ha_deg = coord.ra.deg - deg_lst
        if np.abs(ha_deg) > 180:
            print("HA out of range, skipping")
            continue
        print(f"LST: {lst}, HA: {ha_deg}")
        cascade_data = bf_holo_correction(cascade_data, has, freqs, intensity_norm, ha_deg)
        cascade_data.beams[0].intensity = bf_to_jy(cascade_data.beams[0].intensity, 1)
        #dump the calibrated spectrum
        with open(flux_calibrated_cascade, 'wb') as f:
            #NOTE: we are not correcting for the geometric factor here!!
            pickle.dump(cascade_data, f)

        #subtract median from each channel
        spectra = np.nanmean(cascade_data.beams[0].intensity, axis=1)
        holo = cascade_data.beams[0].holo_correction
        #cut out the very noisy channels (i.e. very insensitive channels)
        spectra_median = np.nanmedian(spectra)
        spectra_iqr = np.quantile(spectra, 0.75) - np.quantile(spectra, 0.25)
        bad_channels = np.abs(spectra-spectra_median) > 3 * spectra_iqr
        #also just lob off the upper quarter of the band
        bad_channels |= np.arange(len(spectra)) > (0.75 * len(spectra))
        spectra[bad_channels] = np.nan

        cascade_data.beams[0].intensity[bad_channels,:] = np.nan
        #first lets make sure the data is actually there
        cascade_copy = copy.deepcopy(cascade_data)
        cascade_copy.beams[0].intensity[bad_channels,:] = np.nan
        cascade_copy.beams[0].intensity = normalise(cascade_copy.beams[0].intensity)
        #downsample the cascade_copy
        cascade_copy.process_cascade(dm=cascade_copy.dm[0], nsub=1024, dedisperse=False, downsample=1)
        cascade_data.beams[0].intensity -= np.nanmean(cascade_data.beams[0].intensity, axis=1)[:,np.newaxis]


        ts = np.nanmean(cascade_data.beams[0].intensity, axis=0)
        #collapse intensity_norm to a ha series
        holo_ha = np.nanmean(intensity_norm, axis=0)
        #subtract the median
        fig, ax = plt.subplots(2,2, figsize=(10,8))
        ax[0,0].imshow(cascade_copy.beams[0].intensity, aspect='auto', origin='lower',cmap='Greys', extent=[0, cascade_copy.beams[0].intensity.shape[1]*cascade_copy.beams[0].dt, cascade_copy.beams[0].fbottom, cascade_copy.beams[0].fbottom + cascade_copy.beams[0].nchan*cascade_copy.beams[0].df])
        ax[0,0].set_xlabel('Time (s)')
        ax[0,0].set_ylabel('Frequency (MHz)')
        # ts -= np.nanmedian(ts)
        ax[1,0].plot(np.arange(len(ts))*cascade_copy.beams[0].dt, ts*5)
        ax[1,0].set_xlabel('Time (bins)')
        ax[1,0].set_ylabel('Intensity (Jy)')
        ax[1,1].plot(np.arange(len(spectra))*cascade_copy.beams[0].df + cascade_copy.beams[0].fbottom, np.log10(spectra)-np.nanmedian(np.log10(spectra)), alpha=0.5, label='Calibrated')
        ax[1,1].plot(np.arange(len(spectra))*cascade_copy.beams[0].df + cascade_copy.beams[0].fbottom, np.log10(holo)-np.nanmedian(np.log10(holo)), alpha=0.5, label=f'Holography median = {np.nanmedian(holo):.2f}')
        ax[1,1].set_xlabel('Frequency (MHz)')
        ax[0,1].plot(has, holo_ha)
        ax[0,1].axvline(ha_deg, color='r', linestyle='--', label='Event HA')
        ax[0,1].set_yscale('log')
        ax[0,1].set_xlabel('Hour Angle (deg)')
        ax[0,1].set_ylabel('Normalized Intensity')

        plt.tight_layout()
        plt.savefig(no_rescale_cascade.replace('.npz','_calibrated.png'))
        plt.close()
