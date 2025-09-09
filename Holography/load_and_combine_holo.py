import beam_model as bm
from beam_model import config
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation
import sys
from beam_model import utils
from scipy.stats import iqr
from pandas import DataFrame as df

def get_primary_beam(freqs):
    #ra dec for cyg a
    source_ra = 299.8682
    source_dec = 40.7339

    chime = config.chime
    lat = chime.lat
    lon = chime.lon
    chime = EarthLocation.of_site("chime")
    #create an array of times across one day
    times = Time("2000-01-01T12:00:00") + np.linspace(0, 24, 10000) * u.hour #set to 2000 for epoch J2000

    #calculate the sidereal times at chime
    lst = times.sidereal_time("mean", longitude=chime)

    deg_lst = lst.deg
    ha_from_source = deg_lst - source_ra
    #position from equitorial
    pos = [utils.get_position_from_equatorial(source_ra, source_dec, t.datetime,0) for t in times]
    x = np.array([p[0] for p in pos])
    y = np.array([p[1] for p in pos])
    #location of the beam model
    bm_path = "/media/fengqiu/a4570436-56f7-4f14-9966-f143b3365018/holography_data/beam_model"
    XX = "beam_v1_SVD_XX.h5"
    YY = "beam_v1_SVD_YY.h5"
    my_config = {
        "datapath_SVD_xpol": os.path.join(bm_path, XX),
        "datapath_SVD_ypol": os.path.join(bm_path, YY),
    }
    primary_beam_model = bm.primary.DataDrivenSVDPrimaryBeamModel(my_config)
    sens = primary_beam_model.get_sensitivity(np.array([x, y]).T, freqs)
    sensitivity = sens["sensitivity"]
    return sensitivity, ha_from_source

from numba import njit, prange, boolean

@njit(parallel=True)
def mask_bad_freq_per_ha(intensity,thresh=2):
    mask_freq = np.zeros((intensity.shape[0],5), dtype=boolean)
    # mask_freq = np.zeros((intensity.shape[0],5), dtype=bool)
    for i in prange(2):
        #do a moving window median filter to smooth out the freq axis
        window_size = 300+(i*100)
        for j in range(0, len(intensity)-window_size):
            window = np.log10(intensity[j:j+window_size])
            if len(window)==np.sum(np.isnan(window)):
                continue
            iqr_val_freq = np.quantile(window[~np.isnan(window)], 0.75) - np.quantile(window[~np.isnan(window)], 0.25)
            med = np.nanmedian(window)
            # iqr_val_freq = iqr(window[~np.isnan(window)])
            mask_freq[j:j+window_size,i] |= np.abs(window-med) > (thresh * iqr_val_freq)
    return mask_freq

def mask_bad_freq(intensity,has,freqs,thresh=3):
    #cycle through all ha and mask
    mask = np.zeros(intensity.shape, dtype=bool)
    intensity[intensity==0] = np.nan
    # intensity_norm = intensity / np.nanmax(intensity, axis=0)[np.newaxis, :]
    for i in range(intensity.shape[1]):
        my_mask = mask_bad_freq_per_ha(intensity[:,i], thresh=thresh)
        mask[:,i] = np.any(my_mask, axis=1)

    intensity[mask] = np.nan
    return intensity, mask

def get_holo_data(holo_fn):
    holo_data = np.load(holo_fn, allow_pickle=True)
    freqs = holo_data['freqs']  #[1024]
    has = holo_data['HA']
    intensity_holo = holo_data['intensity_norm']  #[1024, 2160]
    #collapse this across ha
    intensity_holo, mask = mask_bad_freq(intensity_holo, has, freqs, thresh=2)
    intensity_holo = df(intensity_holo).interpolate(method='linear',limit_direction='both').to_numpy()
    return intensity_holo, freqs, has

if __name__ == "__main__":
    #load in the holography data
    holo_fns = sys.argv[1:]
    #plot overall, -5, 0 , 5 degree ha slices
    fig,ax = plt.subplots(4,1, figsize=(10,15))
    average_holo_arr = []
    for i, holo_fn in enumerate(holo_fns):
        print(f"Loading {holo_fn}")
        intensity_holo, freqs, has = get_holo_data(holo_fn)
        ax[0].plot(has, np.nanmean(intensity_holo, axis=0), label=f"{holo_fn}")
        target_has = [-5, 0, 5]
        for j, th in enumerate(target_has):
            closest_ha_idx = np.argmin(np.abs(has - th))
            ax[j+1].plot(freqs, intensity_holo[:, closest_ha_idx], label=f"{holo_fn} HA={th} deg")

        average_holo_arr.append(intensity_holo)
    average_holo = np.nanmean(np.array(average_holo_arr), axis=0)
    ax[0].plot(has, np.nanmean(average_holo, axis=0), label="Average Holography", color='k', linewidth=2)
    ax[0].set_xlabel('Hour Angle (deg)')
    ax[0].set_ylabel('Normalized Intensity')
    ax[0].set_yscale('log')
    ax[0].legend()
    for j, th in enumerate(target_has):
        closest_ha_idx = np.argmin(np.abs(has - th))
        ax[j+1].plot(freqs, average_holo[:, closest_ha_idx], label=f"Average Holography HA={th} deg", color='k', linewidth=2)
        ax[j+1].set_xlabel('Frequency (MHz)')
        ax[j+1].set_ylabel('Normalized Intensity')
        # ax[j+1].legend()
        ax[j+1].set_title(f"HA = {th} deg")
        ax[j+1].set_yscale('log')
    plt.tight_layout()
    plt.savefig("combined_holography.png")
    plt.figure()
    plt.imshow(np.log10(average_holo), aspect='auto', extent=[has[0], has[-1], freqs[0], freqs[-1]], origin='lower')
    plt.colorbar(label='Log10 Normalized Intensity')
    plt.xlabel('Hour Angle (deg)')
    plt.ylabel('Frequency (MHz)')
    plt.savefig("combined_holography_heatmap.png")
    np.savez("combined_holography.npz", intensity_norm=average_holo, freqs=freqs, HA=has)
