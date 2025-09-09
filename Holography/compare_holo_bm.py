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
#hardcode crab ra dec for now
source_ra = 83.633083
source_dec = 22.0145
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
ha_from_crab = deg_lst - source_ra
#position from equitorial
pos = [utils.get_position_from_equatorial(source_ra, source_dec, t.datetime,0) for t in times]
x = np.array([p[0] for p in pos])
y = np.array([p[1] for p in pos])

#load in the holography data
holo_fn = sys.argv[1]
holo_data = np.load(holo_fn, allow_pickle=True)
freqs = holo_data['freqs']  #[1024]
has = holo_data['HA']
intensity_holo = holo_data['intensity_norm']  #[1024, 2160]
#divide everything by ha = 0
intensity_holo = intensity_holo / intensity_holo[:, np.argmin(np.abs(has))][:, np.newaxis]
#collapse this across ha
from scipy.stats import iqr
def mask_bad_freq(intensity,has,freqs,thresh=2):
    #collapse across ha
    #cut out the middle 20 degrees in has
    #normalise to ha0
    intensity = intensity / intensity[:, np.argmin(np.abs(has))][:, np.newaxis]
    intensity_cut = intensity[:, (has < -2) | (has > 2)]

    #set the 0s to nans
    freq_mean = np.nanmean(intensity_cut, axis=1)
    freq_mean[freq_mean == 0] = np.nan
    z_mask = np.isnan(freq_mean)
    median = np.nanmedian(freq_mean)
    iqr_val = iqr(freq_mean[~np.isnan(freq_mean)])
    mask = np.abs(freq_mean - median) > thresh * iqr_val
    mask = mask | z_mask
    intensity[mask,:] = np.nan

    plt.figure()
    plt.plot(freqs, np.nanmean(intensity_cut, axis=1), label='Mean Intensity')
    plt.axhline(median, color='r', linestyle='--', label='Median')
    plt.axhline(median + thresh * iqr_val, color='g', linestyle='--', label='Upper Threshold')
    plt.axhline(median - thresh * iqr_val, color='g', linestyle='--', label='Lower Threshold')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Mean Intensity')
    plt.legend()
    return intensity
intensity_holo = mask_bad_freq(intensity_holo, has, freqs, thresh=2)
#cast intensity holo as a pandas dataframe
from pandas import DataFrame as df
plt.figure(figsize=(10,6))
plt.imshow(np.log10(intensity_holo), aspect='auto', extent=[has[0], has[-1], freqs[0], freqs[-1]], origin='lower')
intensity_holo = df(intensity_holo).interpolate(method='linear').to_numpy()
np.savez("interpolated_holo.npz", intensity_norm=intensity_holo, freqs=freqs, has=has)

#clip the intensity holo to only between -20 and 20 degrees
# intensity_holo = intensity_holo[:, (has >= -20) & (has <= 20)]
# has = has[(has >= -20) & (has <= 20)]
# intensity_holo[np.isnan(intensity_holo)] = np.nanmedian(intensity_holo)
plt.figure(figsize=(10,6))
plt.imshow(np.log10(intensity_holo), aspect='auto', extent=[has[0], has[-1], freqs[0], freqs[-1]], origin='lower')
plt.colorbar(label='Log10 Normalized Intensity')
plt.xlabel('Hour Angle (deg)')
plt.ylabel('Frequency (MHz)')




#location of the beam model
bm_path = "/media/fengqiu/a4570436-56f7-4f14-9966-f143b3365018/holography_data/beam_model"
XX = "beam_v1_SVD_XX.h5"
YY = "beam_v1_SVD_YY.h5"
config = {
    "datapath_SVD_xpol": os.path.join(bm_path, XX),
    "datapath_SVD_ypol": os.path.join(bm_path, YY),
}
primary_beam_model = bm.primary.DataDrivenSVDPrimaryBeamModel(config)
sens = primary_beam_model.get_sensitivity(np.array([x, y]).T, freqs)
sensitivity = sens["sensitivity"]
my_model_sensitivity = np.nanmedian(sensitivity, axis=1)
#find the non nans
nan_mask = ~np.isnan(my_model_sensitivity)
my_model_sensitivity = my_model_sensitivity[nan_mask]
ha_from_crab = ha_from_crab[nan_mask]
sensitivity = sensitivity[nan_mask,:]
plt.figure(figsize=(10,6))
plt.plot(ha_from_crab, np.log10(my_model_sensitivity/np.max(my_model_sensitivity)), label="Beam Model", alpha=0.7)
av_intensity_holo = np.nanmedian(intensity_holo[-400:,:], axis=0)
av_intensity_holo = av_intensity_holo / np.max(av_intensity_holo)
plt.plot(has, np.log10(av_intensity_holo), label="Holography", alpha=0.7)
plt.xlabel("Hour Angle (deg)")
plt.ylabel("Log10 Normalized Sensitivity")
plt.legend()

#plot slices of the beam model vs holography at different frequencies
target_has = [-10, -5, 0, 5, 10]
fig, axs = plt.subplots(len(target_has), 1, figsize=(10, 3*len(target_has)), sharex=True)
for i, th in enumerate(target_has):
    closest_ha_idx = np.argmin(np.abs(has - th))
    closest_ha = has[closest_ha_idx]
    closest_ha_model_idx = np.argmin(np.abs(ha_from_crab - closest_ha))
    axs[i].plot(freqs, np.log10(intensity_holo[:, closest_ha_idx]), label="Holography", alpha=0.7)
    axs[i].plot(freqs, np.log10(sensitivity[closest_ha_model_idx,:]), label="Beam Model", alpha=0.7)
    axs[i].set_title(f"Hour Angle = {closest_ha:.2f} deg")
    axs[i].legend()

axs[-1].set_xlabel("Frequency (MHz)")
plt.show()
