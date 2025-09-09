from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle

from ch_util import tools
import sys
from scipy.stats import iqr
import copy
def mask_bad_feed(data_slice,has,thresh=2,cyl=None):
    #collapse the dataslice across the HA axis
    #only get the data for |HA| > 10 degrees
    data_slice_cut = copy.deepcopy(data_slice[:,:,(np.abs(has) > 10)])
    data_slice_cut[data_slice_cut == 0] = np.nan
    #divide each data_slice_cut by the 0ha slice
    data_slice_cut = data_slice_cut / data_slice[:,:,np.argmin(np.abs(has))][:,:,np.newaxis]
    data_slice_abs = np.abs(data_slice_cut)**2
    #find the median and iqr of the mean_feed
    mask_feed = np.zeros(data_slice_abs.shape[1], dtype=bool)
    mask_freq = np.zeros(data_slice_abs.shape[0], dtype=bool)
    mask_has = np.zeros(data_slice_abs.shape[2], dtype=bool)

    #first do a quick mask over all three axies
    med = np.nanmedian(data_slice_abs)
    iqr_val = iqr(data_slice_abs[~np.isnan(data_slice_abs)])
    mask_3d = np.abs(data_slice_abs - med) > thresh * iqr_val
    print(f"Initial 3D Masked {np.sum(mask_3d)} points")
    data_slice_abs[mask_3d] = np.nan
    #iteratively mask feeds and freqs that are outliers
    freq_feed_median = np.nanmean(data_slice_abs,axis=2)
    for i in range(5):
        med_feed = np.nanmedian(freq_feed_median,axis=0)
        iqr_val_feed = iqr(med_feed[~np.isnan(med_feed)])
        mask_feed |= np.abs(med_feed - np.nanmedian(med_feed)) > thresh * iqr_val_feed
        print(f"Iteration {i}: Masked {np.sum(mask_feed)} feeds")
        data_slice_abs[:,mask_feed,:] = np.nan

        med_freq = np.nanmedian(freq_feed_median,axis=1)
        #do a moving window median filter to smooth out the freq axis
        window_size = 300+(i*100)
        for j in range(0, len(med_freq)-window_size):
            window = med_freq[j:j+window_size]
            iqr_val_freq = iqr(window[~np.isnan(window)])
            mask_freq[j:j+window_size] |= np.abs(med_freq[j:j+window_size] - np.nanmedian(window)) > thresh * iqr_val_freq
        print(f"Iteration {i}: Masked {np.sum(mask_freq)} freqs")
        data_slice_abs[mask_freq,:,:] = np.nan


    med_feed[mask_feed] = np.nan
    med_freq[mask_freq] = np.nan
    data_slice_abs[mask_freq,:,:] = np.nan
    data_slice_abs[:,mask_feed,:] = np.nan
    med_has = np.nanmedian(data_slice_abs,axis=(0,1))
    fig, ax = plt.subplots(3,1,figsize=(10,15))
    ax[0].plot(med_feed, label='Median Feed')
    ax[0].axhline(np.nanmedian(med_feed), color='r', linestyle='--', label='Median of Median Feed')
    ax[0].axhline(np.nanmedian(med_feed) + thresh * iqr_val_feed, color='g', linestyle='--', label='Upper Threshold')
    ax[0].axhline(np.nanmedian(med_feed) - thresh * iqr_val_feed, color='g', linestyle='--', label='Lower Threshold')
    ax[0].set_xlabel('Feed Index')
    ax[1].plot(med_freq, label='Median Frequency')
    ax[1].axhline(np.nanmedian(med_freq), color='r', linestyle='--', label='Median of Median Frequency')
    ax[1].axhline(np.nanmedian(med_freq) + thresh * iqr_val_freq, color='g', linestyle='--', label='Upper Threshold')
    ax[1].axhline(np.nanmedian(med_freq) - thresh * iqr_val_freq, color='g', linestyle='--', label='Lower Threshold')
    ax[1].set_xlabel('Frequency Index')
    ax[2].plot(med_has, label='Median HA')
    ax[2].set_xlabel('HA Index')
    plt.savefig(f"{cyl}_rfimasking_diagnostics.png")
    plt.close()
    #final mask
    print(f"final number of data points masked : {np.sum(np.isnan(data_slice))}")

    data_slice_cut = data_slice[:,:,(np.abs(has) > 10)]
    data_slice_cut[mask_3d] = np.nan
    data_slice[:,:,(np.abs(has) > 10)] = data_slice_cut

    print(f"final number of data points masked : {np.sum(np.isnan(data_slice))}")
    data_slice[mask_freq,:,:] = np.nan

    print(f"final number of data points masked : {np.sum(np.isnan(data_slice))}")
    data_slice[:,mask_feed,:] = np.nan

    print(f"final number of data points masked : {np.sum(np.isnan(data_slice))}")
    data_slice[data_slice == 0] = np.nan

    print(f"final number of data points masked : {np.sum(np.isnan(data_slice))}")
    return data_slice





def respond(data,plot=False):

    f = h5py.File(data, "r")

    beam_dset = f["beam"]
    weight_dset = f['weight']

    # Read axes names & make array for freqs and HAs
    axes_names = beam_dset.attrs["axis"]
    index_map = f["index_map"]

    has = index_map["pix"]["phi"][:]
    zha_idx = np.argmin(np.abs(has))
    freqs = index_map["freq"][:]

    # Slice out just the data for Cylinder D, y polarization
    ycyls = {
        "A": (0, 256),
        "B": (512, 768),
        "C": (1024, 1280),
        "D": (1536, 1792),
    }
    xcyls = {
        "A": (256, 512),
        "B": (768, 1024),
        "C": (1280, 1536),
        "D": (1792, 2048),
    }

    for slide in xcyls:
        my_slice = slice(ycyls[slide][0], ycyls[slide][1])
        cyl = beam_dset[:,0,my_slice,:]
        weight = weight_dset[:,0,my_slice,:]
        # cyl = cyl * weight  # Apply weights
        cyl_x = mask_bad_feed(cyl,has,cyl="x"+slide)
        # Make the slice I want
        cyl_x = np.abs(cyl_x * np.conj(cyl_x))  # Intensity
        # plt.figure()
        # plt.imshow(np.log10(cyl_x), aspect='auto', extent=[has[0], has[-1], 0, cyl_x.shape[0]])
        # plt.colorbar(label='Log10 Intensity')
        # plt.xlabel('HA (degrees)')
        # plt.ylabel('Feed Index')
        # plt.show()
        cyl_x_averaged = np.nanmean(cyl_x, axis=1)  # Average over feeds

    for slide in ycyls:
        my_slice = slice(ycyls[slide][0], ycyls[slide][1])
        cyl = beam_dset[:,0,my_slice,:]
        weight = weight_dset[:,0,my_slice,:]
        # cyl = cyl * weight
        cyl_y = mask_bad_feed(cyl,has,cyl="y"+slide)
        # Make the slice I want
        cyl_y = np.abs(cyl_y * np.conj(cyl_y))
        # plt.figure()
        # plt.imshow(np.log10(cyl_y), aspect='auto', extent=[has[0], has[-1], 0, cyl_y.shape[0]])
        # plt.colorbar(label='Log10 Intensity')
        # plt.xlabel('HA (degrees)')
        # plt.ylabel('Feed Index')
        # plt.show()
        cyl_y_averaged = np.nanmean(cyl_y, axis=1)

    #combined cylx and cyly
    response = cyl_x_averaged + cyl_y_averaged
    response = response / np.nanmean(response[:,zha_idx])
    out_fn = data.split(".h5")[0] + "_response.npz"
    import pdb; pdb.set_trace()
    np.savez(out_fn,HA=has,freqs=freqs,intensity_norm=response,xx=cyl_x,yy=cyl_y)

    return response, has, freqs


if __name__ == "__main__":
    '''

    Loads in holography data
    For one frequency, creates .npz file with response for XX pol, YY pol, and intensity.
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn", type=str, default=None, nargs="+", help="Holography data file")
    args = parser.parse_args()


    # Load in the data file
    filenames = args.fn
    print(filenames)
    for filename in filenames:
        print(f"Processing {filename}")
        f = h5py.File(filename, "r")
        # Frequency index
        response, has, freqs = respond(filename, plot=True)
