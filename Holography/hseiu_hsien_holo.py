from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle

from ch_util import tools
import sys
from scipy.stats import iqr

def mask_bad_feed(data_slice,has,thresh=2):
    #collapse the dataslice across the HA axis
    #only get the data for |HA| > 10 degrees
    data_slice_cut = data_slice[:,:,(np.abs(has) > 10)]
    data_slice_cut[data_slice_cut == 0] = np.nan
    #divide each data_slice_cut by the 0ha slice
    data_slice_cut = data_slice_cut / data_slice[:,:,np.argmin(np.abs(has))][:,:,np.newaxis]
    mean_feed = np.nanmean(np.abs(data_slice_cut)**2, axis=2)
    #find the median and iqr of the mean_feed
    mask_feed = np.zeros(mean_feed.shape[1], dtype=bool)
    mask_freq = np.zeros(mean_feed.shape[0], dtype=bool)
    #iteratively mask feeds and freqs that are outliers
    for i in range(5):
        med_feed = np.nanmedian(mean_feed,axis=0)
        iqr_val_feed = iqr(med_feed[~np.isnan(med_feed)])
        mask_feed |= np.abs(med_feed - np.nanmedian(med_feed)) > thresh * iqr_val_feed
        print(f"Iteration {i}: Masked {np.sum(mask_feed)} feeds")
        mean_feed[:,mask_feed] = np.nan

        med_freq = np.nanmedian(mean_feed,axis=1)
        iqr_val_freq = iqr(med_freq[~np.isnan(med_freq)])
        mask_freq |= np.abs(med_freq - np.nanmedian(med_freq)) > thresh * iqr_val_freq
        print(f"Iteration {i}: Masked {np.sum(mask_freq)} freqs")
        mean_feed[mask_freq,:] = np.nan


    med_feed[mask_feed] = np.nan
    med_freq[mask_freq] = np.nan
    plt.figure()
    plt.plot(med_feed, label='Median Feed')
    plt.axhline(np.nanmedian(med_feed), color='r', linestyle='--', label='Median of Median Feed')
    plt.axhline(np.nanmedian(med_feed) + thresh * iqr_val_feed, color='g', linestyle='--', label='Upper Threshold')
    plt.axhline(np.nanmedian(med_feed) - thresh * iqr_val_feed, color='g', linestyle='--', label='Lower Threshold')
    plt.xlabel('Feed Index')
    plt.figure()
    plt.plot(med_freq, label='Median Frequency')
    plt.axhline(np.nanmedian(med_freq), color='r', linestyle='--', label='Median of Median Frequency')
    plt.axhline(np.nanmedian(med_freq) + thresh * iqr_val_freq, color='g', linestyle='--', label='Upper Threshold')
    plt.axhline(np.nanmedian(med_freq) - thresh * iqr_val_freq, color='g', linestyle='--', label='Lower Threshold')
    plt.xlabel('Frequency Index')

    #form a 2d mask from the 1d masks
    # mean_feed[mask_freq,:] = np.nanmedian(mean_feed)
    # mean_feed[:,mask_feed] = np.nanmedian(mean_feed)
    # mean_feed[mean_feed == 0] = np.nanmedian(mean_feed)
    plt.figure()
    plt.imshow(np.log10(mean_feed), aspect='auto')
    plt.colorbar(label='Log10 Intensity')
    plt.xlabel('Feed Index')
    plt.ylabel('Frequency Index')
    plt.show()
    import pdb; pdb.set_trace()
    data_slice[mask_freq,:,:] = np.nan
    data_slice[:,mask_feed,:] = np.nan

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
        cyl_x = mask_bad_feed(cyl,has)
        # Make the slice I want
        cyl_x = np.abs(cyl_x * np.conj(cyl_x))  # Intensity
        # plt.figure()
        # plt.imshow(np.log10(cyl_x), aspect='auto', extent=[has[0], has[-1], 0, cyl_x.shape[0]])
        # plt.colorbar(label='Log10 Intensity')
        # plt.xlabel('HA (degrees)')
        # plt.ylabel('Feed Index')
        # plt.show()
        cyl_x = np.nanmean(cyl_x, axis=1)  # Average over feeds

    for slide in ycyls:
        my_slice = slice(ycyls[slide][0], ycyls[slide][1])
        cyl = beam_dset[:,0,my_slice,:]
        weight = weight_dset[:,0,my_slice,:]
        # cyl = cyl * weight
        cyl_y = mask_bad_feed(cyl,has)
        # Make the slice I want
        cyl_y = np.abs(cyl_y * np.conj(cyl_y))
        # plt.figure()
        # plt.imshow(np.log10(cyl_y), aspect='auto', extent=[has[0], has[-1], 0, cyl_y.shape[0]])
        # plt.colorbar(label='Log10 Intensity')
        # plt.xlabel('HA (degrees)')
        # plt.ylabel('Feed Index')
        # plt.show()
        cyl_y = np.nanmean(cyl_y, axis=1)

    #combined cylx and cyly
    import pdb; pdb.set_trace()
    response = cyl_x + cyl_y
    response = response / np.nanmean(response[:,zha_idx])
    out_fn = data.split(".h5")[0] + "_response.npz"
    np.savez(out_fn,HA=has,freqs=freqs,intensity_norm=response,xx=cyl_x,yy=cyl_y)

    return response, has, freqs


'''

Loads in holography data
For one frequency, creates .npz file with response for XX pol, YY pol, and intensity.
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fn", type=str, default=None, help="Filename of holography data")
args = parser.parse_args()


# Load in the data file
filename = args.fn
f = h5py.File(filename, "r")
# Frequency index
response, has, freqs = respond(filename, plot=True)
