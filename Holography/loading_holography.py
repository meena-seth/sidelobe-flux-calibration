from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle

from ch_util import tools
import sys


'''

Loads in holography data 
For one frequency, creates .npz file with response for XX pol, YY pol, and intensity.
Submit this in CEDAR for every frequency using submit_holo_script.sh

'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("f_want", type=float, default=None, help="frequency to extract (in MHz)")
parser.add_argument("--f_index", type=int, default=None, help="Index of frequency to extract")
parser.add_argument("--fn", type=str, default=None, help="Filename of holography data")
args = parser.parse_args()


# Load in the data file
filename = args.fn
f = h5py.File(filename, "r")
beam_dset = f["beam"] 
index_map = f['index_map']

# Hour angle axis
ha = index_map['pix'][:]['phi']

# Index of zero hour angle (i.e. transit)
zha = np.argmin(np.abs(ha))
n_time = len(ha)

# Frequency index
freq = index_map['freq'][:]

if args.f_want is None:
    fsel = args.f_index
else:
    fsel = np.argmin(np.abs(freq - args.f_want))

fsel = np.array([fsel])
# Indices of frequencies of interest
n_freq = 1
# Extract beam data set
beam_dset = f['beam'] # (freq, pol, feed, time)

beam = beam_dset[fsel]
weight_dset = f['weight']
weight = weight_dset[fsel]
# Normalize beams to 1 at zero hour angle
beam = beam * tools.invert_no_zero(beam[:,:,:,zha][:,:,:,np.newaxis])
plt.figure()
beam_plot = np.abs(beam[0,0])
plt.imshow(np.log10(beam_plot), aspect='auto', extent=[ha[0], ha[-1], 0, beam_plot.shape[0]])
plt.colorbar(label='Normalized Beam Response')
plt.xlabel('Hour Angle (Degrees)')
plt.ylabel('Feed Index')
plt.show()



###########

cylprods = {
    "0": [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")],
    "1": [("A", "B"), ("B", "C"), ("C", "D")],
    "2": [("A", "C"), ("B", "D")],
    "3": [("A", "D")],
}

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

allcyls = {"Y": ycyls, "X": xcyls}

cyl_seps = ["1", "2", "3"]



tel_pickle_path = "./tel.pickle"
with open(tel_pickle_path, "rb") as tel_f:
    tel = pickle.load(tel_f)

def process(beam, weight):

    nfreq, npol, _, npixel = beam.shape
    npol_prods = 4

    accumulate = np.zeros((nfreq, npol, npol_prods, npixel), dtype=np.complex64)
    count = np.zeros_like(accumulate, dtype=np.float32)

    # Mask anomalous samples. All inputs should be normalized to ~1 on
    # meridian.
    weight_mask = np.copy(weight).astype(int)
    weight_mask[np.abs(beam) > 2.0] = 0

    pol_axis = []

    for pp, (po, pcyls) in enumerate(allcyls.items()):

        pol_axis.append(po)

        for ew in cyl_seps:

            _mult_ew(ew, pp, po, pcyls, beam, weight_mask, accumulate, count)

    # Calling tools should be unecessary here, because draco version was broken for ints
    out_beam = accumulate * tools.invert_no_zero(count)

    return out_beam

def _mult_ew(
    ew,
    pp,
    po,
    pcyls,
    beam,
    weight_mask,
    acc,
    count,
):
    prodset = cylprods[ew]

    from itertools import product

    for (cyl1, cyl2) in prodset:

        # Iterate over co- and cross-products
        pol_prod = [0, 1]

        for prod, (pi, pj) in enumerate(product(pol_prod, pol_prod)):

            for ns in range(1, 255):

                st1, en1 = pcyls[cyl1]  # First and last input on cyl 1
                st2, en2 = pcyls[cyl2]  # "                         " 2

                st2 += ns  # Start second cyl ahead
                en1 -= ns  # End first cylinder same number of inputs behind

                pmask = np.where(
                    (po == tel.polarisation[st1:en1])
                    & (po == tel.polarisation[st2:en2]),
                    1,
                    0,
                )[:, np.newaxis]

                w_cyl1 = weight_mask[:, pi, st1:en1]
                w_cyl2 = weight_mask[:, pj, st2:en2]
                b_cyl1 = beam[:, pi, st1:en1]
                b_cyl2 = beam[:, pj, st2:en2]

                acc[:, pp, prod] += np.sum(
                    pmask * (w_cyl1 * b_cyl1) * np.conj(w_cyl2 * b_cyl2),
                    axis=1,
                )

                count[:, pp, prod] += np.sum(
                    pmask * w_cyl1 * w_cyl2,
                    axis=1,
                )

# Note this takes awhile to run
out = process(beam, weight)
out_shape = out.shape # axes are: frequency, Y/X pol, copol-copol/copol-cross/cross-copol/cross-cross, HA
# you probably want to keep the third index set to 0 when looking at results

out_yy = np.squeeze(np.abs(out[:, 0, 0]))
out_xx = np.squeeze(np.abs(out[:, 1, 0]))
plt.figure()
plt.plot(ha, out_yy, label='YY')
plt.plot(ha, out_xx, label='XX')
plt.xlabel('Hour Angle (Degrees)')
plt.ylabel('Normalized Beam Response')
plt.title(f'Frequency: {freq[fsel][0]} MHz')
#set to log scale
plt.yscale('log')
plt.legend()
plt.show()
import pdb; pdb.set_trace()
np.savez(f'{freq[fsel][0]}.npz', HA=ha, YY=out_yy, XX=out_xx)
