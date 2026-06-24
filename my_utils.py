import numpy as np 
import matplotlib.pyplot as plt
import pdb
#from iautils import cascade
import pickle
import pdb
import copy
import os
import sys
from uncertainties import ufloat
from uncertainties import unumpy as unp
import re
from glob import glob


def flux_to_luminosity(peak_flux):
    result = 4 * np.pi * np.square(6.171 * 10**19) * peak_flux * 10**(-19)
    return result 


def load_cascade_any(path: str):
    # Allow loading for pkl files
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            cascade_data = pickle.load(f)
        return cascade_data
    else:
        return cascade.load_cascade_from_file(path)



pdb.set_trace()