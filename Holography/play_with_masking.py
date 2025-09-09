import numpy as np
from hseiu_hsien_holo import mask_bad_feed
data = np.load('data.npy')
has = np.load('ha.npy')

mask_bad_feed(data, has)
