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
# source_dec = 22.0145
source_decs = np.linspace(-1, 89, 10)

for source_dec in source_decs:
    chime = config.chime
    lat = chime.lat
    lon = chime.lon
    chime = EarthLocation.of_site("chime")
    #create an array of times across one day
    times = Time("2000-01-01T12:00:00") + np.linspace(0, 24, 10000) * u.hour

    #calculate the sidereal times at chime
    lst = times.sidereal_time("mean", longitude=chime)

    deg_lst = lst.deg
    ha_from_crab = deg_lst - source_ra
    #position from equitorial
    pos = [utils.get_position_from_equatorial(source_ra, source_dec, t.datetime,0) for t in times]
    x = np.array([p[0] for p in pos])
    y = np.array([p[1] for p in pos])

    plt.plot(x, ha_from_crab, label=f"Dec: {source_dec} deg")
    plt.xlabel("x (deg)")
    plt.ylabel("Hour Angle (deg)")
    plt.axvline(0, color='r', linestyle='--')
    plt.axhline(0, color='r', linestyle='--')
plt.legend()
plt.show()
