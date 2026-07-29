import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from beam_model import utils


loc = EarthLocation.of_site('CHIME')

source = SkyCoord.from_name("Crab Pulsar")
source_ra = source.ra.deg * u.deg
source_dec = source.dec.deg * u.deg

lst_list = np.linspace(0, 24, 2000) * u.hourangle
has = lst_list.to(u.deg) - source_ra

def get_app_dec(ha, lat=loc.lat, dec=source_dec):
    ha = ha.to(u.rad)
    lat = lat.to(u.rad)
    dec = dec.to(u.rad)
    real_dec = np.cos(lat)*np.sin(dec) - np.sin(lat)*np.cos(dec)*np.cos(ha)
    app_dec = np.arcsin(real_dec) + lat
    return app_dec.to(u.deg)

app_decs=[]

for h in has:
    app_dec = get_app_dec(h).to_value()
    app_decs.append(app_dec)

app_decs = np.array(app_decs)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(lst_list, app_decs, color='black', linestyle='--', label='PSR B0531+21 (Crab)')
ax.axhline(y=83.5, linestyle='-.', label='No longer visible')
ax.axhline(y=source_dec.to_value(), label='Actual declination', color='k')
ax.set_xlabel('Detected LST (deg)', fontsize=12)
ax.set_ylabel('Detected Declination (deg)', fontsize=12)
ax.set_xlim(0, 24)
ax.set_ylim(15, 105)
ax.set_xticks(np.arange(0, 25, 3))
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend(loc='upper right', frameon=True)

plt.title('Apparent Declination vs. LST', fontsize=13)
plt.savefig('Incorrect_localization_plot.png', dpi=800)
plt.show()


