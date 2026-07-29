import numpy as np
import astropy.units as u
from astropy.time import Time 
from astroplan import Observer, FixedTarget, is_observable, is_event_observable
from astroplan.plots import plot_finder_image
from astroplan.constraints import Constraint, AltitudeConstraint
import matplotlib.pyplot as plt
import pdb

ha_lim = 60
dt = 5 * u.minute
start_mjd = 58406
end_mjd = 60776

chime = Observer.at_site('chime')
crab = FixedTarget.from_name('Crab Pulsar')

t_start = Time(start_mjd, format='mjd', scale='utc')
t_end = Time(end_mjd, format='mjd', scale='utc')
n_steps = int(((t_end - t_start) / dt).decompose().value)
times_grid = t_start + np.arange(n_steps) * dt

alt_constraint = AltitudeConstraint(min=0 * u.deg) #Has to be in the sky 
alt_mask = is_event_observable(
    constraints=[alt_constraint],
    observer=chime,
    target=crab,         
    times=times_grid
)[0]

ha = chime.target_hour_angle(times_grid, crab) #Calculate HA for each
ha_deg = ha.wrap_at(180 * u.deg).deg

ha_mask_e = (ha_deg >= -180) & (ha_deg <= -ha_lim)
ha_mask_w = (ha_deg >= ha_lim) & (ha_deg <= 180)

e_steps = alt_mask & ha_mask_e
w_steps = alt_mask & ha_mask_w
total_steps = sum(e_steps + w_steps) 

total_obs_time = (sum(alt_mask)*dt).to(u.hour)
total_eff_time = (total_steps * dt).to(u.hour)

print(f"""Between MJD {start_mjd} - {end_mjd}:
            Crab was in the sky above CHIME for {total_obs_time:.2f}
            Crab was outside +/- {ha_lim} for {total_eff_time:.2f}""")

pdb.set_trace()