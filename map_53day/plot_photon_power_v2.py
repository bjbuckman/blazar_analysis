import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys

from astropy import units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import sun

# import matplotlib.tri as tri
from scipy.interpolate import griddata


# ZOOM = 1
object_arr = range(0,800)

autocorr_gam_err_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/2001/output/auto_gam/'
autocorr_gam_sig_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/2001/output/autocorr_gam+gam/'
gam_file_num = 0

input_csv = '/mnt/c/Users/psyko/Physics/gamma-optical/data/fermi_4FGL_associations_ext_GRPHorder.csv'

data_in = pd.read_csv(input_csv)
data_in.drop(columns='Unnamed: 0', inplace=True)

###
### PLOT
###
font = {'family' : 'serif',
		'weight' : 'normal',
		'size'   : 6}

matplotlib.rc('font', **font)

# cmap2 = plt.get_cmap('cividis_r')
cmap2 = plt.get_cmap('gist_stern_r')
cmap  = plt.get_cmap('brg')#'gnuplot2')#'BuPu_r')
fig = plt.figure()

ax0 = plt.subplot(111, projection="mollweide")
ax0.grid(True)

# gs = gridspec.GridSpec(1, 1)
# gs.update(wspace=0.0, hspace=0.0)
# plt.subplots_adjust(hspace=0.001)

# ax0 = plt.subplot(gs[0,0])
# ax1 = plt.subplot(gs[1,0:2])

# for axis in ['top','bottom','left','right']:
	# ax0.spines[axis].set_linewidth(1)
# ax0.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
# ax0.tick_params(which='minor',width=0.25, length=5, direction='in')

# for axis in ['top','bottom','left','right']:
	# ax1.spines[axis].set_linewidth(1)
# ax1.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
# ax1.tick_params(which='minor',width=0.25, length=5, direction='in')

# ax1.yaxis.set_ticks_position('both')
# ax1.xaxis.set_ticks_position('both')
# ax1.minorticks_on()
# ax1.grid(True)

MSIZE = 0.6
ALPHA = 0.3
ELINE = 0.5

## VELA
c_vela = SkyCoord('08 35 20.66 -45 10 35.2', unit=(u.hourangle, u.deg), frame='icrs')
l_vela = c_vela.galactic.l.degree
if l_vela > 180:
	l_vela-=360
b_vela = c_vela.galactic.b.degree

l_vela = np.radians(l_vela)
b_vela = np.radians(b_vela)
ax0.scatter(l_vela,b_vela, alpha=0.7, color='g', marker='D', zorder=7, s=2)


# ## 4FGL J0047.1-6203 
# c_vela = SkyCoord('00 42 6 -62 2 0', unit=(u.hourangle, u.deg), frame='icrs')
# l_vela = c_vela.galactic.l.degree
# if l_vela > 180:
	# l_vela-=360
# b_vela = c_vela.galactic.b.degree

# l_vela = np.radians(l_vela)
# b_vela = np.radians(b_vela)
# ax0.scatter(l_vela,b_vela, alpha=0.99, color='c', marker='D', zorder=7, s=10)

# ## SUN
# rng = pd.date_range(start='2015-01-01 00:00', end='2016-01-01 00:00', freq='3D') 
# l_sun = np.zeros(len(rng))
# b_sun = np.zeros(len(rng))
# for ii in range(0,len(rng)):
	# ra_dec = sun.sky_position(rng[ii])
	# c_sun = SkyCoord(ra=ra_dec[0].value, dec=ra_dec[1].value, unit=(u.hourangle, u.deg), frame='icrs')
	
	# l_sun[ii] = c_sun.galactic.l.degree
	# if l_sun[ii] > 180:
		# l_sun[ii]-=360
	# b_sun[ii] = c_sun.galactic.b.degree

# print(l_sun, b_sun)
# l_sun = np.radians(l_sun)
# b_sun = np.radians(b_sun)

# ax0.scatter(l_sun,b_sun, color='y', marker='o', s=3, alpha=0.8, zorder=5.4)

photon_power_in = np.loadtxt('photon_power_E.dat')
photon_power_obj = photon_power_in[:,0]
photon_power = np.sum(photon_power_in[:,1:], axis=1)



l_obj = np.zeros(len(object_arr))
b_obj = np.zeros(len(object_arr))
p_obj = np.zeros(len(object_arr))
phot_obj = np.zeros(len(object_arr))

counter = 0
for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]
	
	ra = data_in['associated_ra'].loc[obj_num]
	# if ra > 180.:
		# ra-= 360
	# ra = np.radians(ra)
	
	dec = data_in['associated_de'].loc[obj_num]
	# dec = np.radians(data_in['associated_de'].loc[obj_num])
	
	c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
	# print(c.galactic)
	# print(c.galactic.l.degree)
	
	l = c.galactic.l.degree
	if l > 180:
		l-=360
	b = c.galactic.b.degree
	
	# l_obj[counter] = np.radians(l)
	# b_obj[counter] = np.radians(b)
	
	l_obj[counter] = l
	b_obj[counter] = b
	
	l = np.radians(l)
	b = np.radians(b)
	
	MARKER = 'o'
	ZORDER = 5
	###
	###
	### AUTOCORRELATIONS
	###
	###
	auto_gam_err_tag = 1
	try:
		filename_err = autocorr_gam_err_dir+'object'+str(obj_num).zfill(4)+'_auto_gam_stats'+str(gam_file_num).zfill(3)+'.dat'
		read_in = np.loadtxt(filename_err)

		time_gam_err = read_in[0]
		data_gam_err = read_in[1:]

		cor_mean_gam = data_gam_err[0]
		cor_err_d_gam = cor_mean_gam-data_gam_err[1]
		cor_err_u_gam = data_gam_err[2]-cor_mean_gam
		
		sp = np.fft.fft(cor_mean_gam)
		sp = abs(sp)
		sp_l = np.fft.fft(cor_mean_gam-cor_err_d_gam)
		sp_u = np.fft.fft(cor_mean_gam+cor_err_u_gam)
		freq = np.fft.fftfreq(len(cor_mean_gam), d=0.4)
		
		period = 1/freq[1:]
		sp = sp[1:]
		
		period_range_arr = np.logical_and(period<70., period>45)
		period_val_arr = np.logical_and(period<54., period>52.)
		
		period_range_arr = np.logical_xor(period_range_arr, period_val_arr)
		
		period_range_avg = np.mean(sp[period_range_arr])
		period_val_avg = np.amax(sp[period_val_arr])
		
		p_obj[counter] = period_val_avg
		
		excess = period_val_avg/period_range_avg
		
		phot_obj[counter] = photon_power[photon_power_obj == obj_num]
		
		COLOR = 'r'
		if excess <= 1.1:
			COLOR = 'k'
		else:
			ZORDER = 5.5
			
		excess/=3.
		if excess >= 1:
			excess = 1.
			MARKER = '*'
			COLOR = 'b'
			ZORDER = 6
		# print(excess)
		ax0.scatter(l,b, alpha=excess, color=COLOR, marker=MARKER, zorder=ZORDER, s=2)
		counter+= 1
		
	except:
		continue
		

l_obj = l_obj[:counter]
b_obj = b_obj[:counter]
p_obj = (phot_obj[:counter])

# print(l_obj, b_obj, p_obj)

l_l = l_obj - 360
l_m = l_obj
l_r = l_obj + 360

b_u = 90. + abs(90. - l_obj)
b_m = b_obj
b_d = -90. - abs(-90. - l_obj)

l_l = np.radians(l_l)
l_m = np.radians(l_m)
l_r = np.radians(l_r)

b_u = np.radians(b_u)
b_m = np.radians(b_m)
b_d = np.radians(b_d)

l_all = np.concatenate((l_l,l_m,l_r,l_l,l_m,l_r,l_l,l_m,l_r))
b_all = np.concatenate((b_u,b_u,b_u,b_m,b_m,b_m,b_d,b_d,b_d))
p_all = np.concatenate((p_obj,p_obj,p_obj,p_obj,p_obj,p_obj,p_obj,p_obj,p_obj))

lb = np.array([l_all, b_all])

l_grid = np.linspace(-180,180, num=600)
b_grid = np.linspace(-90,90, num=300)

l_grid = np.radians(l_grid)
b_grid = np.radians(b_grid)

grid_l, grid_b = np.meshgrid(l_grid,b_grid)

grid_p = griddata(lb.T, p_all, (grid_l, grid_b), method='linear')

ax0.contourf(grid_l, grid_b, grid_p, cmap=cmap2, levels=20)

print('Minimum = '+str(np.amin(grid_p)))
print('Maximum = '+str(np.amax(grid_p)))

# triang = tri.Triangulation(l_obj, b_obj)

# tcf = ax0.tricontourf(triang, p_obj)
# fig.colorbar(tcf)

plt.tight_layout()
# if ZOOM == 0:
# ax1.set_xlim([time_min,time_max])
plot_name = './map_53day_gal_photon_power_'
for n in range(0,100):
	if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
		continue
	else:
		plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
		break
plt.close()
		


