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


# ZOOM = 1
object_arr = range(0,100)

cdir = '../../'

autocorr_gam_err_dir = cdir+'2001/output/auto_gam/'
autocorr_gam_sig_dir = cdir+'2001/output/autocorr_gam+gam/'
gam_file_num = 0

input_csv = cdir+'data/fermi_4FGL_associations_ext_GRPHorder.csv'

data_in = pd.read_csv(input_csv)
data_in.drop(columns='Unnamed: 0', inplace=True)

###
### PLOT
###
font = {'family' : 'serif',
		'weight' : 'normal',
		'size'   : 6}

matplotlib.rc('font', **font)

cmap2 = plt.get_cmap('gist_stern')
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
ax0.scatter(l_vela,b_vela, alpha=0.7, color='g', marker='D', zorder=7)

## SUN
rng = pd.date_range(start='2015-01-01 00:00', end='2016-01-01 00:00', freq='3D') 
l_sun = np.zeros(len(rng))
b_sun = np.zeros(len(rng))
for ii in range(0,len(rng)):
	ra_dec = sun.sky_position(rng[ii])
	c_sun = SkyCoord(ra=ra_dec[0].value, dec=ra_dec[1].value, unit=(u.hourangle, u.deg), frame='icrs')
	
	l_sun[ii] = c_sun.galactic.l.degree
	if l_sun[ii] > 180:
		l_sun[ii]-=360
	b_sun[ii] = c_sun.galactic.b.degree

print(l_sun, b_sun)
l_sun = np.radians(l_sun)
b_sun = np.radians(b_sun)

ax0.scatter(l_sun,b_sun, color='y', marker='o', s=3, alpha=0.8, zorder=5.4)


special_obj = np.array([99,33,26,82,59])

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
		
		excess = period_val_avg/period_range_avg
		
		print(obj_num, excess, period_val_avg)
		
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
		# print(obj_num, excess
		
		# print(np.sum(special_obj == obj_num))
		
		if np.sum(special_obj == obj_num):
			MARKER = 'X'
			COLOR = 'm'


	except:
		continue
		
	
	# print(l,b)
	# ax0.scatter(ra,dec, alpha=excess, color=COLOR, marker=MARKER)
	ax0.scatter(l,b, alpha=excess, color=COLOR, marker=MARKER, zorder=ZORDER)
	
	# if auto_gam_err_tag == 1:
		# ax0.errorbar(time_gam_err, cor_mean_gam, yerr=[cor_err_d_gam,cor_err_u_gam], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		# ax0.set_xlim([-0.001,2999.999])
		# ax0.set_ylim([-0.999,0.999])
		
		# # ax11.set_xticklabels([])
		# # ax11.set_yticklabels([])
		
		# # ax01.yaxis.set_ticks_position('right')
		
		# # ax01.xaxis.set_ticks_position('top')
		# # plt.setp(ax01.get_xticklabels(), ha="left", rotation=45)
		# # ax01.set_xlabel(r'Days')
		# # ax01.xaxis.set_label_position('top') 
		# # ax01.set_ylabel(r'$r_s$')
		# # ax01.yaxis.set_label_position('right')
		
		# # ax01.text(-1850., 0.85, r'AC', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		# # ax01.text(150., -0.85, r'$\gamma$-ray$\rightarrow$', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		# # ax01.text(-150., -0.85, r'$\leftarrow$Optical', verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		# # ax02 = plt.subplot(gs[0:2,4:6])
		# # ax02.plot(np.fft.fftfreq(len(time_gam_err), d=0.4), np.fft.fft(cor_mean_gam))
		# # # ax02.set_xscale('log')
		# # ax02.set_xlim([1.7e-2,2.e-2])
		
		# ax1.plot(1/freq, abs(sp), color='k', alpha=ALPHA)
		# # ax1.plot(1/freq, sp.real, color='b', alpha=ALPHA)
		# # ax1.plot(1/freq, sp.imag, color='r', alpha=ALPHA)
		
		# # ax1.plot(1/freq, sp_l.real, color='b', alpha=ALPHA)
		# # ax1.plot(1/freq, sp_l.imag, color='g', alpha=ALPHA)
		
		# # ax1.plot(1/freq, sp_u.real, color='b', alpha=ALPHA)
		# # ax1.plot(1/freq, sp_u.imag, color='g', alpha=ALPHA)
		
		# ax1.set_xlim(40.,90.)
		# # y0_max = np.amax()
		# ax1.set_ylim(0.,100.)
		
		# # plt.setp(ax0.get_xticklabels(), ha='right', rotation=45)
		# # ax0.zorder=1000
		
			


plt.tight_layout()
# if ZOOM == 0:
# ax1.set_xlim([time_min,time_max])
plot_name = './map_53day_gal_'
for n in range(0,100):
	if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
		continue
	else:
		plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
		break
plt.close()
		
	# elif ZOOM == 1:
		# ii_max = int(3000/300)
		# time_min = -1000.
		# for ii in range(0,ii_max):
			# ax1.set_xlim([300*ii+time_min,300*ii+400+time_min])
			# plot_name = '../analysis/cc-zoom/plot_cc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			# for n in range(0,100):
				# if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					# continue
				# else:
					# plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					# break

		# plt.close()
		
	# elif ZOOM == 2:
		# ii_max = int(4000/150)
		# time_min = -1000.
		# for ii in range(0,ii_max):
			# ax1.set_xlim([150*ii+time_min,150*ii+200+time_min])
			# plot_name = '../analysis/cc-zoom/plot_cc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			# for n in range(0,100):
				# if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					# continue
				# else:
					# plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					# break

		# plt.close()
		
	# print('Plotted object'+str(obj_num).zfill(4))

