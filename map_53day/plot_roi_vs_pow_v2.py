import numpy as np
import pandas as pd
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from astropy.timeseries import LombScargle


object_arr = range(0,400)

gam_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/data/gamma-ray/v3/'
fermi_source_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/fermi_analysis/sources/'


## list of objects
input_csv = '/mnt/c/Users/psyko/Physics/gamma-optical/data/fermi_4FGL_associations_ext_GRPHorder.csv'

data_in = pd.read_csv(input_csv)
data_in.drop(columns='Unnamed: 0', inplace=True)

###
### PLOT
###
font = {'family' : 'serif',
		'weight' : 'normal',
		'size'   : 12}

matplotlib.rc('font', **font)

cmap2 = plt.get_cmap('gist_stern')
cmap  = plt.get_cmap('brg')#'gnuplot2')#'BuPu_r')
fig = plt.figure()
	
gs = gridspec.GridSpec(2, 2)
# gs.update(wspace=0.0, hspace=0.0)
# plt.subplots_adjust(hspace=0.001)

ax0 = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs[0,1])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[1,1])

for axis in ['top','bottom','left','right']:
	ax0.spines[axis].set_linewidth(1)
ax0.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
ax0.tick_params(which='minor',width=0.25, length=5, direction='in')

for axis in ['top','bottom','left','right']:
	ax1.spines[axis].set_linewidth(1)
ax1.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
ax1.tick_params(which='minor',width=0.25, length=5, direction='in')

for axis in ['top','bottom','left','right']:
	ax2.spines[axis].set_linewidth(1)
ax2.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
ax2.tick_params(which='minor',width=0.25, length=5, direction='in')

for axis in ['top','bottom','left','right']:
	ax3.spines[axis].set_linewidth(1)
ax3.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
ax3.tick_params(which='minor',width=0.25, length=5, direction='in')

MSIZE = 0.6
ALPHA = 0.3
ELINE = 0.5


period = np.linspace(52, 54, num=10)
freq = 1/period

power_array = np.zeros([len(object_arr),5])
roi_array = np.zeros([len(object_arr),4])

counter = 0
for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]
	
	power_array[counter,0] = obj_num
	
	try:
		FGL_name = data_in.loc[ii].name_4FGL
		FGL_name = FGL_name.replace(' ', '_')

		out7_file = fermi_source_dir+FGL_name+'_out7_rois.txt'
		out7_dat = np.loadtxt(out7_file)
		## energy, column
		roi = out7_dat[:4,2]
		
		gam_file = gam_dir+'object'+str(obj_num).zfill(4)+'_gam.npy'
		
		read_in = np.load(gam_file)
		
		read_in = read_in[:,:4,:]  ## get 4 lowest energies
		COUNTS = read_in[:,:,5]  ## daily countsget counts
		FLUX = read_in[:,:,1]
		FLUX_err = (read_in[:,:,2] + read_in[:,:,3])/2
		
		time = read_in[:,0,0]/(60*60*24) + 2451910.5
		time -= time.min()
		
		# print(time.shape, COUNTS.shape)
		pg = np.zeros([4,10])
		for jj in range(0,4):
			# pg[jj] = LombScargle(time, COUNTS[:,jj]).power(freq)
			pg[jj] = LombScargle(time, FLUX[:,jj], dy=FLUX_err[:,jj]).power(freq)
		
		pg = np.max(pg, axis=1)
		# print(pg)
		power_array[counter,1:] = pg
		roi_array[counter] = roi
		
		# ax0.plot(roi, pg, alpha=0.5)
		# ax0.scatter(roi[0], pg[0])
		print(roi, pg)
		
		counter+= 1
	except:
		continue

roi_array = roi_array[:counter]
power_array = power_array[:counter]

print('object '+str(power_array[np.argmax(power_array[:,4]),0]))

SS = 1
COLOR = cmap(power_array[:,0]/800)

ax0.scatter(roi_array[:,0], power_array[:,1], marker='o', s=SS, c=COLOR)
ax1.scatter(roi_array[:,1], power_array[:,2], marker='o', s=SS, c=COLOR)
ax2.scatter(roi_array[:,2], power_array[:,3], marker='o', s=SS, c=COLOR)
ax3.scatter(roi_array[:,3], power_array[:,4], marker='o', s=SS, c=COLOR)

ax0.set_yscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')

ax0.set_ylim([1.e-5,1.e-1])
ax1.set_ylim([1.e-5,1.e-1])
ax2.set_ylim([1.e-5,1.e-1])
ax3.set_ylim([1.e-5,1.e-1])

# ax0.set_ylabel('Max Power between 52-54 days')
# ax0.set_xlabel('ROI')

plt.tight_layout()
# if ZOOM == 0:
# ax1.set_xlim([time_min,time_max])
# plot_name = './graph_53day_roi_vs_counts_power_'
plot_name = './graph_53day_roi_vs_flux_power_scatter_'
for n in range(0,100):
	if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
		continue
	else:
		plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
		break
plt.close()