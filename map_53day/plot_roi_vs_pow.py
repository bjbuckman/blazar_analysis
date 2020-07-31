import numpy as np
import pandas as pd
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from astropy.timeseries import LombScargle


object_arr = range(0,800)

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
	
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.0, hspace=0.0)
plt.subplots_adjust(hspace=0.001)

ax0 = plt.subplot(gs[0,0])
for axis in ['top','bottom','left','right']:
	ax0.spines[axis].set_linewidth(1)
ax0.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
ax0.tick_params(which='minor',width=0.25, length=5, direction='in')

MSIZE = 0.6
ALPHA = 0.3
ELINE = 0.5


period = np.linspace(52, 54, num=10)
freq = 1/period

power_array = np.zeros([len(object_arr),5])

counter = 0
for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]
	
	power_array[ii,0] = obj_num
	
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
		power_array[ii,1:] = pg
		
		ax0.plot(roi, pg, alpha=0.5)
		print(roi, pg)
		
		counter+= 1
	except:
		continue

print('object '+str(power_array[np.argmax(power_array[:,4]),0]))

ax0.set_yscale('log')
ax0.set_ylim([1.e-5,1.e-1])

ax0.set_ylabel('Max Power between 52-54 days')
ax0.set_xlabel('ROI')

plt.tight_layout()
# if ZOOM == 0:
# ax1.set_xlim([time_min,time_max])
# plot_name = './graph_53day_roi_vs_counts_power_'
plot_name = './graph_53day_roi_vs_flux_power_'
for n in range(0,100):
	if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
		continue
	else:
		plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
		break
plt.close()