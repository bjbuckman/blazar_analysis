import numpy as np
import pandas as pd
import os
import sys

from astropy.timeseries import LombScargle


object_arr = range(0,800)

gam_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/data/gamma-ray/v3/'
fermi_source_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/fermi_analysis/sources/'


## list of objects
input_csv = '/mnt/c/Users/psyko/Physics/gamma-optical/data/fermi_4FGL_associations_ext_GRPHorder.csv'

data_in = pd.read_csv(input_csv)
data_in.drop(columns='Unnamed: 0', inplace=True)

period = np.linspace(52, 54, num=10)
freq = 1/period

power_array = np.zeros([len(object_arr),5])
counter = 0
for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]
	
	power_array[ii,0] = obj_num
	# FGL_name = data_in.loc[ii].name_4FGL
	# FGL_name = FGL_name.replace(' ', '_')
	
	# out7_file = fermi_source_dir+FGL_name+'_out7_rois.txt'
	# out7_dat = np.loadtxt(out7_file)
	# ## energy, column
	# out7_dat = out7_dat[:4]
	# print(out7_dat.shape)
	
	try:
		gam_file = gam_dir+'object'+str(obj_num).zfill(4)+'_gam.npy'
		
		read_in = np.load(gam_file)
		
		read_in = read_in[:,:4,:]  ## get 4 lowest energies
		COUNTS = read_in[:,:,5]  ## daily countsget counts
		
		time = read_in[:,0,0]/(60*60*24) + 2451910.5
		time -= time.min()
		
		# print(time.shape, COUNTS.shape)
		pg = np.zeros([4,10])
		for jj in range(0,4):
			pg[jj] = LombScargle(time, COUNTS[:,jj]).power(freq)
		
		pg = np.max(pg, axis=1)
		# print(pg)
		power_array[ii,0] = obj_num
		
		counter+= 1
	except:
		continue

# power_array = power_array[:counter]
print(power_array)

filename_out = 'photon_power_E.dat'
np.savetxt(filename_out, power_array)