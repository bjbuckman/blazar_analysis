import numpy as np
import os
from scipy.interpolate import interp1d
import sys

obj_num = int(sys.argv[1]) ## Min optical object number
obj_min = int(sys.argv[2]) ## Min gamma object number
obj_max = int(sys.argv[3]) ## Max gamma object number
out_num = int(sys.argv[4]) ## Bookkeeping number from previous files

opt_num = np.arange(obj_min,obj_max+1) ## array of optical object numbers
gam_num = np.arange(obj_min,obj_max+1) ## array of gamma object numbers

##
## Combine all cross object correlations
##
for ii in range(0,len(opt_num)):
	if (not (opt_num[ii] == obj_num)):
		filename = './cross_object_gam+opt/cross_object_gam'+str(obj_num).zfill(4)+'_opt'+str(opt_num[ii]).zfill(4)+'_'+str(out_num).zfill(3)+'.dat'
		try:
			read_in = np.loadtxt(filename)
		
			time = read_in[0]
			data = read_in[1:]
			
			try:
				data_all = np.append(data_all, data, axis=0)
			except:
				data_all = data
			
		except:
			pass

for ij in range(0,len(gam_num)):
	if (not (gam_num[ij] == obj_num)):
		filename = './cross_object_gam+opt/cross_object_gam'+str(gam_num[ij]).zfill(4)+'_opt'+str(obj_num).zfill(4)+'_'+str(out_num).zfill(3)+'.dat'
		try:
			read_in = np.loadtxt(filename)
		
			time = read_in[0]
			data = read_in[1:]
			
			try:
				data_all = np.append(data_all, data, axis=0)
			except:
				data_all = data
			
		except:
			pass
##
## Sort and calc sigmas
##
num_all = data_all.shape[0]-1
data_all.sort(axis=0)
data_all = np.ma.masked_equal(data_all, -1.1)
data_all = np.ma.filled(data_all, np.nan)

time = np.array([time])

## Calculate mean and multi-sigma values of correlation coefficient
mean     = 0.5

sig_1_dn = 0.158655254
sig_1_up = 0.841344746

sig_2_dn = 0.022750132
sig_2_up = 0.977249868

sig_3_dn = 0.001349898
sig_3_up = 0.998650102

sig_a_dn = 0.0
sig_a_up = 1.0

stats_arr = np.array([ mean, sig_1_dn, sig_1_up, sig_2_dn, sig_2_up, sig_3_dn, sig_3_up, sig_a_dn, sig_a_up])*100.

# index_arr = np.arange(0,num_all+1)
# stats_func = interp1d(index_arr, data_all, axis=0)

corr_stats = np.nanpercentile(data_all, stats_arr, axis=0)

output_arr = np.append(time, corr_stats, axis=0)

output_filename = 'cross_object'+str(obj_num).zfill(4)+'_all_stats_'+str(out_num).zfill(3)+'.dat'
np.savetxt(output_filename, output_arr)

