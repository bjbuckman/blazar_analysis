import numpy as np
import os
from scipy.interpolate import interp1d
import sys

obj_num = int(sys.argv[1]) ## Object number to use
imin = int(sys.argv[2]) ## correlation file bookkeeping min
imax = int(sys.argv[3]) ## correlation file bookkeeping max
iout = int(sys.argv[4]) ## output bookkeeping number

corr_coeff_file = np.arange(imin,imax+1)
# corr_signi_file = np.arange(imin,imax+1)

## Combine correlation coefficient into single list
for ii in range(0,len(corr_coeff_file)):
	filename =  './cor_err/cor_err_'+str(corr_coeff_file[ii]).zfill(3)+'.dat'
	read_in = np.loadtxt(filename)
	
	time = read_in[0]
	data = read_in[1:]
	
	if ii == 0:
		data_all = data
	else:
		data_all = np.append(data_all, data, axis=0)

## sort correlation coefficients
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

# #correlation significance
# for ii in range(0,len(corr_signi_file)):
	# filename = './cor_sig/cor_sig_'+str(corr_signi_file[ii]).zfill(3)+'.dat'
	# read_in = np.loadtxt(filename)
	
	# time = read_in[0]
	# data = read_in[1:]
	
	# if ii == 0:
		# data_all = data
	# else:
		# data_all = np.append(data_all, data, axis=0)

# num_all = data_all.shape[0]-1
# data_all.sort(axis=0)

# mean     = 0.5*num_all

# sig_1_dn = 0.158655254*num_all
# sig_1_up = 0.841344746*num_all

# sig_2_dn = 0.022750132*num_all
# sig_2_up = 0.977249868*num_all

# sig_3_dn = 0.001349898*num_all
# sig_3_up = 0.998650102*num_all

# sig_a_dn = 0.0*num_all
# sig_a_up = 1.0*num_all

# stats_arr = np.array([ mean, sig_1_dn, sig_1_up, sig_2_dn, sig_2_up, sig_3_dn, sig_3_up, sig_a_dn, sig_a_up])
# index_arr = np.arange(0,num_all+1)

# stats_func = interp1d(index_arr, data_all, axis=0)
# corr_stats = stats_func(stats_arr)

# output_arr = np.append(output_arr, corr_stats, axis=0)

## Output
output_filename = 'object'+str(obj_num).zfill(4)+'_stats'+str(iout).zfill(3)+'.dat'
np.savetxt(output_filename, output_arr)

