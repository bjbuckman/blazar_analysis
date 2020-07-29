import pandas as pd
import numpy as np
import os
import sys

from scipy.stats import spearmanr

obj_num = int(sys.argv[1]) ## Object number to use
optic_file = str(sys.argv[2]) ## Input optical file
out_num = int(sys.argv[3]) ## Output number (for bookkeeping)
dt = float(sys.argv[4]) ## time to average data over (0.45)
DT = float(sys.argv[5]) ##+-DT is the time bin to calc mean (10000)
num_iterations = int(sys.argv[6]) ## How many times to calculate corr_coeff

print(dt,DT,num_iterations)

###
###
### Getting full time array
###
###

read_in = pd.read_csv(optic_file)

time = read_in['HJD'].values
time_min_opt = round(time.min())
time_max_opt = round(time.max())

time_min = time_min_opt
time_max = time_max_opt

t_bin = 0.4
t = np.arange(0,time_max-time_min,t_bin) ## Working time array
tnum = len(t) ## number of time values

###
###
### OPTICAL
###
###

read_in = pd.read_csv(optic_file)
data = read_in.values.T

time = read_in['HJD'].values - time_min
flux = read_in['flux'].values
flux_err = read_in['flux_err'].values
flux_err_2 = flux_err**2

##
## Regridding over dt days
## Doing this for simplicity
## Assuming independence of measurements,
## We just avg mu values and inv-sum sigma
f = np.zeros(tnum)-100
fe = np.zeros(tnum)-1

for ii in range(0,tnum):
	t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt) ## Average over +- dt
	t_arr = np.logical_and( t_arr, flux!=99.99) ## Ignore bad values
	if np.any(t_arr):
		f[ii] = np.mean(flux[t_arr])
		fe[ii] = (np.sum(flux_err_2[t_arr])/np.sum(t_arr))**0.5

##
## Normalize values using the mean
## If DT is >4000, finds total mean
F = np.zeros(tnum)-100
Fe = np.zeros(tnum)-10

for ii in range(0,tnum):
	t_arr = np.logical_and( t<=t[ii]+DT, t>=t[ii]-DT) ## Find median within +-DT days
	t_arr = np.logical_and( t_arr, f!=-100) ## Ignore bad values
	if np.any(t_arr) and f[ii]!=-100:
		f_median = np.median(f[t_arr])
		F[ii] = f[ii]/f_median
		Fe[ii] = fe[ii]/f_median

###
###
### Cross correlate
###
###

T = np.arange(0.,3000,t_bin) ## correlation time lag
Tnum = len(T) ## Number correlation times

num_redux = 20
## Foreward array
num_points_f = np.zeros(Tnum).astype(np.int64)
correlation_f = np.zeros([num_redux, Tnum])

## Output array
CORRELATION = np.zeros([num_iterations*num_redux+1, Tnum])
CORRELATION[0] = T

for ip in range(0,num_iterations):
	
	## New flux array for calculation
	F1 = np.zeros(tnum)-100
	for ii in range(0,tnum):
		if F[ii] != -100:
			F1[ii] = np.random.normal(loc=F[ii], scale=Fe[ii])
	
	## Forword
	for ii in range(0,Tnum):
		if ii == 0:
			arr_opt = F1
			arr_gam = F1
		else:
			arr_opt = F1[ii:]
			arr_gam = F1[:-ii]
		
		bool_arr = np.logical_and(arr_opt != -100, arr_gam != -100) ## Ignore bad points
		num_points_f[ii] = int(np.sum(bool_arr)) ## Number of points to correlate
		
		if int(num_points_f[ii]) > 5: ## Min number of points needed
			
			arr_opt = arr_opt[bool_arr]
			arr_gam = arr_gam[bool_arr]
			
			## Make array of points
			all_points = np.zeros([num_points_f[ii], 2])
			for ij in range(0,num_points_f[ii]):
				point_opt = arr_opt[ij]
				point_gam = arr_gam[ij]
				
				all_points[ij,0] = point_opt
				all_points[ij,1] = point_gam
				
			redux_size = int(np.ceil(num_points_f[ii]**0.8))
			redux_index = np.arange(0,num_points_f[ii])
			for jj in range(0,num_redux):
				rand_points = all_points[np.random.choice(redux_index, replace=False, size=redux_size)]
				## Calculate correlation coefficient
				corr, pval = spearmanr(rand_points) 
				correlation_f[jj,ii] = corr
		else:
			correlation_f[:,ii] = -1.1*np.ones(num_redux)
	
	CORRELATION[ip*num_redux+1:ip*num_redux+num_redux+1] = correlation_f

filename = './autocor_err/autocor_err_opt_'
np.savetxt(filename+str(out_num).zfill(3)+'.dat', CORRELATION)

