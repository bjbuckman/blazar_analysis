import pandas as pd
import numpy as np
import os
import sys

from scipy.stats import halfnorm
from scipy.stats import spearmanr
from random import randint

obj_num = int(sys.argv[1]) ## Object number to use
optic_file = str(sys.argv[2]) ## Input optical file
gamma_file = str(sys.argv[3]) ## Input gamma file
out_num = int(sys.argv[4]) ## Output number (for bookkeeping)
dt = float(sys.argv[5]) ## time to average data over (0.45)
DT = float(sys.argv[6]) ##+-DT is the time bin to calc mean (10000)
num_iterations = int(sys.argv[7]) ## How many times to calculate corr_coeff

###
###
### Getting full time array
###
###

read_in = pd.read_csv(optic_file)

time = read_in['HJD'].values
time_min_opt = round(time.min())
time_max_opt = round(time.max())

read_in = np.loadtxt(gamma_file).T
time = read_in[0]/(60*60*24) + 2451910.5
time_min_gam = round(time.min())
time_max_gam = round(time.max())

time_min = round(min(time_min_opt,time_min_gam))
time_max = round(max(time_max_opt,time_max_gam))

t_bin = 0.4
t = np.arange(time_min,time_max,t_bin)
tnum = len(t)

###
###
### OPTICAL
###
###

read_in = pd.read_csv(optic_file)

time = read_in['HJD'].values #- time_min
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
	if np.any(t_arr) and f[ii] != -100:
		f_median = np.median(f[t_arr])
		F[ii] = f[ii]/f_median
		Fe[ii] = fe[ii]/f_median
		
t_opt = t
F_opt = F
fe_opt = Fe

###
###
### GAMMA-RAY
###
###

read_in = np.loadtxt(gamma_file).T

time = read_in[0]/(60*60*24) + 2451910.5 #- time_min
flux = read_in[1]
gamma_lerr_data = read_in[2]
gamma_uerr_data = read_in[3]
flux_err_d = gamma_lerr_data
flux_err_u = gamma_uerr_data
flux_err_2d = flux_err_d**2
flux_err_2u = flux_err_u**2

##
## Regridding over dt days
## Doing this for simplicity
## Assuming independence of measurements,
## We just avg mu values and inv-sum sigma
f = np.zeros(tnum)-100
fe_d = np.zeros(tnum)-1
fe_u = np.zeros(tnum)-1

for ii in range(0,tnum):
	t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt) ## Average over +- dt
	t_arr = np.logical_and( t_arr, flux>0) ## Ignore bad values if any
	if np.any(t_arr):
		f[ii] = np.mean(flux[t_arr])
		fe_d[ii] = (np.sum(flux_err_2d[t_arr])/np.sum(t_arr))**0.5
		fe_u[ii] = (np.sum(flux_err_2u[t_arr])/np.sum(t_arr))**0.5

##
## Normalize values using the mean
## If DT is >4000, finds total mean
F = np.zeros(tnum)-100
Fe_d = np.zeros(tnum)-10
Fe_u = np.zeros(tnum)-10

for ii in range(0,tnum):
	t_arr = np.logical_and( t<=t[ii]+DT, t>=t[ii]-DT) ## Find median within +-DT days
	t_arr = np.logical_and( t_arr, f!=-100) ## Ignore bad values
	if np.any(t_arr) and f[ii] != -100:
		f_median = np.median(f[t_arr])
		F[ii] = f[ii]/f_median
		Fe_d[ii] = fe_d[ii]/f_median
		Fe_u[ii] = fe_u[ii]/f_median
			
t_gam = t
F_gam = F
fe_gam_d = Fe_d
fe_gam_u = Fe_u

###
###
### Cross correlate
###
###

## Times to calculate cross correlations
# time_start = 2454684.15 + 25.
# time_end   = 2458743.87 - 25. 
# time_of_correlation = np.arange(time_start, time_end, 25.)
time_of_correlation = t
num_time_steps = len(time_of_correlation)

##
## Creating array
# T = np.arange(0.,50.,t_bin) ## correlation time lag
T = np.array([-50,0,50])
Tnum = len(T) ## Number correlation times

T_b = T[::-1]
time_shift = np.concatenate((-T_b[:-1], T))

## Foreward array
# num_points_f = np.zeros(Tnum).astype(np.int64)
correlation_f = np.zeros(Tnum)

## Backwords array
# num_points_b = np.zeros(Tnum).astype(np.int64)
correlation_b = np.zeros(Tnum)

## Output arrays
##
## time_of_correlation
## time_shift
CORRELATION = np.zeros([num_iterations, num_time_steps, 2*Tnum-1])

for ip in range(0,num_iterations):
	
	## New flux arrays for calculation
	F1_gam = np.zeros(tnum)-100
	F1_opt = np.zeros(tnum)-100
	for ii in range(0,tnum):
	
		## Gamma-ray
		if F_gam[ii] != -100:
			if randint(0,1) == 0:
				F1_gam[ii] = F_gam[ii] - halfnorm.rvs(scale=fe_gam_d[ii])
			else:
				F1_gam[ii] = F_gam[ii] + halfnorm.rvs(scale=fe_gam_u[ii])
		
		## Optical
		if F_opt[ii] != -100:
			F1_opt[ii] = np.random.normal(F_opt[ii], fe_opt[ii])
	
	for it in range(0,num_time_steps):
		bool_arr_temp = np.logical_and(t>=time_of_correlation[it]-50., t<=time_of_correlation[it]+50.)
		
		F1_opt2 = F1_opt[bool_arr_temp]
		F1_gam2 = F1_gam[bool_arr_temp]
			
		## Forword
		for ii in range(0,Tnum):
			if ii == 0:
				arr_opt = F1_opt2
				arr_gam = F1_gam2
			else:
				arr_opt = F1_opt2[ii:]
				arr_gam = F1_gam2[:-ii]

			bool_arr = np.logical_and(arr_opt != -100, arr_gam != -100) ## Ignore bad points
			
			num_points_f = int(np.sum(bool_arr)) ## Number of points to correlate
			
			if int(num_points_f) > 6: ## Min number of points needed
				arr_opt = arr_opt[bool_arr]
				arr_gam = arr_gam[bool_arr]
				
				## Make array of points
				all_points = np.zeros([num_points_f, 2])
				for ij in range(0,num_points_f):
					point_opt = arr_opt[ij]
					point_gam = arr_gam[ij]
					
					all_points[ij,0] = point_opt
					all_points[ij,1] = point_gam
					
				## Calculate correlation coefficient
				corr, pval = spearmanr(all_points) 
				correlation_f[ii] = corr
			else:
				correlation_f[ii] = -1.1

		## Backword
		for ii in range(0,Tnum):
			if ii == 0:
				arr_opt = F1_opt2
				arr_gam = F1_gam2
			else:
				arr_opt = F1_opt2[:-ii]
				arr_gam = F1_gam2[ii:]

			bool_arr = np.logical_and(arr_opt != -100, arr_gam != -100) ## Ignore bad points
			num_points_b = int(np.sum(bool_arr)) ## Number of points to correlate
			
			if int(num_points_b) > 6: ## Min number of points needed
				arr_opt = arr_opt[bool_arr]
				arr_gam = arr_gam[bool_arr]
				
				## Make array of points
				all_points = np.zeros([num_points_b, 2])
				for ij in range(0,num_points_b):
					point_opt = arr_opt[ij]
					point_gam = arr_gam[ij]
					
					all_points[ij,0] = point_opt
					all_points[ij,1] = point_gam
					
				## Calculate correlation coefficient
				corr, pval = spearmanr(all_points) 
				correlation_b[ii] = corr
			else:
				correlation_b[ii] = -1.1


		correlation_b = correlation_b[::-1]
		correlation = np.concatenate((correlation_b[:-1], correlation_f))
		
		CORRELATION[ip,it] = correlation
		print(ip,it)

filename = './cor_err/cor_err_'+str(obj_num).zfill(4)+'_'
np.savez(filename+str(out_num).zfill(3)+'.npz', CORRELATION=CORRELATION, time_of_correlation=time_of_correlation, time_shift=time_shift)

