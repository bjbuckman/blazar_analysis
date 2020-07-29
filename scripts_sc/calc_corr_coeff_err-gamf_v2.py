import pandas as pd
import numpy as np
import os
import sys

from scipy.stats import halfnorm
from scipy.stats import spearmanr
from random import randint

obj_num = int(sys.argv[1]) ## Object number to use
gamma_file = str(sys.argv[2]) ## Input optical file
out_num = int(sys.argv[3]) ## Output number (for bookkeeping)
dt = float(sys.argv[4]) ## time to average data over (0.45)
DT = float(sys.argv[5]) ##+-DT is the time bin to calc mean (10000)
num_iterations = int(sys.argv[6]) ## How many times to calculate corr_coeff

flare_file = str(sys.argv[7]) ## File with flares
flare_num = int(sys.argv[8]) ## flare number to take out

print(dt,DT,flare)

###
###
### Getting full time array
###
###

read_in = pd.read_csv(optic_file)
data = read_in.values.T

time = data[0]
time_min_opt = round(time.min())
time_max_opt = round(time.max())

read_in = np.loadtxt(gamma_file).T
time = read_in[0]/(60*60*24) + 2451910.5
time_min_gam = round(time.min())
time_max_gam = round(time.max())

time_min = round(min(time_min_opt,time_min_gam))
time_max = round(max(time_max_opt,time_max_gam))

t = np.arange(time_min,time_max,0.2)
tnum = len(t)

###
###
### OPTICAL
###
###

read_in = pd.read_csv(optic_file)
data = read_in.values.T

time = data[0]
flux = data[7]
flux_err = data[8]
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

time = read_in[0]/(60*60*24) + 2451910.5
flux = read_in[1]
gamma_lerr_data = read_in[2]
gamma_uerr_data = read_in[3]

### GAMMA-RAY FLARES
if flare_num > 0:
	read_in = np.loadtxt(flare_file)
	read_in = read_in.astype(int)
	flare_begin = read_in[1][flare_num-1]
	flare_end   = read_in[2][flare_num-1]
	
	## Remove flare
	bins_remove = np.arange(flare_begin,flare_end+1)
	time = np.delete(time, bins_remove)
	flux = np.delete(flux, bins_remove)
	gamma_lerr_data = np.delete(gamma_lerr_data, bins_remove)
	gamma_uerr_data = np.delete(gamma_uerr_data, bins_remove)

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

T = np.arange(0.,3000,0.4) ## correlation time lag
Tnum = len(T) ## Number correlation times

## Foreward array
num_points_f = np.zeros(Tnum).astype(np.int64)
correlation_f = np.zeros(Tnum)

## Backwords array
num_points_b = np.zeros(Tnum)
correlation_b = np.zeros(Tnum)

## Output array
CORRELATION = np.zeros([num_iterations+1, 2*Tnum-1])
T_b = T[::-1]
TT = np.concatenate((-T_b[:-1], T))
CORRELATION[0] = TT

for ip in range(0,num_iterations):
	
	## New flux arrays for calculation
	F1_gam = np.zeros(tnum)
	F1_opt = np.zeros(tnum)
	for ii in range(0,tnum):
	
		## Gamma-ray
		if F_gam[ii] != -100:
			if randint(0,1) == 0:
				F1_gam[ii] = F_gam[ii]
			else:
				F1_gam[ii] = F_gam[ii]
		else:
			F1_gam[ii] = -100
		
		## Optical
		if F_opt[ii] != -100:
			F1_opt[ii] = F_opt[ii]
		else:
			F1_opt[ii] = -100
	
	## Forword
	for ii in range(0,Tnum):
		if ii == 0:
			arr_opt = F1_opt
			arr_gam = F1_gam
		else:
			arr_opt = F1_opt[ii:]
			arr_gam = F1_gam[:-ii]

		bool_arr = np.logical_and(arr_opt != -100, arr_gam != -100) ## Ignore bad points
		num_points_f[ii] = int(np.sum(bool_arr)) ## Number of points to correlate
		
		if int(num_points_f[ii]) > 3: ## Min number of points needed
			arr_opt = arr_opt[bool_arr]
			arr_gam = arr_gam[bool_arr]
			
			## Make array of points
			all_points = np.zeros([num_points_f[ii], 2])
			for ij in range(0,num_points_f[ii]):
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
			arr_opt = F1_opt
			arr_gam = F1_gam
		else:
			arr_opt = F1_opt[:-ii]
			arr_gam = F1_gam[ii:]

		bool_arr = np.logical_and(arr_opt != -100, arr_gam != -100) ## Ignore bad points
		num_points_b[ii] = int(np.sum(bool_arr)) ## Number of points to correlate
		
		if int(num_points_b[ii]) > 3: ## Min number of points needed
			arr_opt = arr_opt[bool_arr]
			arr_gam = arr_gam[bool_arr]
			
			## Make array of points
			all_points = np.zeros([num_points_b[ii], 2])
			for ij in range(0,num_points_b[ii]):
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
	
	CORRELATION[ip+1] = correlation

filename = './cor_err-gam_flare/cor_err-gam_flare'+str(flare_num).zfill(4)+'_'
np.savetxt(filename+str(out_num).zfill(3)+'.dat', CORRELATION)

