import pandas as pd
import numpy as np
import os
import sys

from scipy.stats import halfnorm
from random import randint

obj_num = int(sys.argv[1])

#input files
optic_file = str(sys.argv[2])
gamma_file = str(sys.argv[3])

out_num = int(sys.argv[4])

#time to average data over
# dt = 0.45
dt = float(sys.argv[5])

#+-DT is the time bin to calc mean
# DT = 60.
DT = float(sys.argv[6])

#How many times to calculate corr_coeff
# num_iterations = 100

flare_file = str(sys.argv[7])

flare_num = int(sys.argv[8])

print(dt,DT,flare_num)

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
time = (read_in[0]-239643817.0)/(60*60*24) + 2454684.15527778
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
# Assuming independence of measurements,
# We just avg mu values and inv-sum sigma
f = np.zeros(tnum)-100
fe = np.zeros(tnum)-1

# dt = 0.45
for ii in range(0,tnum):
	t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt)
	t_arr = np.logical_and( t_arr, flux!=99.99)
	if np.any(t_arr):
		f[ii] = np.mean(flux[t_arr])
		fe[ii] = (np.sum(flux_err_2[t_arr])/np.sum(t_arr))**0.5

##
## We need to take care of AVERAGE mean
## In this case, I will just take a moving average to smooth data
# We also need to clean data
# I get rid of fluxes above 7.75
F = np.zeros(tnum)-100
Fe = np.zeros(tnum)-10

# DT = 30
for ii in range(0,tnum):
	t_arr = np.logical_and( t<=t[ii]+DT, t>=t[ii]-DT)
	t_arr = np.logical_and( t_arr, f!=-100)
	# t_arr = np.logical_and( t_arr, f<=7.75)
	if np.any(t_arr):
		f_mean = np.mean(f[t_arr])
		f_median = np.median(f[t_arr])
		if f[ii] != -100:
			F[ii] = f[ii]/f_median
			Fe[ii] = fe[ii]/f_median
		
t_opt = t
F_opt = F
fe_opt = Fe


if flare_num > 0:
	###
	###
	### GAMMA-RAY FLARES
	###
	###

	read_in = np.loadtxt(flare_file)
	read_in = read_in.astype(int)
	flare_begin = read_in[1][flare_num-1]
	flare_end   = read_in[2][flare_num-1]


###
###
### GAMMA-RAY
###
###

read_in = np.loadtxt(gamma_file).T

time = (read_in[0]-239643817.0)/(60*60*24) + 2454684.15527778
flux = read_in[1]
gamma_lerr_data = read_in[2]
gamma_uerr_data = read_in[3]

if flare_num > 0:
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
# Assuming independence of measurements,
# We just avg mu values and inv-sum sigma
f = np.zeros(tnum)-100
fe_d = np.zeros(tnum)-1
fe_u = np.zeros(tnum)-1

# dt = 0.45
for ii in range(0,tnum):
	t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt)
	t_arr = np.logical_and( t_arr, flux>0)
	if np.any(t_arr):
		f[ii] = np.mean(flux[t_arr])
		fe_d[ii] = (np.sum(flux_err_2d[t_arr])/np.sum(t_arr))**0.5
		fe_u[ii] = (np.sum(flux_err_2u[t_arr])/np.sum(t_arr))**0.5

##
## We need to take care of AVERAGE mean
## In this case, I will just take a moving average to smooth data
# We also need to clean data
# I get rid of fluxes above 7.75
F = np.zeros(tnum)-100
Fe_d = np.zeros(tnum)-1
Fe_u = np.zeros(tnum)-1

# DT = 30
for ii in range(0,tnum):
	t_arr = np.logical_and( t<=t[ii]+DT, t>=t[ii]-DT)
	t_arr = np.logical_and( t_arr, f!=-100)
	if np.any(t_arr):
		f_mean = np.mean(f[t_arr])
		f_median = np.median(f[t_arr])
		if f[ii] != -100:
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

# Correlation time
t_min = round(max(t_opt.min(), t_gam.min()))
t_max = round(min(t_opt.max(), t_gam.max()))
# max_t_diff = round( (t_max-t_min)/30. )

# T = np.arange(0.,max_t_diff,0.2)
T = np.arange(0.,300,0.2)
Tnum = len(T)

num_points_f = np.zeros(Tnum).astype(np.int64)
correlation_f = np.zeros(Tnum)

# num_iterations = 101
CORRELATION = np.zeros([1+1, 2*Tnum-1])

T_b = T[::-1]
TT = np.concatenate((-T_b[:-1], T))

CORRELATION[0]=TT

#Forword
for ii in range(0,Tnum):
	# print('TIMEDIFF+'+str(ii))
	if ii == 0:
		arr_opt = F_opt
		arr_opt_err = fe_opt
		arr_gam = F_gam
		arr_gam_err_d = fe_gam_d
		arr_gam_err_u = fe_gam_u
	else:
		arr_opt = F_opt[ii:]
		arr_opt_err = fe_opt[ii:]
		arr_gam = F_gam[:-ii]
		arr_gam_err_d = fe_gam_d[:-ii]
		arr_gam_err_u = fe_gam_u[:-ii]
	bool_arr = np.logical_and(arr_opt != -100, arr_gam != -100)
	# bool_arr = np.logical_and(bool_arr, arr_opt>0)
	# bool_arr = np.logical_and(bool_arr, arr_gam>0)
	num_points_f[ii] = int(np.sum(bool_arr))
	
	if int(num_points_f[ii]) > 3:
		arr_opt = arr_opt[bool_arr]
		arr_opt_err = arr_opt_err[bool_arr]
		arr_gam = arr_gam[bool_arr]
		arr_gam_err_d = arr_gam_err_d[bool_arr]
		arr_gam_err_u = arr_gam_err_u[bool_arr]
		
		for ij in range(0,num_points_f[ii]):
			#Optical point
			point_opt = arr_opt[ij]
			#Gamma ray point
			point_gam = arr_gam[ij]
			
			if ij == 0:
				all_points = np.array([[point_opt, point_gam]])
			else:
				all_points = np.append(all_points, np.array([[point_opt, point_gam]]), axis=0)
			
		# print(all_points.shape)
		corr = np.corrcoef(all_points.T)
		correlation_f[ii] = corr[0,1]
	else:
		correlation_f[ii] = -1.1

num_points_b = np.zeros(Tnum)
correlation_b = np.zeros(Tnum)

#Backword
for ii in range(0,Tnum):
	# print('TIMEDIFF-'+str(ii))
	if ii == 0:
		arr_opt = F_opt
		arr_opt_err = fe_opt
		arr_gam = F_gam
		arr_gam_err_d = fe_gam_d
		arr_gam_err_u = fe_gam_u
	else:
		arr_opt = F_opt[:-ii]
		arr_opt_err = fe_opt[:-ii]
		arr_gam = F_gam[ii:]
		arr_gam_err_d = fe_gam_d[ii:]
		arr_gam_err_u = fe_gam_u[ii:]
	bool_arr = np.logical_and(arr_opt != -100, arr_gam != -100)
	# bool_arr = np.logical_and(bool_arr, arr_opt>0)
	# bool_arr = np.logical_and(bool_arr, arr_gam>0)
	num_points_f[ii] = int(np.sum(bool_arr))
	
	if int(num_points_f[ii]) > 3:
		arr_opt = arr_opt[bool_arr]
		arr_opt_err = arr_opt_err[bool_arr]
		arr_gam = arr_gam[bool_arr]
		arr_gam_err_d = arr_gam_err_d[bool_arr]
		arr_gam_err_u = arr_gam_err_u[bool_arr]
		
		for ij in range(0,num_points_f[ii]):
			#Optical point
			point_opt = arr_opt[ij]
			#Gamma ray point
			point_gam = arr_gam[ij]
			
			if ij == 0:
				all_points = np.array([[point_opt, point_gam]])
			else:
				all_points = np.append(all_points, np.array([[point_opt, point_gam]]), axis=0)
			
		corr = np.corrcoef(all_points.T)
		correlation_b[ii] = corr[0,1]
	else:
		correlation_b[ii] = -1.1


correlation = correlation_b[::-1]
correlation = np.concatenate((correlation[:-1], correlation_f))

CORRELATION[1] = correlation

filename = './cor_err-gam_flare/cor_err-gam_flare'+str(flare_num).zfill(4)+'_'
np.savetxt(filename+str(out_num).zfill(3)+'.dat', CORRELATION)

# for n in range(0,1000):
	# if os.path.isfile(filename+str(out_num).zfill(3)+'.dat'):
		# continue
	# else:
		# np.savetxt(filename+str(n).zfill(3)+'.dat', CORRELATION)
		# break
