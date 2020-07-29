import pandas as pd
import numpy as np
import os
import sys
from shutil import copyfile

from scipy.stats import halfnorm
from random import randint
from scipy.stats import spearmanr

obj_opt = int(sys.argv[1]) ## optical object number
obj_gam = int(sys.argv[2]) ## gamma object number
optic_file = str(sys.argv[3]) ## optical file
gamma_file = str(sys.argv[4]) ## gamma file
out_num = int(sys.argv[5]) ## bookkeeping number
dt = float(sys.argv[6]) ## time to average data over (0.45)
DT = float(sys.argv[7]) ##+-DT is the time bin to calc mean (10000)
num_iterations = int(sys.argv[8]) ## How many times to calculate corr_coeff
RERUN = int(sys.argv[9]) ## Rerun if 1

SCRATCH_DIR = '/fs/scratch/PCON0003/cond0064'
if obj_gam < obj_opt:
	filecheck = SCRATCH_DIR+'/cross_object_gam+gam/cross_object_gam'+str(obj_gam).zfill(4)+'_gam'+str(obj_opt).zfill(4)+'_'+str(out_num).zfill(3)+'.dat' ## Filename to save to
	filename = './cross_object_gam+gam/cross_object_gam'+str(obj_gam).zfill(4)+'_gam'+str(obj_opt).zfill(4)+'_'+str(out_num).zfill(3)+'.dat' ## Filename to save to
else:
	filecheck = SCRATCH_DIR+'/cross_object_gam+gam/cross_object_gam'+str(obj_opt).zfill(4)+'_gam'+str(obj_gam).zfill(4)+'_'+str(out_num).zfill(3)+'.dat' ## Filename to save to
	filename = './cross_object_gam+gam/cross_object_gam'+str(obj_opt).zfill(4)+'_gam'+str(obj_gam).zfill(4)+'_'+str(out_num).zfill(3)+'.dat' ## Filename to save to

if (not os.path.isfile(filecheck)) or (RERUN == 1):

	###
	###
	### Getting full time array
	###
	###

	read_in = np.loadtxt(gamma_file).T
	time = read_in[0]/(60*60*24) + 2451910.5
	time_min_gam = round(time.min())
	time_max_gam = round(time.max())
	
	read_in = np.loadtxt(optic_file).T
	time = read_in[0]/(60*60*24) + 2451910.5
	time_min_opt = round(time.min())
	time_max_opt = round(time.max())

	time_min = round(min(time_min_opt,time_min_gam))
	time_max = round(max(time_max_opt,time_max_gam))
	
	t_bin = 0.4
	t = np.arange(0,time_max-time_min,t_bin)
	tnum = len(t)

	###
	###
	### OPTICAL
	###
	###

	read_in = np.loadtxt(optic_file).T

	time = read_in[0]/(60*60*24) + 2451910.5 - time_min
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
				
	t_opt = t.copy()
	F_opt = F.copy()
	fe_opt_d = Fe_d.copy()
	fe_opt_u = Fe_u.copy()
	
	###
	###
	### GAMMA-RAY
	###
	###

	read_in = np.loadtxt(gamma_file).T

	time = read_in[0]/(60*60*24) + 2451910.5 - time_min
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
	
	
	t_gam = t.copy()
	F_gam = F.copy()
	fe_gam_d = Fe_d.copy()
	fe_gam_u = Fe_u.copy()
	
	###
	###
	### Cross correlate
	###
	###

	T = np.arange(0.,3000,t_bin) ## correlation time lag
	Tnum = len(T) ## Number correlation times
	
	num_redux = 5
	## Foreward array
	num_points_f = np.zeros(Tnum).astype(np.int64)
	correlation_f = np.zeros([num_redux, Tnum])

	## Backwords array
	num_points_b = np.zeros(Tnum).astype(np.int64)
	correlation_b = np.zeros([num_redux, Tnum])

	## Output array
	# CORRELATION = np.zeros([num_iterations+1, 2*Tnum-1])
	# T_b = T[::-1]
	# TT = np.concatenate((-T_b[:-1], T))
	# CORRELATION[0] = TT
	
	## Output array
	CORRELATION = np.zeros([2*num_iterations*num_redux+1, Tnum])
	# T_b = T[::-1]
	TT = T # np.concatenate((-T_b[:-1], T))
	CORRELATION[0] = TT

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
			
			## Gamma-ray
			if F_opt[ii] != -100:
				if randint(0,1) == 0:
					F1_opt[ii] = F_opt[ii] - halfnorm.rvs(scale=fe_opt_d[ii])
				else:
					F1_opt[ii] = F_opt[ii] + halfnorm.rvs(scale=fe_opt_u[ii])
		
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
				
				# ## Calculate correlation coefficient
				# corr, pval = spearmanr(all_points) 
				# correlation_f[ii] = corr
			# else:
				# correlation_f[ii] = -1.1

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
					
				redux_size = int(np.ceil(num_points_b[ii]**0.8))
				redux_index = np.arange(0,num_points_b[ii])
				for jj in range(0,num_redux):
					rand_points = all_points[np.random.choice(redux_index, replace=False, size=redux_size)]
					## Calculate correlation coefficient
					corr, pval = spearmanr(rand_points) 
					correlation_b[jj,ii] = corr
			else:
				correlation_b[:,ii] = -1.1*np.ones(num_redux)
					
				# ## Calculate correlation coefficient
				# corr, pval = spearmanr(all_points) 
				# correlation_b[ii] = corr
			# else:
				# correlation_b[ii] = -1.1


		# correlation_b = correlation_b[::-1]
		# correlation = np.concatenate((correlation_b[:-1], correlation_f))
		
		CORRELATION[2*ip*num_redux+1:2*ip*num_redux+num_redux+1] = correlation_f
		CORRELATION[2*ip*num_redux+num_redux+1:2*ip*num_redux+2*num_redux+1] = correlation_b
		# CORRELATION[2*ip+2] = correlation_b
	
	np.savetxt(filename, CORRELATION)

else:
	copyfile(filecheck, filename)
