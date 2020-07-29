import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from random import randint
from scipy.stats import halfnorm
from scipy.stats import spearmanr

asa_dir = './asas-sn/'
opt_dir = './optical/'

gam_ver = 3


INITIALIZE = 1

if INITIALIZE == 1:
	input_csv = './fermi_4FGL_associations_ext_GRPHorder.csv'

	data_in = pd.read_csv(input_csv)
	data_in.drop(columns='Unnamed: 0', inplace=True)

	## New columns

	## lum_distance
	data_in['opt_flux_median_opt-t'] = np.nan
	data_in['opt_flux_err_median_opt-t'] = np.nan
	data_in['opt_variance_opt-t'] = np.nan

	data_in['gam_flux_median_opt-t'] = np.nan
	data_in['gam_flux_errd_median_opt-t'] = np.nan
	data_in['gam_flux_erru_median_opt-t'] = np.nan
	data_in['gam_variance_opt-t'] = np.nan

	data_in['ratio_median_opt-t'] = np.nan
	data_in['ratio_errd_median_opt-t'] = np.nan
	data_in['ratio_erru_median_opt-t'] = np.nan
	data_in['ratio_variance_opt-t'] = np.nan
	
	data_in['b_median_opt-t'] = np.nan
	data_in['b_errd_median_opt-t'] = np.nan
	data_in['b_erru_median_opt-t'] = np.nan
	data_in['b_variance_opt-t'] = np.nan
	
	data_in['bl_median_opt-t'] = np.nan
	data_in['bl_errd_median_opt-t'] = np.nan
	data_in['bl_erru_median_opt-t'] = np.nan
	data_in['bl_variance_opt-t'] = np.nan
	
	data_in['bu_median_opt-t'] = np.nan
	data_in['bu_errd_median_opt-t'] = np.nan
	data_in['bu_erru_median_opt-t'] = np.nan
	data_in['bu_variance_opt-t'] = np.nan
	
	data_in['k_median_opt-t'] = np.nan
	data_in['k_errd_median_opt-t'] = np.nan
	data_in['k_erru_median_opt-t'] = np.nan
	data_in['k_variance_opt-t'] = np.nan

	data_in['corr_opt_ratio'] = np.nan
	data_in['corr_opt_ratio_std'] = np.nan

	data_in['corr_gam_ratio'] = np.nan
	data_in['corr_gam_ratio_std'] = np.nan

	data_in['corr_b_bl'] = np.nan
	data_in['corr_b_bl_std'] = np.nan
	
	data_in['corr_b_bu'] = np.nan
	data_in['corr_b_bu_std'] = np.nan
	
	data_in['corr_b_k'] = np.nan
	data_in['corr_b_k_std'] = np.nan
	
	data_in['corr_bl_bu'] = np.nan
	data_in['corr_bl_bu_std'] = np.nan

	data_in['corr_bl_k'] = np.nan
	data_in['corr_bl_k_std'] = np.nan

	data_in['corr_bu_k'] = np.nan
	data_in['corr_bu_k_std'] = np.nan

	data_in['corr_opt_b'] = np.nan
	data_in['corr_opt_b_std'] = np.nan

	data_in['corr_gam_b'] = np.nan
	data_in['corr_gam_b_std'] = np.nan

	data_in['corr_ratio_b'] = np.nan
	data_in['corr_ratio_b_std'] = np.nan

	data_in['corr_opt_bl'] = np.nan
	data_in['corr_opt_bl_std'] = np.nan

	data_in['corr_gam_bl'] = np.nan
	data_in['corr_gam_bl_std'] = np.nan

	data_in['corr_ratio_bl'] = np.nan
	data_in['corr_ratio_bl_std'] = np.nan

	data_in['corr_opt_bu'] = np.nan
	data_in['corr_opt_bu_std'] = np.nan

	data_in['corr_gam_bu'] = np.nan
	data_in['corr_gam_bu_std'] = np.nan

	data_in['corr_ratio_bu'] = np.nan
	data_in['corr_ratio_bu_std'] = np.nan

	data_in['corr_opt_k'] = np.nan
	data_in['corr_opt_k_std'] = np.nan

	data_in['corr_gam_k'] = np.nan
	data_in['corr_gam_k_std'] = np.nan

	data_in['corr_ratio_k'] = np.nan
	data_in['corr_ratio_k_std'] = np.nan
	
elif INITIALIZE == 0:
	input_csv = './fermi_4FGL_associations_ext_GRPHorder+analysis1.csv'

	data_in = pd.read_csv(input_csv)
	data_in.drop(columns='Unnamed: 0', inplace=True)



dt = 0.45
gr_bayes_spec_dir = '../2001/output/gamma-ray_bayesien_spectrum/'

## constants
h = 4.1357e-15
erg2eV = 6.24151e11
Jy2erg_cm_s_Hz = 1e-23
optical_energy = 2.4 ##eV

# opt_factor = Jy2erg_cm_s_Hz*erg2eV/h*optical_energy/1.e3/1.e6
# opt_factor = 1.

num_objects = len(data_in)
for kk in range(0,763):
	obj_num = kk
	
	optical_tag = 1
	try:
		optic_file = '../data/optical/object'+str(obj_num).zfill(4)+'_asas-sn.csv'
		read_in = pd.read_csv(optic_file)
		
		opt_time = read_in['HJD'].values #- time_min
		opt_flux = read_in['flux'].values
		opt_flux_err = read_in['flux_err'].values
		
		time = read_in['HJD'].values
		time_min_opt = round(time.min())
		time_max_opt = round(time.max())
		
	except:
		optical_tag = 0
	
	gamma_tag = 1
	try:
		gamma_file = './gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.dat'
		read_in = np.loadtxt(gamma_file).T

		gam_time = read_in[0]/(60*60*24) + 2451910.5 #- time_min
		gam_flux = read_in[1]
		gam_lerr = read_in[2]
		gam_uerr = read_in[3]
		
		time = read_in[0]/(60*60*24) + 2451910.5
		time_min_gam = round(time.min())
		time_max_gam = round(time.max())
		
	except:
		gamma_tag = 0
		
	
	if optical_tag == 1 and gamma_tag == 1:
		time_min = round(min(time_min_opt,time_min_gam))
		time_max = round(max(time_max_opt,time_max_gam))
		t = np.arange(time_min,time_max,0.4) #- time_min
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
		# Assuming independence of measurements,
		# We just avg mu values and inv-sum sigma
		f = np.zeros(tnum)-100
		fe = np.zeros(tnum)-1

		dtf = 0.45
		for ii in range(0,tnum):
			t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt )
			t_arr = np.logical_and( t_arr, flux!=99.99)
			t_arrf = np.logical_and( time<=t[ii]+dtf, time>=t[ii]-dtf ) 
			t_arrf = np.logical_and( t_arrf, flux!=99.99)
			if np.any(t_arr) and np.any(t_arrf):
				f[ii] = np.mean(flux[t_arr])
				fe[ii] = (np.sum(flux_err_2[t_arr])/np.sum(t_arr))**0.5

		t_arr_d = flux!=99.99
		t_opt_d = time[t_arr_d]
		F_opt_d = flux[t_arr_d]
		fe_opt_d = flux_err[t_arr_d]
				
		t_opt = t
		F_opt = f
		fe_opt = fe

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
		# Assuming independence of measurements,
		# We just avg mu values and inv-sum sigma
		f = np.zeros(tnum)-100
		fe_d = np.zeros(tnum)-1
		fe_u = np.zeros(tnum)-1

		dtf = 0.45
		for ii in range(0,tnum):
			t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt)
			t_arr = np.logical_and( t_arr, flux>0)
			t_arrf = np.logical_and( time<=t[ii]+dtf, time>=t[ii]-dtf ) 
			t_arrf = np.logical_and( t_arrf, flux>0)
			if np.any(t_arr) and np.any(t_arrf):
				f[ii] = np.mean(flux[t_arr])
				fe_d[ii] = (np.sum(flux_err_2d[t_arr])/np.sum(t_arr))**0.5
				fe_u[ii] = (np.sum(flux_err_2u[t_arr])/np.sum(t_arr))**0.5	
			
		t_gam_d = time
		F_gam_d = flux
		fe_gam_dd = (gamma_lerr_data)
		fe_gam_ud = (gamma_uerr_data)

		t_gam = t
		F_gam = f
		fe_d_gam = fe_d
		fe_u_gam = fe_u
		
		###
		###
		### RATIO
		###
		###
		bool_arr = np.logical_and(F_gam != -100, F_opt != -100)
		
		h = 4.1357e-15
		erg2eV = 6.24151e11
		Jy2erg_cm_s_Hz = 1e-23
		optical_energy = 2.4 ##eV
		
		opt_factor = Jy2erg_cm_s_Hz*erg2eV/h*optical_energy/1.e3/1.e6
		# opt_factor = 1.
		F_opt_ratio = opt_factor*F_opt
		fe_opt_ratio = opt_factor*fe_opt
		
		F_opt_d_ratio = opt_factor*F_opt_d
		fe_opt_d_ratio = opt_factor*fe_opt_d
		
		t_ratio = t[bool_arr]
		TNUM = int(np.sum(bool_arr))
		# print(TNUM)
		t_opt_ratio = t_opt[bool_arr]
		t_gam_ratio = t_gam[bool_arr]
		F_opt_ratio = F_opt_ratio[bool_arr]
		F_gam_ratio = F_gam[bool_arr]
		fe_opt_ratio = fe_opt_ratio[bool_arr]
		fe_d_gam_ratio = fe_d_gam[bool_arr]
		fe_u_gam_ratio = fe_u_gam[bool_arr]
		
		gam_opt_ratio = F_opt_ratio/F_gam_ratio
		ratio_err_u = ((fe_opt_ratio/F_gam_ratio)**2 + (F_opt_ratio/F_gam_ratio**2*fe_u_gam_ratio)**2)**0.5
		ratio_err_d = ((fe_opt_ratio/F_gam_ratio)**2 + (F_opt_ratio/F_gam_ratio**2*fe_d_gam_ratio)**2)**0.5
		
		
		opt_flux_median = np.median(F_opt_ratio)
		opt_flux_err_median = np.median(fe_opt_ratio)
		
		gam_flux_median = np.median(F_gam_ratio)
		gam_flux_errd_median = np.median(fe_d_gam_ratio)
		gam_flux_erru_median = np.median(fe_u_gam_ratio)
		
		ratio_median = np.median(gam_opt_ratio)
		ratio_errd_median = np.median(ratio_err_u)
		ratio_erru_median = np.median(ratio_err_d)
		
		data_in['opt_flux_median_opt-t'].loc[kk] = opt_flux_median/opt_factor
		data_in['opt_flux_err_median_opt-t'].loc[kk] = opt_flux_err_median/opt_factor
		
		data_in['gam_flux_median_opt-t'].loc[kk] = gam_flux_median
		data_in['gam_flux_errd_median_opt-t'].loc[kk] = gam_flux_errd_median
		data_in['gam_flux_erru_median_opt-t'].loc[kk] = gam_flux_erru_median
		
		data_in['ratio_median_opt-t'].loc[kk] = ratio_median
		data_in['ratio_errd_median_opt-t'].loc[kk] = ratio_errd_median
		data_in['ratio_erru_median_opt-t'].loc[kk] = ratio_erru_median
		
		## CALC CORRELATION COEFFICIENTS
		num_iterations = 100
		corr_opt_ratio = np.zeros(num_iterations)
		corr_gam_ratio = np.zeros(num_iterations)
		
		F_GAM = np.zeros([num_iterations, TNUM])
		F_OPT = np.zeros([num_iterations, TNUM])
		F_RAT = np.zeros([num_iterations, TNUM])
		
		for jj in range(0,num_iterations):
		
			F_gam = np.zeros(TNUM)
			F_opt = np.zeros(TNUM)
			ratio_arr = np.zeros(TNUM)
			rand_points_1 = np.zeros([TNUM,2])
			rand_points_2 = np.zeros([TNUM,2])
			for ii in range(0,TNUM):
				## Gamma-ray
				if randint(0,1) == 0:
					F_gam[ii] = F_gam_ratio[ii] - halfnorm.rvs(scale=fe_d_gam_ratio[ii])
				else:
					F_gam[ii] = F_gam_ratio[ii] + halfnorm.rvs(scale=fe_u_gam_ratio[ii])
				
				## Optical
				F_opt[ii] = np.random.normal(F_opt_ratio[ii], fe_opt_ratio[ii])
				
				## ratio
				if randint(0,1) == 0:
					ratio_arr[ii] = gam_opt_ratio[ii] - halfnorm.rvs(scale=ratio_err_d[ii])
				else:
					ratio_arr[ii] = gam_opt_ratio[ii] + halfnorm.rvs(scale=ratio_err_u[ii])
			
			F_GAM[jj,:] = F_gam
			F_OPT[jj,:] = F_opt
			F_RAT[jj,:] = ratio_arr
			
			rand_points_1[:,0] = ratio_arr
			rand_points_1[:,1] = F_opt
			
			rand_points_2[:,0] = ratio_arr
			rand_points_2[:,1] = F_gam
			
			corr_1, pval_1 = spearmanr(rand_points_1)
			corr_opt_ratio[jj] = corr_1
			
			corr_2, pval_2 = spearmanr(rand_points_2)
			corr_gam_ratio[jj] = corr_2
			
		corr_opt_ratio_mean = np.mean(corr_opt_ratio)
		corr_opt_ratio_std = np.std(corr_opt_ratio)
		
		corr_gam_ratio_mean = np.mean(corr_gam_ratio)
		corr_gam_ratio_std = np.std(corr_gam_ratio)
		
		data_in['corr_opt_ratio'].loc[kk] = corr_opt_ratio_mean
		data_in['corr_opt_ratio_std'].loc[kk] = corr_opt_ratio_std
		
		data_in['corr_gam_ratio'].loc[kk] = corr_gam_ratio_mean
		data_in['corr_gam_ratio_std'].loc[kk] = corr_gam_ratio_std
		
		## VARIANCES
		gam_variance = np.var(F_GAM)
		opt_variance = np.var(F_OPT/opt_factor)
		ratio_variance = np.var(F_RAT)
		
		data_in['gam_variance_opt-t'].loc[kk] = gam_variance
		data_in['opt_variance_opt-t'].loc[kk] = opt_variance
		data_in['ratio_variance_opt-t'].loc[kk] = ratio_variance
		
	
	spectral_tag = 1
	try:
		spectral_file = gr_bayes_spec_dir+'object'+str(obj_num).zfill(4)+'_bayesian_spectrum.dat'
		read_in = np.loadtxt(spectral_file).T
		
		t_beta = read_in[0]/(60*60*24) + 2451910.5
		
		beta = read_in[1]
		dbeta_l = read_in[2]
		dbeta_u = read_in[3]
		
		lbeta = read_in[4]
		ldbeta_l = read_in[5]
		ldbeta_u = read_in[6]
		
		ubeta = read_in[7]
		udbeta_l = read_in[8]
		udbeta_u = read_in[9]
		
		beta_prime = read_in[10]+2.
		dbeta_prime_l = read_in[11]
		dbeta_prime_u = read_in[12]
		
		
		if optical_tag == 1 and gamma_tag == 1:
			##
			## Regridding over dt days
			## Doing this for simplicity
			# Assuming independence of measurements,
			# We just avg mu values and inv-sum sigma
			
			b = np.zeros(tnum)-100
			b_d = np.zeros(tnum)-1
			b_u = np.zeros(tnum)-1
			
			bl = np.zeros(tnum)-100
			bl_d = np.zeros(tnum)-1
			bl_u = np.zeros(tnum)-1
			
			bu = np.zeros(tnum)-100
			bu_d = np.zeros(tnum)-1
			bu_u = np.zeros(tnum)-1
			
			k = np.zeros(tnum)-100
			k_d = np.zeros(tnum)-1
			k_u = np.zeros(tnum)-1

			dtf = 0.45
			for ii in range(0,tnum):
				t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt)
				t_arrf = np.logical_and( time<=t[ii]+dtf, time>=t[ii]-dtf ) 
				# t_arrf = np.logical_and( t_arrf, )
				if np.any(t_arr) and np.any(t_arrf):
					b[ii] = np.mean(beta[t_arr])
					b_d[ii] = (np.sum(dbeta_l[t_arr]**2)/np.sum(t_arr))**0.5
					b_u[ii] = (np.sum(dbeta_u[t_arr]**2)/np.sum(t_arr))**0.5	
				
					bl[ii] = np.mean(lbeta[t_arr])
					bl_d[ii] = (np.sum(ldbeta_l[t_arr]**2)/np.sum(t_arr))**0.5
					bl_u[ii] = (np.sum(ldbeta_u[t_arr]**2)/np.sum(t_arr))**0.5	
					
					bu[ii] = np.mean(ubeta[t_arr])
					bu_d[ii] = (np.sum(udbeta_l[t_arr]**2)/np.sum(t_arr))**0.5
					bu_u[ii] = (np.sum(udbeta_u[t_arr]**2)/np.sum(t_arr))**0.5	
					
					k[ii] = np.mean(beta_prime[t_arr])
					k_d[ii] = (np.sum(dbeta_prime_l[t_arr]**2)/np.sum(t_arr))**0.5
					k_u[ii] = (np.sum(dbeta_prime_u[t_arr]**2)/np.sum(t_arr))**0.5	
			
			b = b[bool_arr]
			b_d = b_d[bool_arr]
			b_u = b_u[bool_arr]
			
			bl = bl[bool_arr]
			bl_d = bl_d[bool_arr]
			bl_u = bl_u[bool_arr]
			
			bu = bu[bool_arr]
			bu_d = bu_d[bool_arr]
			bu_u = bu_u[bool_arr]
			
			k = k[bool_arr]
			k_d = k_d[bool_arr]
			k_u = k_u[bool_arr]
			
			data_in['b_errd_median_opt-t'].loc[kk] = np.median(b_d)
			data_in['b_erru_median_opt-t'].loc[kk] = np.median(b_u)
			
			data_in['bl_errd_median_opt-t'].loc[kk] = np.median(bl_d)
			data_in['bl_erru_median_opt-t'].loc[kk] = np.median(bl_u)
			
			data_in['bu_errd_median_opt-t'].loc[kk] = np.median(bu_d)
			data_in['bu_erru_median_opt-t'].loc[kk] = np.median(bu_u)
			
			data_in['k_errd_median_opt-t'].loc[kk] = np.median(k_d)
			data_in['k_erru_median_opt-t'].loc[kk] = np.median(k_u)
			
			## calculate correlation coefficients
			corr_b_bl = np.zeros(num_iterations)
			corr_b_bu = np.zeros(num_iterations)
			corr_b_k = np.zeros(num_iterations)
			
			corr_bl_bu = np.zeros(num_iterations)
			corr_bl_k = np.zeros(num_iterations)
			corr_bu_k = np.zeros(num_iterations)
			
			corr_opt_b = np.zeros(num_iterations)
			corr_gam_b = np.zeros(num_iterations)
			corr_ratio_b = np.zeros(num_iterations)
			
			corr_opt_bl = np.zeros(num_iterations)
			corr_gam_bl = np.zeros(num_iterations)
			corr_ratio_bl = np.zeros(num_iterations)
			
			corr_opt_bu = np.zeros(num_iterations)
			corr_gam_bu = np.zeros(num_iterations)
			corr_ratio_bu = np.zeros(num_iterations)
			
			corr_opt_k = np.zeros(num_iterations)
			corr_gam_k = np.zeros(num_iterations)
			corr_ratio_k = np.zeros(num_iterations)
			
			F_B = np.zeros([num_iterations, TNUM])
			F_BL = np.zeros([num_iterations, TNUM])
			F_BU = np.zeros([num_iterations, TNUM])
			F_K = np.zeros([num_iterations, TNUM])
			
			for jj in range(0,num_iterations):
			
				F_b = np.zeros(TNUM)
				F_bl = np.zeros(TNUM)
				F_bu = np.zeros(TNUM)
				F_k = np.zeros(TNUM)
				F_gam = np.zeros(TNUM)
				F_opt = np.zeros(TNUM)
				ratio_arr = np.zeros(TNUM)
				
				for ii in range(0,TNUM):
					## b
					if randint(0,1) == 0:
						F_b[ii] = b[ii] - halfnorm.rvs(scale=b_d[ii])
					else:
						F_b[ii] = b[ii] + halfnorm.rvs(scale=b_u[ii])
					
					## bl
					if randint(0,1) == 0:
						F_bl[ii] = bl[ii] - halfnorm.rvs(scale=bl_d[ii])
					else:
						F_bl[ii] = bl[ii] + halfnorm.rvs(scale=bl_u[ii])
					
					## bu
					if randint(0,1) == 0:
						F_bu[ii] = bu[ii] - halfnorm.rvs(scale=bu_d[ii])
					else:
						F_bu[ii] = bu[ii] + halfnorm.rvs(scale=bu_u[ii])
						
					## k
					if randint(0,1) == 0:
						F_k[ii] = k[ii] - halfnorm.rvs(scale=k_d[ii])
					else:
						F_k[ii] = k[ii] + halfnorm.rvs(scale=k_u[ii])
					
					
					F_gam[ii] = F_GAM[jj,ii]
					F_opt[ii] = F_OPT[jj,ii]
					ratio_arr[ii] = F_RAT[jj,ii]
					
					# ## Gamma-ray
					# if randint(0,1) == 0:
						# F_gam[ii] = F_gam_ratio[ii] - halfnorm.rvs(scale=fe_d_gam_ratio[ii])
					# else:
						# F_gam[ii] = F_gam_ratio[ii] + halfnorm.rvs(scale=fe_u_gam_ratio[ii])
					
					# ## Optical
					# F_opt[ii] = np.random.normal(F_opt_ratio[ii], fe_opt_ratio[ii])
					
					# ## ratio
					# if randint(0,1) == 0:
						# ratio_arr[ii] = gam_opt_ratio[ii] - halfnorm.rvs(scale=ratio_err_d[ii])
					# else:
						# ratio_arr[ii] = gam_opt_ratio[ii] + halfnorm.rvs(scale=ratio_err_u[ii])
				
				F_B[jj,:] = F_b
				F_BL[jj,:] = F_bl
				F_BU[jj,:] = F_bu
				F_K[jj,:] = F_k
				
				rand_points_00 = np.zeros([TNUM,2])
				rand_points_01 = np.zeros([TNUM,2])
				rand_points_02 = np.zeros([TNUM,2])
				rand_points_1 = np.zeros([TNUM,2])
				rand_points_2 = np.zeros([TNUM,2])
				rand_points_3 = np.zeros([TNUM,2])
				
				rand_points_00[:,0] = F_b
				rand_points_00[:,1] = F_bl
				
				rand_points_01[:,0] = F_b
				rand_points_01[:,1] = F_bu
				
				rand_points_02[:,0] = F_b
				rand_points_02[:,1] = F_k
				
				rand_points_1[:,0] = F_bl
				rand_points_1[:,1] = F_bu
				
				rand_points_2[:,0] = F_bl
				rand_points_2[:,1] = F_k
				
				rand_points_3[:,0] = F_bu
				rand_points_3[:,1] = F_k
				
				corr_00, pval_00 = spearmanr(rand_points_00)
				corr_b_bl[jj] = corr_00
				
				corr_01, pval_01 = spearmanr(rand_points_01)
				corr_b_bu[jj] = corr_01
				
				corr_02, pval_02 = spearmanr(rand_points_02)
				corr_b_k[jj] = corr_02
				
				corr_1, pval_1 = spearmanr(rand_points_1)
				corr_bl_bu[jj] = corr_1
				
				corr_2, pval_2 = spearmanr(rand_points_2)
				corr_bl_k[jj] = corr_2
				
				corr_3, pval_3 = spearmanr(rand_points_3)
				corr_bu_k[jj] = corr_3
				
				## more correlations
				rand_points_1 = np.zeros([TNUM,2])
				rand_points_2 = np.zeros([TNUM,2])
				rand_points_3 = np.zeros([TNUM,2])
				
				rand_points_1[:,0] = F_opt
				rand_points_1[:,1] = F_b
				
				rand_points_2[:,0] = F_gam
				rand_points_2[:,1] = F_b
				
				rand_points_3[:,0] = ratio_arr
				rand_points_3[:,1] = F_b
				
				corr_1, pval_1 = spearmanr(rand_points_1)
				corr_opt_b[jj] = corr_1
				
				corr_2, pval_2 = spearmanr(rand_points_2)
				corr_gam_b[jj] = corr_2
				
				corr_3, pval_3 = spearmanr(rand_points_3)
				corr_ratio_b[jj] = corr_3
				
				## more correlations
				rand_points_1 = np.zeros([TNUM,2])
				rand_points_2 = np.zeros([TNUM,2])
				rand_points_3 = np.zeros([TNUM,2])
				
				rand_points_1[:,0] = F_opt
				rand_points_1[:,1] = F_bl
				
				rand_points_2[:,0] = F_gam
				rand_points_2[:,1] = F_bl
				
				rand_points_3[:,0] = ratio_arr
				rand_points_3[:,1] = F_bl
				
				corr_1, pval_1 = spearmanr(rand_points_1)
				corr_opt_bl[jj] = corr_1
				
				corr_2, pval_2 = spearmanr(rand_points_2)
				corr_gam_bl[jj] = corr_2
				
				corr_3, pval_3 = spearmanr(rand_points_3)
				corr_ratio_bl[jj] = corr_3
				
				## more correlations
				rand_points_1 = np.zeros([TNUM,2])
				rand_points_2 = np.zeros([TNUM,2])
				rand_points_3 = np.zeros([TNUM,2])
				
				rand_points_1[:,0] = F_opt
				rand_points_1[:,1] = F_bu
				
				rand_points_2[:,0] = F_gam
				rand_points_2[:,1] = F_bu
				
				rand_points_3[:,0] = ratio_arr
				rand_points_3[:,1] = F_bu
				
				corr_1, pval_1 = spearmanr(rand_points_1)
				corr_opt_bu[jj] = corr_1
				
				corr_2, pval_2 = spearmanr(rand_points_2)
				corr_gam_bu[jj] = corr_2
				
				corr_3, pval_3 = spearmanr(rand_points_3)
				corr_ratio_bu[jj] = corr_3
				
				## more correlations
				rand_points_1 = np.zeros([TNUM,2])
				rand_points_2 = np.zeros([TNUM,2])
				rand_points_3 = np.zeros([TNUM,2])
				
				rand_points_1[:,0] = F_opt
				rand_points_1[:,1] = F_k
				
				rand_points_2[:,0] = F_gam
				rand_points_2[:,1] = F_k
				
				rand_points_3[:,0] = ratio_arr
				rand_points_3[:,1] = F_k
				
				corr_1, pval_1 = spearmanr(rand_points_1)
				corr_opt_k[jj] = corr_1
				
				corr_2, pval_2 = spearmanr(rand_points_2)
				corr_gam_k[jj] = corr_2
				
				corr_3, pval_3 = spearmanr(rand_points_3)
				corr_ratio_k[jj] = corr_3
			
			##
			b_median = np.median(b)
			b_variance = np.var(F_B)
			
			bl_median = np.median(bl)
			bl_variance = np.var(F_BL)
			
			bu_median = np.median(bu)
			bu_variance = np.var(F_BU)
			
			k_median = np.median(k)
			k_variance = np.var(F_K)
			
			data_in['b_median_opt-t'].loc[kk] = b_median
			data_in['b_variance_opt-t'].loc[kk] = b_variance
			
			data_in['bl_median_opt-t'].loc[kk] = bl_median
			data_in['bl_variance_opt-t'].loc[kk] = bl_variance
			
			data_in['bu_median_opt-t'].loc[kk] = bu_median
			data_in['bu_variance_opt-t'].loc[kk] = bu_variance
			
			data_in['k_median_opt-t'].loc[kk] = k_median
			data_in['k_variance_opt-t'].loc[kk] = k_variance
			
			##
			corr_b_bl_mean = np.mean(corr_b_bl)
			corr_b_bl_std = np.std(corr_b_bl)
			
			corr_b_bu_mean = np.mean(corr_b_bu)
			corr_b_bu_std = np.std(corr_b_bu)
			
			corr_b_k_mean = np.mean(corr_b_k)
			corr_b_k_std = np.std(corr_b_k)
			
			data_in['corr_b_bl'].loc[kk] = corr_b_bl_mean
			data_in['corr_b_bl_std'].loc[kk] = corr_b_bl_std
			
			data_in['corr_b_bu'].loc[kk] = corr_b_bu_mean
			data_in['corr_b_bu_std'].loc[kk] = corr_b_bu_std
			
			data_in['corr_b_k'].loc[kk] = corr_b_k_mean
			data_in['corr_b_k_std'].loc[kk] = corr_b_k_std
			
			##
			corr_bl_bu_mean = np.mean(corr_bl_bu)
			corr_bl_bu_std = np.std(corr_bl_bu)
			
			corr_bl_k_mean = np.mean(corr_bl_k)
			corr_bl_k_std = np.std(corr_bl_k)
			
			corr_bu_k_mean = np.mean(corr_bu_k)
			corr_bu_k_std = np.std(corr_bu_k)
			
			data_in['corr_bl_bu'].loc[kk] = corr_bl_bu_mean
			data_in['corr_bl_bu_std'].loc[kk] = corr_bl_bu_std
			
			data_in['corr_bl_k'].loc[kk] = corr_bl_k_mean
			data_in['corr_bl_k_std'].loc[kk] = corr_bl_k_std
			
			data_in['corr_bu_k'].loc[kk] = corr_bu_k_mean
			data_in['corr_bu_k_std'].loc[kk] = corr_bu_k_std
			
			##
			corr_opt_b_mean = np.mean(corr_opt_b)
			corr_opt_b_std = np.std(corr_opt_b)
			
			corr_gam_b_mean = np.mean(corr_gam_b)
			corr_gam_b_std = np.std(corr_gam_b)
			
			corr_ratio_b_mean = np.mean(corr_ratio_b)
			corr_ratio_b_std = np.std(corr_ratio_b)
			
			data_in['corr_opt_b'].loc[kk] = corr_opt_b_mean
			data_in['corr_opt_b_std'].loc[kk] = corr_opt_b_std
			
			data_in['corr_gam_b'].loc[kk] = corr_gam_b_mean
			data_in['corr_gam_b_std'].loc[kk] = corr_gam_b_std
			
			data_in['corr_ratio_b'].loc[kk] = corr_ratio_b_mean
			data_in['corr_ratio_b_std'].loc[kk] = corr_ratio_b_std
			
			##
			corr_opt_bl_mean = np.mean(corr_opt_bl)
			corr_opt_bl_std = np.std(corr_opt_bl)
			
			corr_gam_bl_mean = np.mean(corr_gam_bl)
			corr_gam_bl_std = np.std(corr_gam_bl)
			
			corr_ratio_bl_mean = np.mean(corr_ratio_bl)
			corr_ratio_bl_std = np.std(corr_ratio_bl)
			
			data_in['corr_opt_bl'].loc[kk] = corr_opt_bl_mean
			data_in['corr_opt_bl_std'].loc[kk] = corr_opt_bl_std
			
			data_in['corr_gam_bl'].loc[kk] = corr_gam_bl_mean
			data_in['corr_gam_bl_std'].loc[kk] = corr_gam_bl_std
			
			data_in['corr_ratio_bl'].loc[kk] = corr_ratio_bl_mean
			data_in['corr_ratio_bl_std'].loc[kk] = corr_ratio_bl_std
			
			##
			corr_opt_bu_mean = np.mean(corr_opt_bu)
			corr_opt_bu_std = np.std(corr_opt_bu)
			
			corr_gam_bu_mean = np.mean(corr_gam_bu)
			corr_gam_bu_std = np.std(corr_gam_bu)
			
			corr_ratio_bu_mean = np.mean(corr_ratio_bu)
			corr_ratio_bu_std = np.std(corr_ratio_bu)
			
			data_in['corr_opt_bu'].loc[kk] = corr_opt_bu_mean
			data_in['corr_opt_bu_std'].loc[kk] = corr_opt_bu_std
			
			data_in['corr_gam_bu'].loc[kk] = corr_gam_bu_mean
			data_in['corr_gam_bu_std'].loc[kk] = corr_gam_bu_std
			
			data_in['corr_ratio_bu'].loc[kk] = corr_ratio_bu_mean
			data_in['corr_ratio_bu_std'].loc[kk] = corr_ratio_bu_std
			
			##
			corr_opt_k_mean = np.mean(corr_opt_k)
			corr_opt_k_std = np.std(corr_opt_k)
			
			corr_gam_k_mean = np.mean(corr_gam_k)
			corr_gam_k_std = np.std(corr_gam_k)
			
			corr_ratio_k_mean = np.mean(corr_ratio_k)
			corr_ratio_k_std = np.std(corr_ratio_k)
			
			data_in['corr_opt_k'].loc[kk] = corr_opt_k_mean
			data_in['corr_opt_k_std'].loc[kk] = corr_opt_k_std
			
			data_in['corr_gam_k'].loc[kk] = corr_gam_k_mean
			data_in['corr_gam_k_std'].loc[kk] = corr_gam_k_std
			
			data_in['corr_ratio_k'].loc[kk] = corr_ratio_k_mean
			data_in['corr_ratio_k_std'].loc[kk] = corr_ratio_k_std
			
	except:
		spectral_tag = 0
		print('No bayesian spectral file for '+str(obj_num))
		
	print('finished '+str(kk).zfill(4))
		
	
data_in.to_csv('fermi_4FGL_associations_ext_GRPHorder+analysis1.csv')