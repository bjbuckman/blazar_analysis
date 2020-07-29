import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec
from matplotlib import transforms
# from scipy import fft

# obj_num = 11
# ZOOM = 0

gam_ver = 3
# object_arr = np.array([0,1,2])
# object_arr = range(26,27)
object_arr = range(0,765)

autocorr_opt_err_dir = '../2001/output/auto_opt/'
autocorr_opt_sig_dir = '../2001/output/autocorr_opt+opt/'
opt_file_num = 0

autocorr_gam_err_dir = '../2001/output/auto_gam/'
autocorr_gam_sig_dir = '../2001/output/autocorr_gam+gam/'
gam_file_num = 0

cc_dir = '../2001/output/cross_corr/'
cross_object_cc_dir = '../2001/output/cross_object_gam+opt/'
cc_file_num = 0

gr_bayes_spec_dir = '../2001/output/gamma-ray_bayesien_spectrum/'

cc_bayesian_block_dir = '../2001/output/bayesian_block_cc/'

dt = 0.45


for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]	
	
	# CHANGE_POINTS = 0
	# if CHANGE_POINTS == 1:
		# cc_file = '../data/gamma-ray/object_'+str(obj_num).zfill(4)+'_change_points_gam_002.dat'
		# cc_in_gam = np.loadtxt(cc_file)
		# cc_file = '../data/optical/object_'+str(obj_num).zfill(4)+'_change_points_opt_001.dat'
		# cc_in_opt = np.loadtxt(cc_file)
		# # print(cc_in.shape)
	
	number_file = '../data/fermi_4FGL_associations_ext_GRPHorder.csv'
	read_in = pd.read_csv(number_file)
	
	name_4FGL = read_in.loc[obj_num]['name_4FGL']
	source_type = read_in.loc[obj_num]['source_type']
	associated_source = read_in.loc[obj_num]['associated_source']
	associated_ra = read_in.loc[obj_num]['associated_ra']
	associated_de = read_in.loc[obj_num]['associated_de']
	gamma_photon_flux = read_in.loc[obj_num]['flux']
	redshift = read_in.loc[obj_num]['redshift']
	try:
		optical_flux_ned = float(read_in.loc[obj_num]['median_optical_flux'])
	except:
		optical_flux_ned = np.nan
	
	###
	###
	### Getting full time array
	###
	###
	optic_tag = 1
	try:
		optic_file = '../data/optical/object'+str(obj_num).zfill(4)+'_asas-sn.csv'
		read_in = pd.read_csv(optic_file)
		
		time = read_in['HJD'].values
		time_min_opt = round(time.min())
		time_max_opt = round(time.max())
	except:
		optic_tag = 0
	
	gamma_tag = 1
	try:
		gamma_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.dat'
		read_in = np.loadtxt(gamma_file).T
		
		time = read_in[0]/(60*60*24) + 2451910.5
		time_min_gam = round(time.min())
		time_max_gam = round(time.max())
	except:
		gamma_tag = 0
		

	if optic_tag == 0 and gamma_tag == 0:
		print('No data for object'+str(obj_num).zfill(4))
		continue
		
	if gamma_tag == 0:
		print('No gamma-ray data for object'+str(obj_num).zfill(4))
		continue	

	time_min = round(min(time_min_opt,time_min_gam))
	time_max = round(max(time_max_opt,time_max_gam))
	t = np.arange(time_min,time_max,0.4) #- time_min
	tnum = len(t)
	
	t_year = (t-2440587.5)/(365.25) + 1970
	t_year_min = t_year[0]
	t_year_max = t_year[-1]
	t_year_limits = [t_year_min, t_year_max]

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
	
	gamma_method = 0
	
	if gamma_method == 0:
		read_in = np.loadtxt(gamma_file).T

		time = read_in[0]/(60*60*24) + 2451910.5 #- time_min
		flux = read_in[1]
		gamma_lerr_data = read_in[2]
		gamma_uerr_data = read_in[3]
	
	elif gamma_method == 1:
		## remove higher energy bins
		gamma_flux_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.npy'
		read_in = np.load(gamma_flux_file)
		read_in = read_in[:,2:]
		
		time = read_in[:,0,0]/(60*60*24) + 2451910.5
		enum = read_in.shape[1]
		
		counts_E = read_in[:,:,5]
		counts_E_tot = np.sum(counts_E,axis=0)
		E_use = counts_E_tot >= 20.
		
		flux_E = read_in[:,:,1]
		flux_E_ll = read_in[:,:,2]
		flux_E_ul = read_in[:,:,3]
		
		flux = np.sum(flux_E[:,E_use], axis=1)
		gamma_lerr_data = np.sum(flux_E_ll[:,E_use]**2, axis=1)**0.5
		gamma_uerr_data = np.sum(flux_E_ul[:,E_use]**2, axis=1)**0.5
	
	elif gamma_method == 2:
		## photon flux
		energy_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_energy.dat'
		ENERGIES = np.loadtxt(energy_file)
		ENERGIES = ENERGIES[2:]
		
		gamma_flux_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.npy'
		read_in = np.load(gamma_flux_file)
		read_in = read_in[:,2:]
		
		time = read_in[:,0,0]/(60*60*24) + 2451910.5
		enum = read_in.shape[1]
		
		counts_E = read_in[:,:,5]
		counts_E_tot = np.sum(counts_E,axis=0)
		E_use = counts_E_tot >= 20.
		
		flux_E = read_in[:,:,1]
		flux_E_ll = read_in[:,:,2]
		flux_E_ul = read_in[:,:,3]
		
		flux = np.sum(flux_E[:,E_use]/ENERGIES[E_use], axis=1)
		gamma_lerr_data = np.sum((flux_E_ll[:,E_use]/ENERGIES[E_use])**2, axis=1)**0.5
		gamma_uerr_data = np.sum((flux_E_ul[:,E_use]/ENERGIES[E_use])**2, axis=1)**0.5
	
	
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

	###
	###
	### SPECTRAL STUFF
	###
	###
	
	# spectral_tag = 1
	# try:
		# spectral_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_spectrum.dat'
		# read_in = np.loadtxt(spectral_file).T
		
		# t_beta = read_in[0]/(60*60*24) + 2451910.5
		# lbeta = read_in[1]
		# ldbeta_l = read_in[2]
		# ldbeta_u = read_in[3]
		
		# ubeta = read_in[4]
		# udbeta_l = read_in[5]
		# udbeta_u = read_in[6]
		
		# beta_prime = read_in[7]
		# dbeta_prime_l = read_in[8]
		# dbeta_prime_u = read_in[9]
		
	# except:
		# spectral_tag = 0
	
	
	spectral_tag = 1
	try:
		spectral_file = gr_bayes_spec_dir+'/object'+str(obj_num).zfill(4)+'_bayesian_spectrum.dat'
		read_in = np.loadtxt(spectral_file).T
		
		t_beta = read_in[0]/(60*60*24) + 2451910.5
		
		lbeta = read_in[4]
		ldbeta_l = read_in[5]
		ldbeta_u = read_in[6]
		
		ubeta = read_in[7]
		udbeta_l = read_in[8]
		udbeta_u = read_in[9]
		
		beta_prime = read_in[10]+2.
		dbeta_prime_l = read_in[11]
		dbeta_prime_u = read_in[12]
		
	except:
		spectral_tag = 0

	###
	###
	### AUTOCORRELATIONS
	###
	###
	auto_gam_err_tag = 1
	try:
		filename_err = autocorr_gam_err_dir+'object'+str(obj_num).zfill(4)+'_auto_gam_stats'+str(gam_file_num).zfill(3)+'.dat'
		read_in = np.loadtxt(filename_err)

		time_gam_err = read_in[0]
		data_gam_err = read_in[1:]

		cor_mean_gam = data_gam_err[0]
		cor_err_d_gam = cor_mean_gam-data_gam_err[1]
		cor_err_u_gam = data_gam_err[2]-cor_mean_gam
		
	except:
		auto_gam_err_tag = 0
		
	auto_gam_sig_tag = 1
	try:
		filename_sig = autocorr_gam_sig_dir+'cross_stats_gam+gam_object'+str(obj_num).zfill(4)+'_'+str(gam_file_num).zfill(3)+'.dat'
		read_in = np.loadtxt(filename_sig)
		time_gam_sig = read_in[0]
		data_gam_sig = read_in[1:]

		# time_gam_sig_pos_arr = time_gam_sig >= 0.
		# time_gam_sig_neg_arr = time_gam_sig <= 0.

		# data_gam_sig_pos = data_gam_sig[:,time_gam_sig_pos_arr]
		# data_gam_sig_neg = data_gam_sig[:,time_gam_sig_neg_arr]
		# data_gam_sig_neg = data_gam_sig_neg[:,::-1]

		# time_gam_sig = time_gam_sig[time_gam_sig_pos_arr]
		# data_gam_sig = np.mean([data_gam_sig_pos, data_gam_sig_neg], axis=0)

		sig_mean_gam = data_gam_sig[0]
		sig_sig1_d_gam = data_gam_sig[1]
		sig_sig1_u_gam = data_gam_sig[2]
		sig_sig2_d_gam = data_gam_sig[3]
		sig_sig2_u_gam = data_gam_sig[4]
		sig_sig3_d_gam = data_gam_sig[5]
		sig_sig3_u_gam = data_gam_sig[6]
		sig_siga_d_gam = data_gam_sig[7]
		sig_siga_u_gam = data_gam_sig[8]


		sigma3_corr_gam = cor_mean_gam > sig_sig3_u_gam
		sigma3_anti_gam = cor_mean_gam < sig_sig3_d_gam

		sigma2_corr_gam = cor_mean_gam > sig_sig2_u_gam
		sigma2_anti_gam = cor_mean_gam < sig_sig2_d_gam
		sigma2_corr_gam = np.logical_xor(sigma2_corr_gam, sigma3_corr_gam)
		sigma2_anti_gam = np.logical_xor(sigma2_anti_gam, sigma3_anti_gam)

		time_sig3_corr_gam = time_gam_sig[sigma3_corr_gam]
		time_sig3_anti_gam = time_gam_sig[sigma3_anti_gam]
		time_sig2_corr_gam = time_gam_sig[sigma2_corr_gam]
		time_sig2_anti_gam = time_gam_sig[sigma2_anti_gam]
	except:
		auto_gam_sig_tag = 0

	auto_opt_err_tag = 1
	try:
		filename_err = autocorr_opt_err_dir+'object'+str(obj_num).zfill(4)+'_auto_opt_stats'+str(opt_file_num).zfill(3)+'.dat'
		read_in = np.loadtxt(filename_err)

		time_opt_err = read_in[0]
		data_opt_err = read_in[1:]

		cor_mean_opt = data_opt_err[0]
		cor_err_d_opt = cor_mean_opt-data_opt_err[1]
		cor_err_u_opt = data_opt_err[2]-cor_mean_opt
		
	except:
		auto_opt_err_tag = 0
		
	auto_opt_sig_tag = 1
	try:
		filename_sig = autocorr_opt_sig_dir+'cross_stats_opt+opt_object'+str(obj_num).zfill(4)+'_'+str(opt_file_num).zfill(3)+'.dat'
		read_in = np.loadtxt(filename_sig)
		time_opt_sig = read_in[0]
		data_opt_sig = read_in[1:]

		time_opt_sig_pos_arr = time_opt_sig >= 0.
		time_opt_sig_neg_arr = time_opt_sig <= 0.

		data_opt_sig_pos = data_opt_sig[:,time_opt_sig_pos_arr]
		data_opt_sig_neg = data_opt_sig[:,time_opt_sig_neg_arr]
		data_opt_sig_neg = data_opt_sig_neg[:,::-1]

		time_opt_sig = time_opt_sig[time_opt_sig_pos_arr]
		# data_opt_sig = np.mean([data_opt_sig_pos, data_opt_sig_neg], axis=0)

		sig_mean_opt = data_opt_sig[0]
		sig_sig1_d_opt = data_opt_sig[1]
		sig_sig1_u_opt = data_opt_sig[2]
		sig_sig2_d_opt = data_opt_sig[3]
		sig_sig2_u_opt = data_opt_sig[4]
		sig_sig3_d_opt = data_opt_sig[5]
		sig_sig3_u_opt = data_opt_sig[6]
		sig_siga_d_opt = data_opt_sig[7]
		sig_siga_u_opt = data_opt_sig[8]


		sigma3_corr_opt = cor_mean_opt > sig_sig3_u_opt
		sigma3_anti_opt = cor_mean_opt < sig_sig3_d_opt

		sigma2_corr_opt = cor_mean_opt > sig_sig2_u_opt
		sigma2_anti_opt = cor_mean_opt < sig_sig2_d_opt
		sigma2_corr_opt = np.logical_xor(sigma2_corr_opt, sigma3_corr_opt)
		sigma2_anti_opt = np.logical_xor(sigma2_anti_opt, sigma3_anti_opt)

		time_sig3_corr_opt = time_opt_sig[sigma3_corr_opt]
		time_sig3_anti_opt = time_opt_sig[sigma3_anti_opt]
		time_sig2_corr_opt = time_opt_sig[sigma2_corr_opt]
		time_sig2_anti_opt = time_opt_sig[sigma2_anti_opt]
	except:
		auto_opt_sig_tag = 0

	###
	###
	### CROSS CORRELATION
	###
	###
	cross_corr_err_tag = 1
	try:
		filename_err = cc_dir+'object'+str(obj_num).zfill(4)+'_stats'+str(cc_file_num).zfill(3)+'.dat'
		
		read_in = np.loadtxt(filename_err)

		time_err = read_in[0]
		data_err = read_in[1:]

		cor_mean = data_err[0]
		cor_err_d = cor_mean-data_err[1]
		cor_err_u = data_err[2]-cor_mean
		
	except:
		cross_corr_err_tag = 0
	
	cross_corr_sig_tag = 1
	try:
		filename_sig = cross_object_cc_dir+'cross_object'+str(obj_num).zfill(4)+'_all_stats_'+str(cc_file_num).zfill(3)+'.dat'
		
		read_in = np.loadtxt(filename_sig)
		time_sig = read_in[0]
		data_sig = read_in[1:]

		sig_mean = data_sig[0]
		sig_sig1_d = data_sig[1]
		sig_sig1_u = data_sig[2]
		sig_sig2_d = data_sig[3]
		sig_sig2_u = data_sig[4]
		sig_sig3_d = data_sig[5]
		sig_sig3_u = data_sig[6]
		sig_siga_d = data_sig[7]
		sig_siga_u = data_sig[8]


		sigma3_corr = cor_mean > sig_sig3_u
		sigma3_anti = cor_mean < sig_sig3_d

		sigma2_corr = cor_mean > sig_sig2_u
		sigma2_anti = cor_mean < sig_sig2_d
		sigma2_corr = np.logical_xor(sigma2_corr, sigma3_corr)
		sigma2_anti = np.logical_xor(sigma2_anti, sigma3_anti)

		time_sig3_corr = time_sig[sigma3_corr]
		time_sig3_anti = time_sig[sigma3_anti]
		time_sig2_corr = time_sig[sigma2_corr]
		time_sig2_anti = time_sig[sigma2_anti]
	except:
		cross_corr_sig_tag = 0

	
	cross_corr_bb_tag = 1
	try:
		cc_bb_file = cc_bayesian_block_dir+'object'+str(obj_num).zfill(4)+'_cc_change_points_000.dat'
		read_in = np.loadtxt(cc_bb_file)
		# print(read_in)
		cc_bb_norm = read_in[1]
		cc_bb_time_index = read_in[0].astype(np.int)
		
	except:
		cross_corr_bb_tag = 0
	
	###
	### PLOT
	###
	font = {'family' : 'serif',
			'weight' : 'normal',
			'size'   : 5.5}

	matplotlib.rc('font', **font)
	matplotlib.rcParams['mathtext.fontset']='dejavuserif'

	cmap = plt.get_cmap('gist_stern')
	cmap2 = plt.get_cmap('gist_stern')
	cmap3 = plt.get_cmap('brg')
	# cmap  = plt.get_cmap('brg')#'gnuplot2')#'BuPu_r')
	fig = plt.figure()

	gs = gridspec.GridSpec(5, 4)
	gs.update(wspace=0.0, hspace=0.0)
	plt.subplots_adjust(hspace=0.001)
	
	ax00 = plt.subplot(gs[0,0:2])
	ax10 = plt.subplot(gs[1,0:2])
	ax20 = plt.subplot(gs[2,0:2])
	ax30 = plt.subplot(gs[3,0:2])
	ax40 = plt.subplot(gs[4,0:2])
	
	ax01 = plt.subplot(gs[0:2,2:4])
	# ax11 = plt.subplot(gs[0:2,3])
	ax21 = plt.subplot(gs[2:4,2:4])
	ax41 = plt.subplot(gs[4,2:4])
	
	# ax0 = plt.subplot(gs[0,0])
	# ax1 = plt.subplot(gs[1,0])


	for axis in ['top','bottom','left','right']:
		ax00.spines[axis].set_linewidth(0)
	ax00.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax00.tick_params(which='minor',width=0.2, length=3, direction='in')

	for axis in ['top','bottom','left','right']:
		ax10.spines[axis].set_linewidth(0)
	ax10.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax10.tick_params(which='minor',width=0.2, length=3, direction='in')
	ax10.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
	
	for axis in ['top','bottom','left','right']:
		ax20.spines[axis].set_linewidth(0)
	ax20.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax20.tick_params(which='minor',width=0.2, length=3, direction='in')
	ax20.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

	for axis in ['top','bottom','left','right']:
		ax30.spines[axis].set_linewidth(0)
	ax30.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax30.tick_params(which='minor',width=0.2, length=3, direction='in')
	ax30.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

	for axis in ['top','bottom','left','right']:
		ax40.spines[axis].set_linewidth(0)
	ax40.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax40.tick_params(which='minor',width=0.2, length=3, direction='in')
	
	for axis in ['top','bottom','left','right']:
		ax01.spines[axis].set_linewidth(0)
	ax01.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax01.tick_params(which='minor',width=0.2, length=3, direction='in')
	ax01.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

	# for axis in ['top','bottom','left','right']:
		# ax11.spines[axis].set_linewidth(1)
	# ax11.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	# ax11.tick_params(which='minor',width=0.25, length=5, direction='in')
	
	for axis in ['top','bottom','left','right']:
		ax21.spines[axis].set_linewidth(0)
	ax21.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax21.tick_params(which='minor',width=0.2, length=3, direction='in')
	
	for axis in ['top','bottom','left','right']:
		ax21.spines[axis].set_linewidth(0)
	ax21.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax21.tick_params(which='minor',width=0.2, length=3, direction='in')
	
	for axis in ['top','bottom','left','right']:
		ax41.spines[axis].set_linewidth(0)
	ax41.tick_params(which='major',width=0.4, length=5, labelsize='small', direction='in')
	ax41.tick_params(which='minor',width=0.2, length=3, direction='in')
	ax41.tick_params(axis='x', which='minor', bottom=True, top=True)
	
	# ax0.yaxis.set_ticks_position('both')
	# ax0.xaxis.set_ticks_position('both')
	# ax0.minorticks_on()
	# ax0.grid(True, color='grey', zorder=0, alpha=0.15)

	# ax1.xaxis.set_ticks_position('both')
	# ax1.yaxis.set_ticks_position('both')
	# ax1.minorticks_on()
	# ax1.grid(True, color='grey', zorder=0, alpha=0.15)


	MSIZE = 0.1
	ALPHA = 0.5
	ELINE = 0.1
	ALPHA_FILL_DAT = 0.7
	CAPSIZE = 0.0
	
	MED_COL = cmap(0.125)

	t_upper = np.amax([t_opt, t_gam])
	t_lower = np.amin([t_opt, t_gam])
	axN0_limits = [t_lower, t_upper]

	if optic_tag == 1:
		## OPTICAL DATA
		t_opt = (t_opt-2440587.5)/(365.25) + 1970
		t_opt_d = (t_opt_d-2440587.5)/(365.25) + 1970
		
		ax00.errorbar(t_opt_d[F_opt_d>0], F_opt_d[F_opt_d>0], yerr=fe_opt_d[F_opt_d>0], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.02), zorder=20)
		ax00.errorbar(t_opt_d[F_opt_d<=0], F_opt_d[F_opt_d<=0], yerr=fe_opt_d[F_opt_d<=0], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.12), zorder=20)
		
		## median and average horizontal lines
		ax00.hlines(np.median(F_opt_d), xmin=t_opt_d[0], xmax=t_opt_d[-1], alpha=0.6, color=cmap2(0.02), linewidth=0.5)
		# ax00.hlines(np.mean(F_opt_d), xmin=t_opt_d[0], xmax=t_opt_d[-1], alpha=0.6, color=cmap2(0.02), linewidth=0.5)
		
		# if optical_flux_ned < 0.1 np.median()
		# ax00.scatter(2014, optical_flux_ned, s=10, marker='x', color=cmap2(0.35))
		
		## Plot properties
		opt_min = np.percentile(abs(F_opt_d[F_opt_d!=-100]), 5)/1.2
		opt_max = 1.2*np.amax(F_opt_d + fe_opt_d)
		while opt_max/opt_min <= 11:
			opt_max*= 1.2
			opt_min/= 1.2
		
		
		# ## PLOT MEDIAN NED FLUX
		# ned_opt_x = 2012.25
		# if optical_flux_ned > opt_max:
			# arrow_begin = np.exp(np.log(opt_min) + 0.8*np.log(opt_max/opt_min))
			# ax00.annotate('', xy=(ned_opt_x,opt_max), xytext=(ned_opt_x, arrow_begin), arrowprops=dict(arrowstyle="->", color=cmap2(0.35), alpha=0.8))
			
			# text_center_y = np.exp(np.log(opt_min) + 0.9*np.log(opt_max/opt_min))
			# ax00.text(ned_opt_x+0.1, text_center_y, str('{:10.3f}'.format(optical_flux_ned)), ha='left', va='center', color=cmap2(0.35), fontsize='x-small')
		# elif optical_flux_ned < opt_min:
			# arrow_begin = np.exp(np.log(opt_min) + 0.2*np.log(opt_max/opt_min))
			# ax00.annotate('', xy=(ned_opt_x,opt_min), xytext=(ned_opt_x, arrow_begin), arrowprops=dict(arrowstyle="->", color=cmap2(0.35), alpha=0.8))
			
			# text_center_y = np.exp(np.log(opt_min) + 0.1*np.log(opt_max/opt_min))
			# ax00.text(ned_opt_x+0.1, text_center_y, str("{:10.3f}".format(optical_flux_ned)), ha='left', va='center', color=cmap2(0.35), fontsize='x-small')
		# else:
			# ax00.scatter(ned_opt_x, optical_flux_ned, s=8, marker='x', color=cmap2(0.35), alpha=0.8)
		
		ax00.set_ylim([opt_min,opt_max])
		t_limit_plot = [(t_lower-2440587.5)/(365.25) + 1970, (t_upper-2440587.5)/(365.25) + 1970]
		ax00.set_xlim(t_limit_plot)
		
		ax00.set_yscale('log')
		ax00.set_ylabel('$F$ [mJy]')
		
		ax00.xaxis.set_ticks_position('top')
		plt.setp(ax00.get_xticklabels(), ha="left", rotation=45)
		ax00.set_xlabel(r'Year')
		ax00.xaxis.set_label_position('top') 
		# ax00.set_xticklabels([])

	if gamma_tag == 1:
		## GAMMA-RAY DATA
		# ax10.errorbar(t_gam, F_gam, yerr=[fe_d_gam,fe_u_gam], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.1), zorder=21)
		ax10.errorbar(t_gam_d, F_gam_d, yerr=[fe_gam_dd,fe_gam_ud], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.6), zorder=20)
		
		## median and average lines
		ax10.hlines(np.median(F_gam_d), xmin=t_gam_d[0], xmax=t_gam_d[-1], alpha=0.6, color=cmap2(0.6), linewidth=0.5)
		# ax10.hlines(np.mean(F_gam_d), xmin=t_gam_d[0], xmax=t_gam_d[-1], alpha=0.6, color=cmap2(0.02), linewidth=0.5)
		
		## plot properties
		gam_min = min(np.percentile(F_gam[F_gam>0], 10), np.percentile(F_gam_d[F_gam_d>0], 10))
		gam_max = max(np.amax(F_gam + fe_u_gam), np.amax(F_gam_d + fe_gam_ud))

		ax10.set_yscale('log')
		ax10.set_ylabel(r'$E\Phi_\gamma$ $\left[\frac{\mathrm{MeV}}{\mathrm{cm}^{2}\ \mathrm{s}} \right]$')
		
		ax10.set_ylim([gam_min/2.,gam_max*1.01])
		gam_min, gam_max = ax10.get_ylim()
		while gam_max/gam_min <= 11:
			gam_max*= 1.2
			gam_min/= 1.2
		ax10.set_ylim([gam_min, gam_max])
		
		ax10.set_xlim(axN0_limits)
		ax10.set_xticklabels([])
		
	if optic_tag == 1 and gamma_tag == 1:
		## optical to gamma-ray ratio
		ax20.errorbar(t_ratio[gam_opt_ratio>0], gam_opt_ratio[gam_opt_ratio>0], yerr=[ratio_err_d[gam_opt_ratio>0],ratio_err_u[gam_opt_ratio>0]], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.35), zorder=21)
		ax20.errorbar(t_ratio[gam_opt_ratio<=0], abs(gam_opt_ratio[gam_opt_ratio<=0]), yerr=[ratio_err_d[gam_opt_ratio<=0],ratio_err_u[gam_opt_ratio<=0]], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.55), zorder=20)
		
		## median and mean of ratio
		ax20.hlines(np.median(gam_opt_ratio), xmin=t_ratio[0], xmax=t_ratio[-1], alpha=0.6, color=cmap2(0.35), linewidth=0.5)
		# ax20.hlines(np.mean(F_opt_ratio)/np.mean(F_gam_ratio), xmin=t_ratio[0], xmax=t_ratio[-1], alpha=0.6, color=cmap2(0.02), linewidth=0.5)
		
		## plot properties
		ax20.set_xlim(axN0_limits)
		
		ax20.set_yscale('log')
		ax20.set_ylabel(r'$\frac{\mathrm{Optical\ Flux}}{\gamma\mathrm{-ray\ Flux}}$')
		
		ratio_min = np.percentile(gam_opt_ratio[gam_opt_ratio>0], 10)
		ratio_max = np.amax(gam_opt_ratio + ratio_err_u)
		
		ax20.set_ylim([ratio_min/2.,ratio_max*1.01])
		ratio_min, ratio_max = ax20.get_ylim()
		while ratio_max/ratio_min <= 11:
			ratio_max*= 1.2
			ratio_min/= 1.2
		ax20.set_ylim([ratio_min, ratio_max])
		
		ax20.set_xticklabels([])
		
		# ax20.fill_between([t_ratio[0], t_ratio[-1]], [1.e1, 1.e1], [1.e5, 1.e5], color=cmap2(0.8), edgecolor='none', hatch='//', alpha=0.2)
		
		
	if spectral_tag == 1:
		##
		## SPECTRAL INDEX
		##
		
		## data points for lower energy band
		ax30.errorbar(t_beta, lbeta, yerr=[ldbeta_l, ldbeta_u], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.25), zorder=22)
		
		# ## median and average lines
		ax30.hlines(np.median(lbeta), xmin=t_beta[0], xmax=t_gam_d[-1], alpha=0.6, color=cmap2(0.25), linewidth=0.5)
		# ax30.hlines(np.mean(lbeta), xmin=t_beta[0], xmax=t_gam_d[-1], alpha=0.6, color=cmap2(0.02), linewidth=0.5)
		
		## data points for higher energy band
		ax30.errorbar(t_beta, ubeta, yerr=[udbeta_l, udbeta_u], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.75), zorder=21)
		
		# ## median and average lines
		ax30.hlines(np.median(ubeta), xmin=t_beta[0], xmax=t_gam_d[-1], alpha=0.6, color=cmap2(0.75), linewidth=0.5)
		# ax30.hlines(np.mean(ubeta), xmin=t_beta[0], xmax=t_gam_d[-1], alpha=0.6, color=cmap2(0.09), linewidth=0.5)
		
		## plot properties
		ax30.set_ylabel(r'$\beta$')
		
		ax30.set_xlim(axN0_limits)
		ax30.set_xticklabels([])
		
		ax30.set_ylim([-3.999, -1.001])
		
		##
		## CURVATURE
		##
		
		## data points for curvature
		ax40.errorbar(t_beta-2450000, beta_prime, yerr=[dbeta_prime_l, dbeta_prime_u], fmt='x', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=CAPSIZE, capthick=0.5, color=cmap2(0.15), zorder=21)
		
		# ## median and average lines
		ax40.hlines(np.median(beta_prime), xmin=t_beta[0]-2450000, xmax=t_beta[-1]-2450000, alpha=0.6, color=cmap2(0.15), linewidth=0.5)
		# ax40.hlines(np.mean(beta_prime), xmin=t_beta[0]-2450000, xmax=t_beta[-1]-2450000, alpha=0.6, color=cmap2(0.02), linewidth=0.5)
		
		## plot properties
		ax40.set_ylabel(r'$\kappa$')
		ax40.set_xlabel('HJD-2450000')
		ax40.set_xlim([t_lower-2450000, t_upper-2450000])
		ax40.set_ylim([-0.599, 0.599])
		
		plt.setp(ax40.get_xticklabels(), ha="right", rotation=45)
		
	if auto_opt_err_tag == 1:
		## optical autocorrelation DATA
		ax01.fill_between(-time_opt_err, cor_mean_opt-cor_err_d_opt, cor_mean_opt+cor_err_u_opt, alpha=ALPHA_FILL_DAT, facecolor=cmap2(0.12), edgecolor='k', linewidth=ELINE)
		# ax01.errorbar(-time_opt_err, cor_mean_opt, yerr=[cor_err_d_opt,cor_err_u_opt], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		ax41.fill_between(-time_opt_err, cor_mean_opt-cor_err_d_opt, cor_mean_opt+cor_err_u_opt, alpha=ALPHA_FILL_DAT, facecolor=cmap2(0.12), edgecolor='k', linewidth=ELINE)
		# ax41.errorbar(-time_opt_err, cor_mean_opt, yerr=[cor_err_d_opt,cor_err_u_opt], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		# ax01.set_xlim([0,2999.999])
		# ax01.set_ylim([-0.999,0.999])
		
		# # ax01.set_xticklabels([])
		# ax01.set_yticklabels([])
		
		# ax01.set_xlim(ax01.get_xlim()[::-1])
		# ax01.xaxis.set_ticks_position('top')
		# plt.setp(ax01.get_xticklabels(), ha="left", rotation=45)
		
	if auto_opt_sig_tag == 1:
		## optical correlation SIGNIFICANCE
		alpha_fill = 0.2
		lwidth_fill = 0.1
		ax01.fill_between(-time_opt_sig, sig_sig1_d_opt, sig_sig1_u_opt, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.914)
		ax01.fill_between(-time_opt_sig, sig_sig2_d_opt, sig_sig2_u_opt, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.913)
		ax01.fill_between(-time_opt_sig, sig_sig3_d_opt, sig_sig3_u_opt, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.912)
		# ax01.fill_between(-time_opt_sig, sig_siga_d_opt, sig_siga_u_opt, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.911)
		
		LWIDTH = 0.075
		alpha_fill = 0.2
		for tc in time_sig3_corr_opt:
			ax01.axvline(x=-tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
		for tc in time_sig3_anti_opt:
			ax01.axvline(x=-tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
			
		for tc in time_sig2_corr_opt:
			ax01.axvline(x=-tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
		for tc in time_sig2_anti_opt:
			ax01.axvline(x=-tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
		
	if auto_gam_err_tag == 1:
		ax01.fill_between(time_gam_err, cor_mean_gam-cor_err_d_gam, cor_mean_gam+cor_err_u_gam, alpha=ALPHA_FILL_DAT, facecolor=cmap(0.6), edgecolor='k', linewidth=ELINE)
		# ax01.errorbar(time_gam_err, cor_mean_gam, yerr=[cor_err_d_gam,cor_err_u_gam], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		ax01.set_xlim([-1999.999,2999.999])
		ax01.set_ylim([-0.999,0.999])
		
		# ax11.set_xticklabels([])
		# ax11.set_yticklabels([])
		
		ax01.yaxis.set_ticks_position('right')
		
		ax01.xaxis.set_ticks_position('top')
		plt.setp(ax01.get_xticklabels(), ha="left", rotation=45)
		ax01.set_xlabel(r'Days')
		ax01.xaxis.set_label_position('top') 
		ax01.set_ylabel(r'$r_s$')
		ax01.yaxis.set_label_position('right')
		
		ax01.text(-1850., 0.85, r'AC', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		ax01.text(150., -0.85, r'$\gamma$-ray$\rightarrow$', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		ax01.text(-150., -0.85, r'$\leftarrow$Optical', verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		ax41.fill_between(time_gam_err, cor_mean_gam-cor_err_d_gam, cor_mean_gam+cor_err_u_gam, alpha=ALPHA_FILL_DAT, facecolor=cmap(0.6), edgecolor='k', linewidth=ELINE)
		# ax41.errorbar(time_gam_err, cor_mean_gam, yerr=[cor_err_d_gam,cor_err_u_gam], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		# ax02 = plt.subplot(gs[0:2,4:6])
		# ax02.plot(np.fft.fftfreq(len(time_gam_err), d=0.4), np.fft.fft(cor_mean_gam))
		# # ax02.set_xscale('log')
		# ax02.set_xlim([1.7e-2,2.e-2])
		
	if auto_gam_sig_tag == 1:
		alpha_fill = 0.2
		lwidth_fill = 0.1
		ax01.fill_between(time_gam_sig, sig_sig1_d_gam, sig_sig1_u_gam, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.914)
		ax01.fill_between(time_gam_sig, sig_sig2_d_gam, sig_sig2_u_gam, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.913)
		ax01.fill_between(time_gam_sig, sig_sig3_d_gam, sig_sig3_u_gam, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.912)
		# ax01.fill_between(time_gam_sig, sig_siga_d_gam, sig_siga_u_gam, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.911)
		
		LWIDTH = 0.075
		alpha_fill = 0.2
		for tc in time_sig3_corr_gam:
			ax01.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
		for tc in time_sig3_anti_gam:
			ax01.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
			
		for tc in time_sig2_corr_gam:
			ax01.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
		for tc in time_sig2_anti_gam:
			ax01.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
			
		
		
	if cross_corr_err_tag == 1:
		ax21.fill_between(time_err, cor_mean-cor_err_d, cor_mean+cor_err_u, alpha=ALPHA_FILL_DAT, facecolor='k', edgecolor='k', linewidth=ELINE)
		# ax21.errorbar(time_err, cor_mean, yerr=[cor_err_d,cor_err_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		ax21.set_xlim([-1999.999,2999.999])
		ax21.set_ylim([-0.999,0.999])
		
		ax21.text(-1850., 0.85, r'CC', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		ax21.text(150., -0.85, r'$\gamma$-ray lead$\rightarrow$', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		ax21.text(-150., -0.85, r'$\leftarrow$Optical lead', verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		ax21.set_ylabel(r'$r_s$')
		ax21.yaxis.set_label_position('right')
		
		# ax21.set_xticklabels([])
		# ax21.set_yticklabels([])
		
		ax41.fill_between(time_err, cor_mean-cor_err_d, cor_mean+cor_err_u, alpha=ALPHA_FILL_DAT, facecolor='k', edgecolor='k', linewidth=ELINE)
		# ax41.errorbar(time_err, cor_mean, yerr=[cor_err_d,cor_err_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		ax41.set_xlim([-39.99,59.99])
		ax41.set_ylim([0.001,0.999])
		
		ax21.yaxis.set_ticks_position('right')
		plt.setp(ax21.get_xticklabels(), ha='right', rotation=45)
		ax21.zorder=1000
		
		ax41.yaxis.set_ticks_position('right')
		plt.setp(ax41.get_xticklabels(), ha="right", rotation=45)
		
		ax41.text(0., 0.15, r'AC+CC zoom', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		ax41.set_xlabel(r'Days')
		ax41.set_ylabel(r'$r_s$')
		ax41.yaxis.set_label_position('right')
		
		if cross_corr_bb_tag == 1:
			ax21.step(time_err[cc_bb_time_index], cc_bb_norm, where='post', color=cmap2(0.75), linewidth=0.5, alpha=0.8)
			ax41.step(time_err[cc_bb_time_index], cc_bb_norm, where='post', color=cmap2(0.75), linewidth=0.5, alpha=0.8)
		
		
		
	if cross_corr_sig_tag == 1:
		alpha_fill = 0.2
		lwidth_fill = 0.1
		ax21.fill_between(time_sig, sig_sig1_d, sig_sig1_u, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.914)
		ax21.fill_between(time_sig, sig_sig2_d, sig_sig2_u, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.913)
		ax21.fill_between(time_sig, sig_sig3_d, sig_sig3_u, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.912)
		# ax21.fill_between(time_sig, sig_siga_d, sig_siga_u, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.911)

		LWIDTH = 0.075
		alpha_fill = 0.2
		for tc in time_sig3_corr:
			ax21.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
		for tc in time_sig3_anti:
			ax21.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
			
		for tc in time_sig2_corr:
			ax21.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
		for tc in time_sig2_anti:
			ax21.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
			
		
		
		
		alpha_fill = 0.2
		lwidth_fill = 0.1
		ax41.fill_between(time_sig, sig_sig1_d, sig_sig1_u, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.914)
		ax41.fill_between(time_sig, sig_sig2_d, sig_sig2_u, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.913)
		ax41.fill_between(time_sig, sig_sig3_d, sig_sig3_u, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.912)
		# ax41.fill_between(time_sig, sig_siga_d, sig_siga_u, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.911)

		# LWIDTH = 0.6
		# alpha_fill = 0.1
		# for tc in time_sig3_corr:
			# ax41.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
		# for tc in time_sig3_anti:
			# ax41.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
			
		# for tc in time_sig2_corr:
			# ax41.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
		# for tc in time_sig2_anti:
			# ax41.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)

	# if CHANGE_POINTS == 1:
		# ax0.step(t_opt_d[cc_in_opt[0].astype(np.int)], cc_in_opt[1], where='post', color='orange', linewidth=0.2)
		# ax1.step(t_gam_d[cc_in_gam[0].astype(np.int)], cc_in_gam[1], where='post', color='orange', linewidth=0.2)
	
	# ax1.set_xlabel('Time [days]')
	
	
	type_color = 'black'
	if source_type == 'FSRQ':
		type_color = cmap3(0.49)
	elif source_type == 'fsrq':
		type_color = cmap3(0.4)
	elif source_type == 'BCU':
		type_color = cmap3(0.25)
	elif source_type == 'bcu':
		type_color = cmap3(0.25)
	elif source_type == 'bll':
		type_color = cmap3(0.1)
	elif source_type == 'BLL':
		type_color = cmap3(0.01)
	
	vd = 0.133
	ax00.text(0.1, 0.833-vd, 'Object '+str(obj_num), transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	
	ax00.text(0.1, 0.7-vd, name_4FGL, transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	ax00.text(0.1, 0.567-vd, associated_source, transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	ax00.text(0.1, 0.433-vd, 'Type =', transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	ax00.text(0.213, 0.433-vd, source_type, transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', color=type_color, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	ax00.text(0.1, 0.3-vd, 'z =', transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	
	try:
		redshift_num = float(redshift)
		if redshift_num < 2:
			z_color = cmap3(redshift_num/4.)
		else:
			z_color = cmap3(0.5)
		ax00.text(0.155, 0.3-vd, str('{:1.3f}'.format(float(redshift))), transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', color=z_color, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	except:
		z_color = 'black'
		ax00.text(0.155, 0.3-vd, str(redshift), transform=ax00.transAxes, verticalalignment='center', horizontalalignment='left', color=z_color, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	
	plt.tight_layout()
	plot_name = '../analysis/full_all_v1/gamma_method'+str(gamma_method)+'/plot_all_object'+str(obj_num).zfill(4)+'_'
	# plot_name = '../analysis/full_all_v1/pdf/plot_all_object'+str(obj_num).zfill(4)+'_'
	for n in range(0,100):
		if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
			continue
		else:
			plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
			break
	plt.close()
		
	print('Plotted object'+str(obj_num).zfill(4))

