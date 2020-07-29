import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec

# obj_num = 11
# ZOOM = 0

gam_ver = 3
# object_arr = np.array([0,1])
object_arr = range(26,34)
# object_arr = range(0,3131)


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

	###
	###
	### Getting full time array
	###
	###
	# optic_tag = 1
	# try:
		# optic_file = '../data/optical/object'+str(obj_num).zfill(4)+'_asas-sn.csv'
		# read_in = pd.read_csv(optic_file)
		
		# time = read_in['HJD'].values
		# time_min_opt = round(time.min())
		# time_max_opt = round(time.max())
	# except:
		# optic_tag = 0

	gamma_tag = 1
	try:
		gamma_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.dat'
		read_in = np.loadtxt(gamma_file).T
		
		time = read_in[0]/(60*60*24) + 2451910.5
		time_min_gam = round(time.min())
		time_max_gam = round(time.max())
	except:
		gamma_tag = 0
		

	if gamma_tag == 0:
		print('No data for object'+str(obj_num).zfill(4))
		continue

	# time_min = round(min(time_min_opt,time_min_gam))
	# time_max = round(max(time_max_opt,time_max_gam))
	t = np.arange(time_min_gam,time_max_gam,0.4) #- time_min
	tnum = len(t)
	
	t_year = (t-2440587.5)/(365.25) + 1970
	t_year_min = t_year[0]
	t_year_max = t_year[-1]
	t_year_limits = [t_year_min, t_year_max]

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

	# ##
	# ## Regridding over dt days
	# ## Doing this for simplicity
	# # Assuming independence of measurements,
	# # We just avg mu values and inv-sum sigma
	# f = np.zeros(tnum)-100
	# fe_d = np.zeros(tnum)-1
	# fe_u = np.zeros(tnum)-1

	# dtf = 0.45
	# for ii in range(0,tnum):
		# t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt)
		# t_arr = np.logical_and( t_arr, flux>0)
		# t_arrf = np.logical_and( time<=t[ii]+dtf, time>=t[ii]-dtf ) 
		# t_arrf = np.logical_and( t_arrf, flux>0)
		# if np.any(t_arr) and np.any(t_arrf):
			# f[ii] = np.mean(flux[t_arr])
			# fe_d[ii] = (np.sum(flux_err_2d[t_arr])/np.sum(t_arr))**0.5
			# fe_u[ii] = (np.sum(flux_err_2u[t_arr])/np.sum(t_arr))**0.5	
		
	t_gam_d = time
	F_gam_d = flux
	fe_gam_dd = (gamma_lerr_data)
	fe_gam_ud = (gamma_lerr_data)

	# t_gam = t
	# F_gam = f
	# fe_d_gam = fe_d
	# fe_u_gam = fe_u

	###
	###
	### GAMMA-RAY FULL FLUX
	###
	###
	gamma_flux_tag = 1
	try:
		gamma_flux_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.npy'
		read_in = np.load(gamma_flux_file)
		enum = read_in.shape[1]
		t_flux = read_in[:,0,0]/(60*60*24) + 2451910.5 #- time_min
		tnum = len(t_flux)
		flux = read_in[:,:,1]
		gamma_lerr_data_flux = read_in[:,:,2]
		gamma_uerr_data_flux = read_in[:,:,3]
		flux_err_d = gamma_lerr_data_flux
		flux_err_u = gamma_uerr_data_flux
		
		ENERGY = np.array([125.9, 199.5, 316.2, 501.2, 794.3, 1258.9, 1995.3, 3162.3, 5011.9, 7943.3, 12589.3, 19952.6, 31622.8, 50118.7, 79432.8, 125892.5, 199526.2, 316227.8, 501187.2, 794328.2])
		ENERGY = ENERGY[:enum]
		
	except:
		gamma_flux_tag = 0

	# ###
	# ###
	# ### SPECTRAL STUFF
	# ###
	# ###
	
	# spectral_tag = 1
	# try:
		# spectral_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_spectrum.dat'
		# read_in = np.loadtxt(spectral_file).T
		
		# t_beta = (read_in[0]-239643817.0)/(60*60*24) + 2454684.15527778
		# beta = read_in[1]
		# dbeta_l = read_in[2]
		# dbeta_u = read_in[3]
		
		# beta_prime = read_in[4]
		# dbeta_prime_l = read_in[5]
		# dbeta_prime_u = read_in[6]
		
	# except:
		# spectral_tag = 0
		
		
	# bayesian_tag = 1
	# try:
		# bayes_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_bayesian_spectrum.dat'
		# read_in = np.loadtxt(bayes_file).T
		# t_beta_b = (read_in[0]-239643817.0)/(60*60*24) + 2454684.15527778
		# beta_b = read_in[1]
		# dbeta_b = read_in[2]
	# except:
		# bayesian_tag = 0



	###
	### PLOT
	###
	font = {'family' : 'serif',
			'weight' : 'normal',
			'size'   : 6}

	matplotlib.rc('font', **font)
	matplotlib.rcParams['mathtext.fontset']='dejavuserif'

	cmap = plt.get_cmap('gist_stern')
	cmap2= plt.get_cmap('gnuplot2')#'BuPu_r')
	fig = plt.figure()

	gs = gridspec.GridSpec(5, 5)
	gs.update(wspace=0.0, hspace=0.0)
	plt.subplots_adjust(hspace=0.001)
	
	total_num = 25
	
	# ax00 = plt.subplot(gs[0,0:2])
	# ax10 = plt.subplot(gs[1:3,0:2])
	# ax20 = plt.subplot(gs[2,0:2])
	# ax30 = plt.subplot(gs[3,0:2])
	# ax40 = plt.subplot(gs[4,0:2])
	
	# ax01 = plt.subplot(gs[0,2])
	# ax11 = plt.subplot(gs[1,2])
	# ax21 = plt.subplot(gs[2,2])
	
	# ax0 = plt.subplot(gs[0,0])
	# ax1 = plt.subplot(gs[1,0])


	# for axis in ['top','bottom','left','right']:
		# ax0.spines[axis].set_linewidth(1)
	# ax0.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	# ax0.tick_params(which='minor',width=0.25, length=5, direction='in')

	# for axis in ['top','bottom','left','right']:
		# ax1.spines[axis].set_linewidth(1)
	# ax1.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	# ax1.tick_params(which='minor',width=0.25, length=5, direction='in')

	# ax0.yaxis.set_ticks_position('both')
	# ax0.xaxis.set_ticks_position('both')
	# ax0.minorticks_on()
	# ax0.grid(True, color='grey', zorder=0, alpha=0.15)

	# ax1.xaxis.set_ticks_position('both')
	# ax1.yaxis.set_ticks_position('both')
	# ax1.minorticks_on()
	# ax1.grid(True, color='grey', zorder=0, alpha=0.15)


	MSIZE = 0.6
	ALPHA = 0.2
	ELINE = 0.5

	# t_upper = np.amax([t_gam_d])
	# t_lower = np.amin([t_gam_d])
	# axN0_limits = [t_lower, t_upper]

	flux_max = np.amax(flux+flux_err_u)
	flux_min = np.amin(flux-flux_err_d)

	time_edge = np.arange(0, tnum, int(tnum/25)-1)
	if gamma_tag == 1:
		for mm in range(0,5):
			for nn in range(0,5):
				ax = plt.subplot(gs[mm,nn])
				times_index = np.random.randint(time_edge[5*mm+nn], time_edge[5*mm+nn+1], size=5)
				times_index = np.sort(times_index)
				for oo in range(0,5):
					# print(len(flux[times_index[oo]]), len(flux_err_d[times_index[oo]]), len(flux_err_u[times_index[oo]]) )
					ax.errorbar(ENERGY+ENERGY*0.05*oo, flux[times_index[oo]], yerr=[flux_err_d[times_index[oo]],flux_err_u[times_index[oo]]], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2((oo+1)/7), zorder=20)
					
					ax.set_xscale('log')
					ax.set_yscale('log')
					
					ax.set_xticklabels([])
					ax.set_yticklabels([])
					
					ax.set_ylim([flux_min,flux_max])
	
		# ax00.errorbar(t_gam, F_gam, yerr=[fe_d_gam,fe_u_gam], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.1), zorder=21)
		# ax00.errorbar(t_gam_d, F_gam_d, yerr=[fe_gam_dd,fe_gam_ud], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.5), zorder=20)
		
		# gam_min = np.percentile(F_gam_d[F_gam_d>0], 10)
		# gam_max = np.amax(F_gam_d + fe_gam_ud)

		# ax00.set_ylim([gam_min/2.,gam_max*1.5])
		# ax00.set_xlim(axN0_limits)
		
		# ax00.set_yscale('log')
		# ax00.set_ylabel(r'$E\Phi$ [MeV cm$^{-2}$ s$^{-1}$]')
		
		# ax00.set_xticklabels([])
		
	# print(gamma_flux_tag)
	# if gamma_flux_tag == 1:
	
		
	
	
		# plot_e = np.array([0,-5,-4,-3,-2,-1])
		# for jj in range(0,len(plot_e)):
			# # print('stuff'+str(jj))
			# factor = 1.#10**(2*jj)
			# ax10.errorbar(t_flux[:], flux[:,plot_e[jj]]*factor, yerr=[flux_err_d[:,plot_e[jj]]*factor,flux_err_u[:,plot_e[jj]]*factor], fmt='o', markersize=MSIZE,alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(jj/len(plot_e)), zorder=21)
			
		# ax10.set_yscale('log')
		# ax10.set_xlim(axN0_limits)
		
	# if spectral_tag == 1:
		# ax30.errorbar(t_beta, beta, yerr=[dbeta_l, dbeta_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.35), zorder=21)
		
		# ax30.set_ylabel(r'$\beta$')
		
		# ax30.set_xlim(axN0_limits)
		# ax30.set_xticklabels([])
		
		# ax40.errorbar(t_beta-2450000, beta_prime, yerr=[dbeta_prime_l, dbeta_prime_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.35), zorder=21)
		
		# ax40.set_ylabel('Curvature')
		# ax40.set_xlabel('HJD-2450000')
		# ax40.set_xlim([t_lower-2450000, t_upper-2450000])
		
		# plt.setp(ax40.get_xticklabels(), ha="right", rotation=45)
		
	# if bayesian_tag == 1:
		# ax30.errorbar(t_beta_b, beta_b, yerr=dbeta_b, fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.6), zorder=21.1)

	# if CHANGE_POINTS == 1:
		# ax0.step(t_opt_d[cc_in_opt[0].astype(np.int)], cc_in_opt[1], where='post', color='orange', linewidth=0.2)
		# ax1.step(t_gam_d[cc_in_gam[0].astype(np.int)], cc_in_gam[1], where='post', color='orange', linewidth=0.2)
	
	# ax1.set_xlabel('Time [days]')
	
	plt.tight_layout()
	plot_name = '../analysis/full_all_gam_spectrum_v1/plot_gam_spectrum_object'+str(obj_num).zfill(4)+'_'
	for n in range(0,100):
		if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
			continue
		else:
			plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
			break
	plt.close()
		
	print('Plotted object'+str(obj_num).zfill(4))

