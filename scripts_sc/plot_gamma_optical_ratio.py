import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec

# obj_num = 11
ZOOM = 0
gam_ver = 2

object_arr = range(0,3130)

for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]	
		
	#time to average data over
	dt = 0.45

	CHANGE_POINTS = 0
	if CHANGE_POINTS == 1:
		cc_file = '../data/gamma-ray/object_'+str(obj_num).zfill(4)+'_change_points_gam_002.dat'
		cc_in_gam = np.loadtxt(cc_file)
		cc_file = '../data/optical/object_'+str(obj_num).zfill(4)+'_change_points_opt_001.dat'
		cc_in_opt = np.loadtxt(cc_file)
		# print(cc_in.shape)

	###
	###
	### Getting full time array
	###
	###
	optic_tag = 1
	try:
		optic_file = '../data/optical/object'+str(obj_num).zfill(4)+'_asas-sn.csv'
		read_in = pd.read_csv(optic_file)
	except:
		optic_tag = 0
		optic_file = '../data/optical/object'+str(0).zfill(4)+'_asas-sn.csv'
		read_in = pd.read_csv(optic_file)

	time = read_in['HJD'].values
	time_min_opt = round(time.min())
	time_max_opt = round(time.max())

	gamma_tag = 1
	try:
		gamma_file = '../data/gamma-ray/v'+str(gam_ver)+'//object'+str(obj_num).zfill(4)+'_gam.dat'
		read_in = np.loadtxt(gamma_file).T
	except:
		gamma_tag = 0
		gamma_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(0).zfill(4)+'_gam.dat'
		read_in = np.loadtxt(gamma_file).T
	
	try:
		time = read_in[0]/(60*60*24) + 2451910.5
		time_min_gam = round(time.min())
		time_max_gam = round(time.max())

		time_min = round(min(time_min_opt,time_min_gam))
		time_max = round(max(time_max_opt,time_max_gam))

		t = np.arange(time_min,time_max,0.2) -time_min
		tnum = len(t)
	except:
		gamma_tag = 0

	if optic_tag == 0 or gamma_tag == 0:
		print('No data for object'+str(obj_num).zfill(4))
		continue

	###
	###
	### OPTICAL
	###
	###

	read_in = pd.read_csv(optic_file)

	time = read_in['HJD'].values-time_min
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

	# ##
	# ## We need to take care of AVERAGE mean
	# ## In this case, I will just take a moving average to smooth data
	# # We also need to clean data
	# # I get rid of fluxes above 7.75
	# F = np.zeros(tnum)-100
	# Fe = np.zeros(tnum)-10

	# # DT = 30
	# for ii in range(0,tnum):
		# t_arr = np.logical_and( t<=t[ii], t>=t[ii]-DT)
		# t_arr = np.logical_and( t_arr, f!=-100)
		# # t_arr = np.logical_and( t_arr, f<=7.75)
		# if np.any(t_arr):
			# f_mean = np.mean(f[t_arr])
			# if f[ii] != -100 and f[ii]<=7.75:
				# F[ii] = (f[ii]-f_mean)/fe[ii]
				# Fe[ii] = 1. # fe[ii]#/f_mean
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

	# ##
	# ## We need to take care of AVERAGE mean
	# ## In this case, I will just take a moving average to smooth data
	# # We also need to clean data
	# # I get rid of fluxes above 7.75
	# F = np.zeros(tnum)-100
	# Fe = np.zeros(tnum)-1

	# # DT = 30
	# for ii in range(0,tnum):
		# t_arr = np.logical_and( t<=t[ii], t>=t[ii]-DT)
		# t_arr = np.logical_and( t_arr, f>0.)
		# # t_arr = np.logical_and( t_arr, f<=7.75)
		# if np.any(t_arr):
			# f_mean = np.mean(f[t_arr])
			# if f[ii] > 0.:
				# F[ii] = (f[ii]-f_mean)/fe[ii]
				# Fe[ii] = 1. # fe[ii]#/f_mean
	gfactor = 1.		
		
	t_gam_d = time
	F_gam_d = flux*gfactor
	fe_gam_dd = (gamma_lerr_data)*gfactor
	fe_gam_ud = (gamma_lerr_data)*gfactor

	t_gam = t
	F_gam = f*gfactor
	fe_d_gam = fe_d*gfactor
	fe_u_gam = fe_u*gfactor

	###
	###
	### CALC RATIO
	###
	bool_arr = np.logical_and(F_gam != -100, F_opt != -100)
	
	h = 4.1357e-15
	erg2eV = 6.24151e11
	Jy2erg_cm_s_Hz = 1e-23
	optical_energy = 2.4 ##eV
	
	opt_factor = Jy2erg_cm_s_Hz*erg2eV/h*optical_energy/1.e3/1.e6
	F_opt*= opt_factor
	fe_opt*= opt_factor
	
	F_opt_d*= opt_factor
	fe_opt_d*= opt_factor
	
	t = t[bool_arr]
	t_opt = t_opt[bool_arr]
	t_gam = t_gam[bool_arr]
	F_opt = F_opt[bool_arr]
	F_gam = F_gam[bool_arr]
	fe_opt = fe_opt[bool_arr]
	fe_d_gam = fe_d_gam[bool_arr]
	fe_u_gam = fe_u_gam[bool_arr]
	
	gam_opt_ratio = F_opt/F_gam
	ratio_err_u = ((fe_opt/F_gam)**2 + (F_opt/F_gam**2*fe_u_gam)**2)**0.5
	ratio_err_d = ((fe_opt/F_gam)**2 + (F_opt/F_gam**2*fe_d_gam)**2)**0.5

	# print(F_gam.shape, F_opt.shape)
	
	###
	### PLOT
	###
	font = {'family' : 'serif',
			'weight' : 'normal',
			'size'   : 12}

	matplotlib.rc('font', **font)
	matplotlib.rcParams['mathtext.fontset']='dejavuserif'

	cmap2 = plt.get_cmap('gist_stern')
	# cmap2= plt.get_cmap('gnuplot2')#'BuPu_r')
	fig = plt.figure()

	gs = gridspec.GridSpec(2, 1)
	gs.update(wspace=0.0, hspace=0.0)
	plt.subplots_adjust(hspace=0.001)

	ax0 = plt.subplot(gs[0,0])
	ax1 = plt.subplot(gs[1,0])


	for axis in ['top','bottom','left','right']:
		ax0.spines[axis].set_linewidth(1)
	ax0.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	ax0.tick_params(which='minor',width=0.25, length=5, direction='in')

	for axis in ['top','bottom','left','right']:
		ax1.spines[axis].set_linewidth(1)
	ax1.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	ax1.tick_params(which='minor',width=0.25, length=5, direction='in')

	ax0.yaxis.set_ticks_position('both')
	ax0.xaxis.set_ticks_position('both')
	ax0.minorticks_on()
	ax0.grid(True, color='grey', zorder=0, alpha=0.15)

	ax1.xaxis.set_ticks_position('both')
	ax1.yaxis.set_ticks_position('both')
	ax1.minorticks_on()
	ax1.grid(True, color='grey', zorder=0, alpha=0.15)


	MSIZE = 0.6
	ALPHA = 0.2
	ELINE = 0.5

	ax0.errorbar(t[gam_opt_ratio>0], gam_opt_ratio[gam_opt_ratio>0], yerr=[ratio_err_d[gam_opt_ratio>0],ratio_err_u[gam_opt_ratio>0]], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.35), zorder=21)
	ax0.errorbar(t[gam_opt_ratio<=0], abs(gam_opt_ratio[gam_opt_ratio<=0]), yerr=[ratio_err_d[gam_opt_ratio<=0],ratio_err_u[gam_opt_ratio<=0]], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.75), zorder=20)

	if optic_tag == 1:
		ax1.errorbar(t_opt, F_opt, yerr=fe_opt, fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.25), zorder=21)
		ax1.errorbar(t_opt_d, F_opt_d, yerr=fe_opt_d, fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.65), zorder=20)

	if gamma_tag == 1:
		ax1.errorbar(t_gam, F_gam, yerr=[fe_d_gam,fe_u_gam], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.2), zorder=21)
		ax1.errorbar(t_gam_d, F_gam_d, yerr=[fe_gam_dd,fe_gam_ud], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.6), zorder=20)

	if CHANGE_POINTS == 1:
		ax0.step(t_opt_d[cc_in_opt[0].astype(np.int)], cc_in_opt[1], where='post', color='orange', linewidth=0.2)
		ax1.step(t_gam_d[cc_in_gam[0].astype(np.int)], cc_in_gam[1], where='post', color='orange', linewidth=0.2)

	## Limits
	ratio_min = np.percentile(abs(gam_opt_ratio), 5)
	ratio_max = np.percentile(abs(gam_opt_ratio), 95)
	
	opt_min = min(np.percentile(abs(F_opt[F_opt!=-100]), 10), np.percentile(abs(F_opt_d[F_opt_d!=-100]), 10))
	opt_max = max(np.amax(F_opt),np.amax(F_opt_d))

	gam_min = min(np.percentile(F_gam[F_gam>0], 10), np.percentile(F_gam_d[F_gam_d>0], 10))
	gam_max = max(np.amax(F_gam),np.amax(F_gam_d))

	flux_min = min(opt_min, gam_min)
	flux_max = max(opt_max, gam_max)
	
	ax0.set_ylim([ratio_min/5.,ratio_max*5.])
	ax1.set_ylim([flux_min*0.66,flux_max*1.5])

	ax0.set_xlim([-100,4100])
	ax1.set_xlim([-100,4100])

	# ax0.set_xlim([2700,3000])
	# ax1.set_xlim([2700,3000])

	ax0.set_yscale('log')
	ax1.set_yscale('log')

	ax1.set_xlabel('Time [days]')
	ax1.set_ylabel(r'$E\Phi$ [MeV cm$^{-2}$ s$^{-1}$]')

	ax0.set_ylabel('Optical/Gamma Ratio')
	ax0.set_xticklabels([])


	plt.tight_layout()
	if ZOOM == 0:
		# plot_name = '../analysis/object'+str(obj_num).zfill(4)+'/plot_lc_object'+str(obj_num).zfill(4)+'_'
		plot_name = '../analysis/full_ratio_curves/plot_rc_object'+str(obj_num).zfill(4)+'_'
		for n in range(0,100):
			if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
				continue
			else:
				plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
				break
		plt.close()
	elif ZOOM == 1:
		ii_max = int(4000/500)
		for ii in range(0,ii_max):
			ax0.set_xlim([500*ii,500*ii+600])
			ax1.set_xlim([500*ii,500*ii+600])
			plot_name = '../analysis/full_ratio_curves_zoom/plot_rc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			for n in range(0,100):
				if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					continue
				else:
					plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					break

		plt.close()
		
	print('Plotted object'+str(obj_num).zfill(4))

