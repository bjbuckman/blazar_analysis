import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from matplotlib import gridspec

# obj_num = 11
ZOOM = 2

MODEL = 3

gam_ver = 3
object_arr = range(26,29)
# object_arr = range(0,3131)
# object_arr = [21, 24, 42, 46, 87, 164, 213, 312, 342, 344, 347, 392, 393, 476, 483, 574, 694, 715]

for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]	
		
	#time to average data over
	dt = 0.45

	###
	###
	### Getting full time array
	###
	###

	gamma_tag = 1
	try:
		gamma_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.npy'
		read_in = np.load(gamma_file)
	except:
		gamma_tag = 0
		print('No data for object'+str(obj_num).zfill(4))
		continue
	
	# print(read_in.shape)
	time = read_in[:,0,0]/(60*60*24) + 2451910.5 -2450000.
	time_min = round(time.min()) 
	time_max = round(time.max())

	# time_max = round(max(time_max_opt,time_max_gam))
	# time_min = round(min(time_min_opt,time_min_gam))

	t = np.arange(time_min,time_max,0.2) -time_min
	tnum = len(t)
	enum = read_in.shape[1]


	###
	###
	### GAMMA-RAY
	###
	###
	
	time = read_in[:,0,0]/(60*60*24) + 2451910.5 -2450000.
	flux = read_in[:,:,1]
	flux_ll = read_in[:,:,2]
	flux_ul = read_in[:,:,3]
	expo = read_in[:,:,4] 
	counts = read_in[:,:,5] 
	try:
		events = read_in[:,:,6]
	except:
		print('no_events')
		continue
	
	time_tot = time
	flux_tot = np.sum(flux[:,:], axis=1)
	flux_err_d = np.sqrt(np.sum(flux_ll[:,:]**2, axis=1))
	flux_err_u = np.sqrt(np.sum(flux_ul[:,:]**2, axis=1))
	
	
	# try:
	# gamma_file = '../data/gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.dat'
	# read_in = np.loadtxt(gamma_file).T
	# time_tot = read_in[0]/(60*60*24) + 2451910.5 -2450000.
	# flux_tot = read_in[1]
	# gamma_lerr_data = read_in[2]
	# gamma_uerr_data = read_in[3]
	# flux_err_d = gamma_lerr_data
	# flux_err_u = gamma_uerr_data
	# flux_err_2d = flux_err_d**2
	# flux_err_2u = flux_err_u**2
	
	# time = time[:-40]
	# expo = expo[40:]
	# counts = counts[:-40]
	
	# counts_expo = expo/counts
	# gamma_lerr_data = read_in[:,:,2]
	# gamma_uerr_data = read_in[:,:,3]
	# flux_err_d = gamma_lerr_data
	# flux_err_u = gamma_uerr_data
	# flux_err_2d = flux_err_d**2
	# flux_err_2u = flux_err_u**2

	##
	## Regridding over dt days
	## Doing this for simplicity
	# Assuming independence of measurements,
	# We just avg mu values and inv-sum sigma
	# f = np.zeros(tnum)-100
	# fe_d = np.zeros(tnum)-1
	# fe_u = np.zeros(tnum)-1

	# dtf = 0.45
	# for ii in range(0,tnum):
		# for jj in range(0,enum):
			# t_arr = np.logical_and( time<=t[ii]+dt, time>=t[ii]-dt)
			# t_arr = np.logical_and( t_arr, flux>0)
			# t_arrf = np.logical_and( time<=t[ii]+dtf, time>=t[ii]-dtf ) 
			# t_arrf = np.logical_and( t_arrf, flux>0)
			# if np.any(t_arr) and np.any(t_arrf):
				# f[ii] = np.mean(flux[t_arr])
				# fe_d[ii] = (np.sum(flux_err_2d[t_arr])/np.sum(t_arr))**0.5
				# fe_u[ii] = (np.sum(flux_err_2u[t_arr])/np.sum(t_arr))**0.5

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

	# print(time.shape)

	###
	### PLOT
	###
	font = {'family' : 'serif',
			'weight' : 'normal',
			'size'   : 12}

	matplotlib.rc('font', **font)
	matplotlib.rcParams['mathtext.fontset']='dejavuserif'

	cmap = plt.get_cmap('gist_stern')
	cmap2= plt.get_cmap('gnuplot2')#'BuPu_r')
	fig = plt.figure()

	gs = gridspec.GridSpec(4, 1)
	gs.update(wspace=0.0, hspace=0.0)
	plt.subplots_adjust(hspace=0.001)

	# ax0 = plt.subplot(gs[0,0])
	ax1 = plt.subplot(gs[0,0])
	ax2 = plt.subplot(gs[1,0])
	ax3 = plt.subplot(gs[2,0])
	ax4 = plt.subplot(gs[3,0])

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
	ALPHA = 0.7
	ELINE = 0.5

	if gamma_tag == 1:
		for jj in range(0,4):
			enum=5
			factor = 1.# 10**(2*jj)
			
			# ax0.scatter(time[:], expo[:,jj]*factor, alpha=ALPHA, color=cmap2(jj/enum), s=0.2)
			ax1.scatter(time[:], counts[:,jj]*factor, alpha=ALPHA, color=cmap2(jj/enum), s=0.2)
			ax2.scatter(time[:], events[:,jj]*factor, alpha=ALPHA, color=cmap2(jj/enum), s=0.2)
			ax3.errorbar(time[:], flux[:,jj]*factor, yerr=[flux_ll[:,jj],flux_ul[:,jj]], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(jj/enum), zorder=20)
			
				
		ax4.errorbar(time_tot, flux_tot, yerr=[flux_err_d,flux_err_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color=cmap2(0.5), zorder=20)
		ax4.hlines([5.e-6, 2.e-5, 4.e-5, 8.e-5, 2.e-4, 4.e-4], xmin=time_min, xmax=time_max, linestyles='dashed', color='k', alpha=0.2)
		ax4.hlines([1.e-5, 1.e-4], xmin=time_min, xmax=time_max, linestyles='solid', color='k', alpha=0.2)

	# if CHANGE_POINTS == 1:
		# ax0.step(t_opt_d[cc_in_opt[0].astype(np.int)], cc_in_opt[1], where='post', color='orange', linewidth=0.2)
		# ax1.step(t_gam_d[cc_in_gam[0].astype(np.int)], cc_in_gam[1], where='post', color='orange', linewidth=0.2)

	## Limits
	# opt_min = min(np.percentile(abs(F_opt[F_opt!=-100]), 10), np.percentile(abs(F_opt_d[F_opt_d!=-100]), 10))
	# opt_max = max(np.amax(F_opt),np.amax(F_opt_d))

	# gam_min = min(np.percentile(F_gam[F_gam>0], 10), np.percentile(F_gam_d[F_gam_d>0], 10))
	# gam_max = max(np.amax(F_gam),np.amax(F_gam_d))
	
	# plot_min = np.percentile(counts[:,3], 5)/2.
	# plot_max = np.percentile(counts[:,3], 95)*2.
	# ax0.set_ylim([plot_min, plot_max])
	# ax1.set_ylim([gam_min*0.66,gam_max*1.5])

	# ax0.set_xlim([-100,4100])
	# ax0.set_xlim([3000,3500])
	# ax0.set_xlim([-100,4100])

	# ax0.set_ylim([min,])
	# ax1.set_xlim([2700,3000])

	# ax0.set_yscale('log')
	ax2.set_yscale('log')
	ax3.set_yscale('log')
	ax4.set_yscale('log')

	ax2.set_ylim([1.e-1,1.e1])
	
	ax1.set_xticklabels([])
	ax2.set_xticklabels([])
	ax3.set_xticklabels([])
	
	
	# ax4.set_ylim([min(flux_tot)/2,max(flux_tot)*2])
	# ax0.set_xlabel('Time [days]')
	
	# if MODEL == 0:
		# ax0.set_ylabel(r'Exposure [cm$^{-2}$ s$^{-1}$]')
	# elif MODEL == 1:
		# ax0.set_ylabel(r'Counts []')
	# elif MODEL == 2:
		# ax0.set_ylabel(r'Exposure/(Counts+1)')
	# elif MODEL == 3:
		# ax0.set_ylabel(r'Normalized Counts and Exposure')

	# ax0.set_ylabel('$F$ [mJy]')
	# ax0.set_xticklabels([])


	plt.tight_layout()
	if ZOOM == 0:
		ax1.set_xlim([time_min,time_max])
		ax2.set_xlim([time_min,time_max])
		ax3.set_xlim([time_min,time_max])
		ax4.set_xlim([time_min,time_max])
		# plot_name = '../analysis/object'+str(obj_num).zfill(4)+'/plot_lc_object'+str(obj_num).zfill(4)+'_'
		plot_name = '../analysis/full_event_expo-flux/plot_lc_object'+str(obj_num).zfill(4)+'_'
		for n in range(0,100):
			if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
				continue
			else:
				plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
				break
		plt.close()
	elif ZOOM == 1:
		ii_max = int(4000/300)
		for ii in range(0,ii_max):
			# ax0.set_xlim([300*ii,300*ii+400])
			ax1.set_xlim([300*ii+time_min,300*ii+400+time_min])
			ax2.set_xlim([300*ii+time_min,300*ii+400+time_min])
			ax3.set_xlim([300*ii+time_min,300*ii+400+time_min])
			ax4.set_xlim([300*ii+time_min,300*ii+400+time_min])
			# plot_name = '../analysis/object'+str(obj_num).zfill(4)+'/plot_lc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			plot_name = '../analysis/full_event_expo-flux_zoom/plot_lc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			for n in range(0,100):
				if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					continue
				else:
					plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					break

		plt.close()
		
	elif ZOOM == 2:
		ii_max = int(4000/150)
		for ii in range(0,ii_max):
			# ax0.set_xlim([300*ii,300*ii+400])
			ax1.set_xlim([150*ii+time_min,150*ii+200+time_min])
			ax2.set_xlim([150*ii+time_min,150*ii+200+time_min])
			ax3.set_xlim([150*ii+time_min,150*ii+200+time_min])
			ax4.set_xlim([150*ii+time_min,150*ii+200+time_min])
			# plot_name = '../analysis/object'+str(obj_num).zfill(4)+'/plot_lc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			plot_name = '../analysis/full_event_expo-flux_zoom/plot_lc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			for n in range(0,100):
				if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					continue
				else:
					plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					break

		plt.close()
		
	print('Plotted object'+str(obj_num).zfill(4))

