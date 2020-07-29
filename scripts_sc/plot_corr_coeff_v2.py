import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


ZOOM = 1
object_arr = range(0,3)

cc_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/2001/output/cross_corr/'
cross_object_cc_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/2001/output/cross_object_gam+opt/'
cc_file_num = 0

for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]

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

	###
	### PLOT
	###
	font = {'family' : 'serif',
			'weight' : 'normal',
			'size'   : 12}

	matplotlib.rc('font', **font)

	cmap2 = plt.get_cmap('gist_stern')
	cmap  = plt.get_cmap('brg')#'gnuplot2')#'BuPu_r')
	fig = plt.figure()
		
	f1, ax1 = plt.subplots(1,1)#, figsize=(18,6))

	for axis in ['top','bottom','left','right']:
		ax1.spines[axis].set_linewidth(1)
	ax1.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	ax1.tick_params(which='minor',width=0.25, length=5, direction='in')

	ax1.yaxis.set_ticks_position('both')
	ax1.xaxis.set_ticks_position('both')
	ax1.minorticks_on()
	ax1.grid(True)

	MSIZE = 0.6
	ALPHA = 0.3
	ELINE = 0.5

	ax1.errorbar(time_err, cor_mean, yerr=[cor_err_d,cor_err_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
	ax1.set_xlim([-1999.999,2999.999])
	ax1.set_ylim([-0.999,0.999])
	
	# ax1.text(-1850., 0.85, r'CC', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	
	# ax1.text(150., -0.85, r'$\gamma$-ray lead$\rightarrow$', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	# ax1.text(-150., -0.85, r'$\leftarrow$Optical lead', verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
	
	ax1.set_ylabel(r'$r_s$')
	ax1.set_xlabel(r'Days [$\gamma$-ray leading]')
	# ax1.yaxis.set_label_position('right')
	
	alpha_fill = 0.2
	lwidth_fill = 0.1
	ax1.fill_between(time_sig, sig_sig1_d, sig_sig1_u, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.914)
	ax1.fill_between(time_sig, sig_sig2_d, sig_sig2_u, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.913)
	ax1.fill_between(time_sig, sig_sig3_d, sig_sig3_u, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.912)
	ax1.fill_between(time_sig, sig_siga_d, sig_siga_u, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.911)

	LWIDTH = 0.3
	alpha_fill = 0.6
	for tc in time_sig3_corr:
		ax1.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
	for tc in time_sig3_anti:
		ax1.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
		
	for tc in time_sig2_corr:
		ax1.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
	for tc in time_sig2_anti:
		ax1.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)


	plt.tight_layout()
	if ZOOM == 0:
		ax1.set_xlim([time_min,time_max])
		plot_name = '../analysis/cc/plot_cc_object'+str(obj_num).zfill(4)+'_'
		for n in range(0,100):
			if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
				continue
			else:
				plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
				break
		plt.close()
		
	elif ZOOM == 1:
		ii_max = int(3000/300)
		time_min = -1000.
		for ii in range(0,ii_max):
			ax1.set_xlim([300*ii+time_min,300*ii+400+time_min])
			plot_name = '../analysis/cc-zoom/plot_cc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			for n in range(0,100):
				if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					continue
				else:
					plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					break

		plt.close()
		
	elif ZOOM == 2:
		ii_max = int(4000/150)
		time_min = -1000.
		for ii in range(0,ii_max):
			ax1.set_xlim([150*ii+time_min,150*ii+200+time_min])
			plot_name = '../analysis/cc-zoom/plot_cc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			for n in range(0,100):
				if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					continue
				else:
					plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					break

		plt.close()
		
	print('Plotted object'+str(obj_num).zfill(4))

