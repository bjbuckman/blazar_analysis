import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys


ZOOM = 1
object_arr = range(0,100)

autocorr_gam_err_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/2001/output/auto_gam_v0/'
autocorr_gam_sig_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/2001/output/autocorr_gam+gam_v0/'
gam_file_num = 0

for ii in range(0,len(object_arr)):
	obj_num = object_arr[ii]

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
		
		sp = np.fft.fft(cor_mean_gam)
		sp_l = np.fft.fft(cor_mean_gam-cor_err_d_gam)
		sp_u = np.fft.fft(cor_mean_gam+cor_err_u_gam)
		freq = np.fft.fftfreq(len(cor_mean_gam), d=0.4)
		
		
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
		
	gs = gridspec.GridSpec(2, 2)
	gs.update(wspace=0.0, hspace=0.0)
	plt.subplots_adjust(hspace=0.001)
	
	ax0 = plt.subplot(gs[0,0:2])
	ax1 = plt.subplot(gs[1,0:2])

	for axis in ['top','bottom','left','right']:
		ax0.spines[axis].set_linewidth(1)
	ax0.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	ax0.tick_params(which='minor',width=0.25, length=5, direction='in')

	for axis in ['top','bottom','left','right']:
		ax1.spines[axis].set_linewidth(1)
	ax1.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
	ax1.tick_params(which='minor',width=0.25, length=5, direction='in')

	# ax1.yaxis.set_ticks_position('both')
	# ax1.xaxis.set_ticks_position('both')
	# ax1.minorticks_on()
	# ax1.grid(True)

	MSIZE = 0.6
	ALPHA = 0.3
	ELINE = 0.5

	if auto_gam_err_tag == 1:
		ax0.errorbar(time_gam_err, cor_mean_gam, yerr=[cor_err_d_gam,cor_err_u_gam], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)
		
		ax0.set_xlim([-0.001,2999.999])
		ax0.set_ylim([-0.999,0.999])
		
		# ax11.set_xticklabels([])
		# ax11.set_yticklabels([])
		
		# ax01.yaxis.set_ticks_position('right')
		
		# ax01.xaxis.set_ticks_position('top')
		# plt.setp(ax01.get_xticklabels(), ha="left", rotation=45)
		# ax01.set_xlabel(r'Days')
		# ax01.xaxis.set_label_position('top') 
		# ax01.set_ylabel(r'$r_s$')
		# ax01.yaxis.set_label_position('right')
		
		# ax01.text(-1850., 0.85, r'AC', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		# ax01.text(150., -0.85, r'$\gamma$-ray$\rightarrow$', verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		# ax01.text(-150., -0.85, r'$\leftarrow$Optical', verticalalignment='center', horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=2))
		
		# ax02 = plt.subplot(gs[0:2,4:6])
		# ax02.plot(np.fft.fftfreq(len(time_gam_err), d=0.4), np.fft.fft(cor_mean_gam))
		# # ax02.set_xscale('log')
		# ax02.set_xlim([1.7e-2,2.e-2])
		
		ax1.plot(1/freq, abs(sp), color='k', alpha=ALPHA)
		ax1.plot(1/freq, sp.real, color='b', alpha=ALPHA)
		ax1.plot(1/freq, sp.imag, color='r', alpha=ALPHA)
		
		# ax1.plot(1/freq, sp_l.real, color='b', alpha=ALPHA)
		# ax1.plot(1/freq, sp_l.imag, color='g', alpha=ALPHA)
		
		# ax1.plot(1/freq, sp_u.real, color='b', alpha=ALPHA)
		# ax1.plot(1/freq, sp_u.imag, color='g', alpha=ALPHA)
		
		ax1.set_xlim(40.,60.)
		# y0_max = np.amax()
		# ax1.set_ylim(-100.,100.)
		
		# plt.setp(ax0.get_xticklabels(), ha='right', rotation=45)
		ax0.zorder=1000
		
	if auto_gam_sig_tag == 1:
		alpha_fill = 0.2
		lwidth_fill = 0.1
		ax0.fill_between(time_gam_sig, sig_sig1_d_gam, sig_sig1_u_gam, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.914)
		ax0.fill_between(time_gam_sig, sig_sig2_d_gam, sig_sig2_u_gam, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.913)
		ax0.fill_between(time_gam_sig, sig_sig3_d_gam, sig_sig3_u_gam, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.912)
		ax0.fill_between(time_gam_sig, sig_siga_d_gam, sig_siga_u_gam, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.911)
		
		LWIDTH = 0.075
		alpha_fill = 0.4
		for tc in time_sig3_corr_gam:
			ax0.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
		for tc in time_sig3_anti_gam:
			ax0.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
			
		for tc in time_sig2_corr_gam:
			ax0.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
		for tc in time_sig2_anti_gam:
			ax0.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
			


	plt.tight_layout()
	# if ZOOM == 0:
	# ax1.set_xlim([time_min,time_max])
	plot_name = '../analysis/ac_gam_fft/plot_ac_gam_fft_object'+str(obj_num).zfill(4)+'_'
	for n in range(0,100):
		if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
			continue
		else:
			plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
			break
	plt.close()
		
	# elif ZOOM == 1:
		# ii_max = int(3000/300)
		# time_min = -1000.
		# for ii in range(0,ii_max):
			# ax1.set_xlim([300*ii+time_min,300*ii+400+time_min])
			# plot_name = '../analysis/cc-zoom/plot_cc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			# for n in range(0,100):
				# if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					# continue
				# else:
					# plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					# break

		# plt.close()
		
	# elif ZOOM == 2:
		# ii_max = int(4000/150)
		# time_min = -1000.
		# for ii in range(0,ii_max):
			# ax1.set_xlim([150*ii+time_min,150*ii+200+time_min])
			# plot_name = '../analysis/cc-zoom/plot_cc_object'+str(obj_num).zfill(4)+'_zoom'+str(ii).zfill(2)+'_'
			# for n in range(0,100):
				# if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
					# continue
				# else:
					# plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
					# break

		# plt.close()
		
	print('Plotted object'+str(obj_num).zfill(4))

