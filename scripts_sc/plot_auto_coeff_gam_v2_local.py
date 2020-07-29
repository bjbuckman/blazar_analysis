import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

obj_num = int(sys.argv[1])
file_num = int(sys.argv[2])

filename_err = '../../output/auto_gam/object'+str(obj_num).zfill(4)+'_auto_gam_stats'+str(file_num).zfill(3)+'.dat'
filename_sig = '../../output/autocorr_gam+gam/cross_stats_gam+gam_object'+str(obj_num).zfill(4)+'_'+str(file_num).zfill(3)+'.dat'

#stats
read_in = np.loadtxt(filename_err)

time_err = read_in[0]
data_err = read_in[1:]

cor_mean = data_err[0]
cor_err_d = cor_mean-data_err[1]
cor_err_u = data_err[2]-cor_mean

read_in = np.loadtxt(filename_sig)
time_sig = read_in[0]
data_sig = read_in[1:]

time_sig_pos_arr = time_sig >= 0.
time_sig_neg_arr = time_sig <= 0.

data_sig_pos = data_sig[:,time_sig_pos_arr]
data_sig_neg = data_sig[:,time_sig_neg_arr]
data_sig_neg = data_sig_neg[:,::-1]

time_sig = time_sig[time_sig_pos_arr]
data_sig = np.mean([data_sig_pos, data_sig_neg], axis=0)

sig_mean = data_sig[0]
sig_sig1_d = data_sig[1]
sig_sig1_u = data_sig[2]
sig_sig2_d = data_sig[3]
sig_sig2_u = data_sig[4]
sig_sig3_d = data_sig[5]
sig_sig3_u = data_sig[6]
sig_siga_d = data_sig[7]
sig_siga_u = data_sig[8]


sigma3_corr = cor_mean - cor_err_d > sig_sig3_u
sigma3_anti = cor_mean + cor_err_u < sig_sig3_d

sigma2_corr = cor_mean - cor_err_d > sig_sig2_u
sigma2_anti = cor_mean + cor_err_u < sig_sig2_d
sigma2_corr = np.logical_xor(sigma2_corr, sigma3_corr)
sigma2_anti = np.logical_xor(sigma2_anti, sigma3_anti)

time_sig3_corr = time_sig[sigma3_corr]
time_sig3_anti = time_sig[sigma3_anti]
time_sig2_corr = time_sig[sigma2_corr]
time_sig2_anti = time_sig[sigma2_anti]

###
### PLOT
###
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

cmap2= plt.get_cmap('gist_stern')
cmap = plt.get_cmap('gnuplot2')
fig = plt.figure()
	
f1, ax = plt.subplots(1,1)#, figsize=(18,6))

for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(1)
ax.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
ax.tick_params(which='minor',width=0.25, length=5, direction='in')

ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.minorticks_on()
# ax.grid(True)

MSIZE = 0.6
ALPHA = 0.3
ELINE = 0.5

ax.errorbar(time_err, cor_mean, yerr=[cor_err_d,cor_err_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)

alpha_fill = 0.25
lwidth_fill = 0.1
ax.fill_between(time_sig, sig_sig1_d, sig_sig1_u, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.914)
ax.fill_between(time_sig, sig_sig2_d, sig_sig2_u, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.913)
ax.fill_between(time_sig, sig_sig3_d, sig_sig3_u, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.912)
ax.fill_between(time_sig, sig_siga_d, sig_siga_u, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=0.911)

LWIDTH = 0.075
alpha_fill = 0.5
for tc in time_sig3_corr:
	ax.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
for tc in time_sig3_anti:
	ax.axvline(x=tc, color=cmap2(0.02), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.9)
	
for tc in time_sig2_corr:
	ax.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)
for tc in time_sig2_anti:
	ax.axvline(x=tc, color=cmap2(0.60), alpha=alpha_fill, linewidth=LWIDTH, zorder=0.8)

# ax.set_xscale('log')

ax.set_xlim([0,3000])
ax.set_ylim([-1,1])


ax.set_xlabel(r'Time Lag [days]')
ax.set_ylabel('Gamma-ray Auto-Corr Coeff')


plt.tight_layout()
plot_name = './plot_cc_object'+str(obj_num).zfill(4)+'_stats'+str(file_num).zfill(3)+'_'
for n in range(0,100):
	if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
		continue
	else:
		plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
		break
plt.close()
