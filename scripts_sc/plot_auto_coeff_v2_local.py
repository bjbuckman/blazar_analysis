import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

obj_num = int(sys.argv[1])
file_num = int(sys.argv[2])

filename = './object_'+str(obj_num).zfill(4)+'_auto_gam_stats'+str(file_num).zfill(3)+'.dat'

#stats
read_in = np.loadtxt(filename)

time = read_in[0]
data = read_in[1:]

cor_mean = data[0]
cor_err_d = cor_mean-data[1]
cor_err_u = data[2]-cor_mean

sig_mean = data[9]
sig_sig1_d = data[10]
sig_sig1_u = data[11]
sig_sig2_d = data[12]
sig_sig2_u = data[13]
sig_sig3_d = data[14]
sig_sig3_u = data[15]
sig_siga_d = data[16]
sig_siga_u = data[17]

###
### PLOT
###
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

cmap = plt.get_cmap('gist_stern')
cmap2= plt.get_cmap('gnuplot2')
fig = plt.figure()
	
f1, ax = plt.subplots(1,1)#, figsize=(18,6))

for axis in ['top','bottom','left','right']:
	ax.spines[axis].set_linewidth(1)
ax.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
ax.tick_params(which='minor',width=0.25, length=5, direction='in')

ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.minorticks_on()
ax.grid(True)

MSIZE = 0.6
ALPHA = 0.3
ELINE = 0.5

ax.errorbar(time, cor_mean, yerr=[cor_err_d,cor_err_u], fmt='o', markersize=MSIZE, alpha=ALPHA, elinewidth=ELINE, capsize=0.5, capthick=0.5, color='k', zorder=21)

alpha_fill = 0.25
lwidth_fill = 0.1
ax.fill_between(time, sig_sig1_d, sig_sig1_u, facecolor=cmap2(0.05), edgecolor=cmap2(0.05), alpha=alpha_fill, linewidth=lwidth_fill, zorder=20)
ax.fill_between(time, sig_sig2_d, sig_sig2_u, facecolor=cmap2(0.15), edgecolor=cmap2(0.15), alpha=alpha_fill, linewidth=lwidth_fill, zorder=19)
ax.fill_between(time, sig_sig3_d, sig_sig3_u, facecolor=cmap2(0.25), edgecolor=cmap2(0.25), alpha=alpha_fill, linewidth=lwidth_fill, zorder=18)
ax.fill_between(time, sig_siga_d, sig_siga_u, facecolor=cmap2(0.35), edgecolor=cmap2(0.35), alpha=alpha_fill, linewidth=lwidth_fill, zorder=17)

ax.set_xlim([-300,300])
ax.set_ylim([-1,1])


ax.set_xlabel(r'Time difference ($t_\mathrm{gamma}-t_\mathrm{optical}$) [days]')
ax.set_ylabel('Corr Coeff')


plt.tight_layout()
plot_name = './plot_cc_object'+str(obj_num).zfill(4)+'_stats'+str(file_num).zfill(3)+'_'
for n in range(0,100):
	if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
		continue
	else:
		plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
		break
plt.close()
