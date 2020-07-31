#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import sys

from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.timeseries import LombScargle

# import tkinter
import matplotlib
# %matplotlib inline
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

import seaborn as sns
sns.set(style="darkgrid")

from scipy.stats import gaussian_kde


data_dir = './data/'

object_arr = ['J0102.8+5824', 'J0957.6+5523', 'J1503.7-5801', 'J1543.0+6130', 'J2015.5+3710']

for kk in range(0,len(object_arr)):  ## look at all objects 
    for mm in range(0,4): ## energy bins to look at
        object = object_arr[kk]
        ebin_number = mm 

        filename_events = data_dir+object+'/'+'events_4FGL_'+object+'_e'+str(ebin_number)+'.txt'
        filename_out7   = data_dir+object+'/'+'4FGL_'+object+'_out7_rois.txt'
        filename_out10  = data_dir+object+'/'+'out10_lc_e'+str(ebin_number)+'.txt'


        # In[ ]:


        '''
        out10 file
        day binning
        '''

        out10 = np.loadtxt(filename_out10)
        print('out10 fileshape: '+str(out10.shape))

        time_bin_start = out10[:,0]
        time_bin_end   = time_bin_start + out10[:,1]

        counts_bin     = out10[:,2]
        exposure_bin   = out10[:,4]

        total_counts_out10 = np.sum(counts_bin)
        print('Total counts from out10: '+str(int(total_counts_out10)))

        ## sampling array
        sample_rate = 5
        sample_offset = 0
        time_bin_start_samp = time_bin_start[sample_offset:-sample_rate:sample_rate]
        time_bin_end_samp = time_bin_end[sample_offset+sample_rate-1::sample_rate]
        num_samp_bins = len(time_bin_start_samp)

        time_bin_samp_mets = (time_bin_start_samp + time_bin_end_samp)/2
        time_bin_samp_days = time_bin_samp_mets/(60*60*24) + 2451910.5
        time_bin_samp_days-= time_bin_samp_days.min()
        # print(time_bin_start_samp[:5])
        # print(time_bin_end_samp[:5])

        counts_bin_samp = np.zeros(num_samp_bins)
        exposure_bin_samp = np.zeros(num_samp_bins)
        for ii in range(0,num_samp_bins):
            samp_arr = np.logical_and(time_bin_start>=time_bin_start_samp[ii], time_bin_end<=time_bin_end_samp[ii])
            counts_bin_samp[ii] = np.sum(counts_bin[samp_arr])
            exposure_bin_samp[ii] = np.sum(exposure_bin[samp_arr])
        print('Number condensed bins: '+str(len(time_bin_start_samp)))

        ## renormalize exposure to counts
        counts_bin_samp_avg = np.mean(counts_bin_samp[ii])
        exposure_bin_samp_avg = np.mean(exposure_bin_samp)

        # exposure_bin_samp*= counts_bin_samp_avg/exposure_bin_samp_avg

        # print(counts_bin_samp[:5])
        # print(exposure_bin_samp[:5])


        # In[ ]:


        '''
        out7 file
        '''

        out7 = np.loadtxt(filename_out7)
        print('out7 fileshape: '+str(out7.shape))

        bin_energy = out7[ebin_number,1]
        bin_roi    = out7[ebin_number,2]
        print('Bin Energy: '+str(bin_energy))
        print('Bin ROI: '+str(bin_roi))


        # In[ ]:


        '''
        EVENTS FILE
        0 - Arrival time (METS)
        1 - energy (MeV)
        2 - RA
        3 - DEC
        4 - Long
        5 - Lat
        6 - Theta (instrumental)
        7 - SkyCoordinate (instrumental)
        8 - Zenith angle (earth)
        9 - Earth azimuth (Earth)
        '''

        read_in = np.loadtxt(filename_events)

        time_mets = read_in[:,0]
        total_events = len(time_mets)

        ## time in days
        time = read_in[:,0]/(60*60*24) + 2451910.5
        time-= np.amin(time)

        energy = read_in[:,1]


        # In[ ]:


        print('Data shape: '+str(read_in.shape))
        print('First Event: '+str(read_in[0]))


        # In[ ]:
        ## object_arr = ['J0102.8+5824', 'J0957.6+5523', 'J1503.7-5801', 'J1543.0+6130', 'J2015.5+3710']
        if object == 'J0102.8+5824':
            RA_obj = 1. *u.hour + 2.8 *u.min
            DE_obj = +58.24 *u.deg
        elif object == 'J0957.6+5523':
            RA_obj = 9. *u.hour + 57.6 *u.min
            DE_obj = 55.23 *u.deg
        elif object == 'J1503.7-5801':
            RA_obj = 15. *u.hour + 3.7 *u.min
            DE_obj = -58.01 *u.deg
        elif object == 'J1543.0+6130':
            RA_obj = 15. *u.hour + 43.0 *u.min
            DE_obj = 61.30 *u.deg
        elif object == 'J2015.5+3710':
            RA_obj = 20. *u.hour + 15.5 *u.min
            DE_obj = 37.10 *u.deg
        
        # RA_obj = 15 *u.hour + 3.7 *u.min
        # DE_obj = -58.01 *u.deg

        c_obj = SkyCoord(ra=RA_obj, dec=DE_obj)
        print(c_obj)


        # In[ ]:


        RA = read_in[:,2] *u.deg
        DE = read_in[:,3] *u.deg

        c_photon = SkyCoord(ra=RA, dec=DE)
        c_photon_dra, c_photon_dde = c_photon.spherical_offsets_to(c_obj)

        dRA = c_photon_dra.deg # relative RA
        dDE = c_photon_dde.deg # relative DE

        c_sep = c_photon.separation(c_obj)
        angle_from_obj = c_sep.deg
        max_angle_from_obj = np.amax(angle_from_obj)
        print('Max angular distance: '+str(round(max_angle_from_obj,2)))


        # In[ ]:


        ''' Making pandas DataFrame'''

        d = {'time_mets':time_mets, 'time_days':time, 'dRA':dRA, 'dDE':dDE, 'energy':energy, 
             'angle_from_obj':angle_from_obj}
        df = pd.DataFrame(data=d)
        print(df.head(10))
        # print(df.loc[:10])


        # In[ ]:


        XLIM = [-max_angle_from_obj,max_angle_from_obj]
        YLIM = [-max_angle_from_obj,max_angle_from_obj]

        # sns.jointplot(x='dRA',y='dDE', data=df.loc[:100], kind='kde', xlim=XLIM, ylim=YLIM) 

        # ''' Animation '''


        # def get_data(ii=0):
            # bin_array = np.logical_and(time_mets>=time_bin_start_samp[ii], time_mets<=time_bin_end_samp[ii])
            # data = df.loc[bin_array]
            # x = data['dRA'].values
            # y = data['dDE'].values
            # return x,y

        # x,y = get_data()
        # g = sns.JointGrid(x=x, y=y)
        # # g = sns.jointplot(x='dRA',y='dDE', data=data, kind='kde', xlim=XLIM, ylim=YLIM) 

        # lim = (-1.2*max_angle_from_obj,1.2*max_angle_from_obj)

        # def prep_axes(g, xlim, ylim):
            # g.ax_joint.clear()
            # g.ax_joint.set_xlim(xlim)
            # g.ax_joint.set_ylim(ylim)
            # g.ax_marg_x.clear()
            # g.ax_marg_x.set_xlim(xlim)
            # g.ax_marg_y.clear()
            # g.ax_marg_y.set_ylim(ylim)
            # plt.setp(g.ax_marg_x.get_xticklabels(), visible=False)
            # plt.setp(g.ax_marg_y.get_yticklabels(), visible=False)
            # plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=False)
            # plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=False)
            # plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=False)
            # plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=False)
            # plt.setp(g.ax_marg_x.get_yticklabels(), visible=False)
            # plt.setp(g.ax_marg_y.get_xticklabels(), visible=False)

        # def animate(i):
            # g.x, g.y = get_data(i)
            # prep_axes(g, lim, lim)
            # g.plot_joint(sns.kdeplot, bw=0.2, shade=True) #, cmap="Purples_d")
            # g.plot_marginals(sns.kdeplot, bw=0.2, shade=True) #, color="m", shade=True)


        # frames=np.arange(1,100)
        # ani = animation.FuncAnimation(g.fig, animate, frames=frames, repeat=True)
        # ani.save('test2.mp4')
        # # plt.show()
        # # In[ ]:


        max_angle = 1.2*max_angle_from_obj

        num_along_dim = 100
        var1 = np.linspace(-max_angle, max_angle, num=num_along_dim)
        var2 = np.linspace(-max_angle, max_angle, num=num_along_dim)

        grid_var1, grid_var2 = np.meshgrid(var1, var2)
        positions = np.vstack([grid_var1.flatten(), grid_var2.flatten()])
        # print(positions)

        def get_data(ii=0):
            bin_array = np.logical_and(time_mets>=time_bin_start_samp[ii], time_mets<=time_bin_end_samp[ii])
            if np.sum(bin_array) >= 3:
                data = df.loc[bin_array]
                x = data['dRA'].values
                y = data['dDE'].values
                return np.array([x,y]), 1
            else:
                return np.array([0,0]), 0

        num_bins_analyze = num_samp_bins
        photon_grid = np.zeros([num_bins_analyze, num_along_dim, num_along_dim])
        for ii in range(0,num_bins_analyze):
            bin_photons, is_points = get_data(ii)
            if is_points == 1:
                num_photons = bin_photons.shape[1]
        #         print(bin_photons)

                gkde = gaussian_kde(bin_photons, bw_method=0.3)
                pdf = gkde.evaluate(positions)
                photon_grid[ii] = num_photons*pdf.reshape([num_along_dim,num_along_dim])
            else:
                continue

        all_photons = np.array([df['dRA'].values, df['dDE'].values])
        # all_y = df['dDE'].values
        num_photons = total_events
        gkde = gaussian_kde(all_photons, bw_method=0.1)
        pdf = gkde.evaluate(positions)
        all_photon_grid = num_photons*pdf.reshape([num_along_dim,num_along_dim])

        # print(photon_grid)
            
        # num_points = get_data().shape[1]
        # # print(get_data().T)
        # gkde = gaussian_kde(get_data(), bw_method=0.3)

        # pdf = gkde.evaluate(positions)
        # # print(blah*num_points)


        # In[ ]:


        ''' Finding power at 53-days '''

        num_freq = 10
        period = np.linspace(52, 54, num=num_freq)
        freq = 1/period

        power_array = np.zeros([num_along_dim, num_along_dim, num_freq])
        for ii in range(0,num_along_dim):
            for jj in range(0,num_along_dim):
        #         print(time_bin_samp_days.shape, photon_grid[:,ii,jj].shape)
        #         blah = LombScargle(time_bin_samp_days[:num_bins_analyze], photon_grid[:,ii,jj]).power(freq)
                power_array[ii,jj] = LombScargle(time_bin_samp_days[:num_bins_analyze], photon_grid[:,ii,jj]).power(freq)

        power_array = np.max(power_array,axis=-1)

        exposure_power_array = LombScargle(time_bin_samp_days[:num_bins_analyze], exposure_bin_samp[:num_bins_analyze]).power(freq)
        exposure_power = np.max(exposure_power_array)
        print('Exposure Power array: '+str(exposure_power_array))

        counts_power_array = LombScargle(time_bin_samp_days[:num_bins_analyze], counts_bin_samp[:num_bins_analyze]).power(freq)
        counts_power = np.max(counts_power_array)
        print('Counts Power array: '+str(counts_power_array))


        # In[ ]:


        ''' Plotting power '''
        ###
        ### PLOT
        ###
        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 12}

        matplotlib.rc('font', **font)

        cmap2 = plt.get_cmap('gist_stern_r')
        cmap  = plt.get_cmap('Greys')#'gnuplot2')#'BuPu_r')
        fig = plt.figure()

        gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0,0])

        for axis in ['top','bottom','left','right']:
            ax0.spines[axis].set_linewidth(1)
        ax0.tick_params(which='major',width=0.5, length=8, labelsize='small', direction='in')
        ax0.tick_params(which='minor',width=0.25, length=5, direction='in')

        grid_dist = (grid_var1**2 + grid_var2**2)**0.5
        in_roi = grid_dist < bin_roi
        print(np.sum(in_roi))

        max_power = np.max(power_array[in_roi])
        min_power = np.min(power_array[in_roi])
        LEVELS = np.linspace(min_power, max_power, num=21)

        power_array_masked = np.ma.masked_where(in_roi == False, power_array)

        CF1 = ax0.contourf(grid_var1, grid_var2, power_array, cmap=cmap2, levels=LEVELS, zorder=9)
        EXPO = ax0.contour(grid_var1, grid_var2, power_array, colors='r', levels=[exposure_power], zorder=10)

        cbar = fig.colorbar(CF1)
        cbar.set_label('Power at 53 days')

        C1 = ax0.contour(grid_var1, grid_var2, all_photon_grid, levels=10, cmap=cmap, zorder=9.5)

        ax0.set_aspect('equal')
        ax0.text(0.05, 0.95, 'Exposure Power = '+str(round(exposure_power,4)), transform=ax0.transAxes, verticalalignment='center', horizontalalignment='left', zorder=11)
        ax0.text(0.05, 0.05, 'Counts Power = '+str(round(counts_power,4)), transform=ax0.transAxes, verticalalignment='center', horizontalalignment='left', zorder=11)

        ax0.set_xlabel('Relative RA [deg]')
        ax0.set_ylabel('Relative DEC [deg]')


        # plt.show()

        plot_name = './map_photon_power_'+object+'_e'+str(ebin_number)+'_'
        for n in range(0,100):
            if os.path.isfile(plot_name+str(n).zfill(2)+'.png'):
                continue
            else:
                plt.savefig(plot_name+str(n).zfill(2)+'.png', bbox_inches='tight', dpi=400)
                break
        plt.close()



