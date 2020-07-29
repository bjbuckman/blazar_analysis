import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.cosmology import WMAP9 as cosmo

asa_dir = './asas-sn/'
opt_dir = './optical/'

gam_ver = 3

input_csv = './fermi_4FGL_associations_ext_GRPHorder+analysis1.csv'

data_in = pd.read_csv(input_csv)
# data_in = data_in.drop(columns='Unnamed: 0') ## delete redundant column
data_in.drop(columns='Unnamed: 0', inplace=True)

data_in['redshift'] = pd.to_numeric(data_in['redshift'], errors='coerce') 

peak_input = './asu.tsv'
peak_data = pd.read_csv(peak_input, sep=';', header=77)
peak_data = peak_data.iloc[2:]
peak_data = peak_data[['3FGL', 'C']]

## New columns

## lum_distance
data_in['lum_distance'] = cosmo.luminosity_distance(data_in['redshift'])
data_in['synch_peak'] = '-'

## optical
data_in['opt_flux_average'] = np.nan
data_in['opt_flux_average_err'] = np.nan

## gamma-ray
data_in['gam_flux_median'] = np.nan
# data_in['gam_flux_median_lerr'] = np.nan
# data_in['gam_flux_median_uerr'] = np.nan
data_in['gam_chi2_var'] = np.nan
data_in['gam_var_amplitude'] = np.nan
data_in['gam_doubling_time'] = np.nan
data_in['gam_doubling_time_err'] = np.nan

## gamma-ray total
data_in['gam_flux_total'] = np.nan
data_in['gam_flux_total_lerr'] = np.nan
data_in['gam_flux_total_uerr'] = np.nan

## gamma-ray total WEIGHTED AVERAGE
# data_in['gam_flux_total_wa'] = np.nan
# data_in['gam_flux_total_lerr_wa'] = np.nan
# data_in['gam_flux_total_uerr_wa'] = np.nan
data_in['gam_flux_total_ns'] = np.nan ## signal counts/exposure
data_in['gam_flux_total_cb'] = np.nan ## total counts - background/exposure
data_in['gam_flux_total_ba'] = np.nan
data_in['gam_flux_total_err_ba'] = np.nan

data_in['gam_photon_flux_total'] = np.nan
data_in['gam_photon_flux_total_lerr'] = np.nan
data_in['gam_photon_flux_total_uerr'] = np.nan
data_in['gam_photon_flux_total_ns'] = np.nan
data_in['gam_photon_flux_total_cb'] = np.nan
data_in['gam_photon_flux_total_ba'] = np.nan
data_in['gam_photon_flux_total_err_ba'] = np.nan

## gamma-ray total spectral
data_in['beta_total'] = np.nan
data_in['beta_total_lerr'] = np.nan
data_in['beta_total_uerr'] = np.nan

data_in['beta_low_total'] = np.nan
data_in['beta_low_total_lerr'] = np.nan
data_in['beta_low_total_uerr'] = np.nan

data_in['beta_high_total'] = np.nan
data_in['beta_high_total_lerr'] = np.nan
data_in['beta_high_total_uerr'] = np.nan

data_in['curve_total'] = np.nan
data_in['curve_total_lerr'] = np.nan
data_in['curve_total_uerr'] = np.nan

## ratio
data_in['opt_gam_ratio'] = np.nan
data_in['opt_gam_ratio_lerr'] = np.nan
data_in['opt_gam_ratio_uerr'] = np.nan

## timescales
data_in['timescale_opt'] = np.nan
data_in['timescale_opt_lerr'] = np.nan
data_in['timescale_opt_uerr'] = np.nan
data_in['timescale_gam'] = np.nan
data_in['timescale_gam_lerr'] = np.nan
data_in['timescale_gam_uerr'] = np.nan

## cross correlation at t=0
data_in['cross_corr_t0'] = np.nan
data_in['cross_corr_t0_lerr'] = np.nan
data_in['cross_corr_t0_uerr'] = np.nan
data_in['cross_corr_sig_t0'] = np.nan
data_in['cross_corr_sig_t0_lerr'] = np.nan
data_in['cross_corr_sig_t0_uerr'] = np.nan


## constants
h = 4.1357e-15
erg2eV = 6.24151e11
Jy2erg_cm_s_Hz = 1e-23
optical_energy = 2.4 ##eV

opt_factor = Jy2erg_cm_s_Hz*erg2eV/h*optical_energy/1.e3/1.e6

num_objects = len(data_in)
for ii in range(0,num_objects):
	obj_num = ii
	
	try:
		obj_3FGL_name = data_in['alt_name_gamma'].iloc[ii]
		obj_3FGL_name = obj_3FGL_name[5:]+' '
		C_val = peak_data['C'].loc[peak_data['3FGL'] == obj_3FGL_name].values
		data_in['synch_peak'].loc[ii] = C_val[0][0]
		# print(data_in['synch_peak'].loc[ii])
	except:
		pass 
	
	optical_tag = 1
	try:
		optic_file = '../data/optical/object'+str(obj_num).zfill(4)+'_asas-sn.csv'
		read_in = pd.read_csv(optic_file)
		
		opt_time = read_in['HJD'].values #- time_min
		opt_flux = read_in['flux'].values
		opt_flux_err = read_in['flux_err'].values
		
		# opt_flux_average = np.average(opt_flux[opt_flux!=99.99], weights=1/opt_flux_err[opt_flux!=99.99]**2)
		# opt_flux_average_err = 1/np.sqrt(np.sum(1/opt_flux_err[opt_flux!=99.99]**2))
		
		opt_flux_average = np.average(opt_flux[opt_flux!=99.99])
		opt_flux_average_err = np.sqrt(np.sum(opt_flux_err[opt_flux!=99.99]**2)/len(opt_flux[opt_flux!=99.99]))
		
		data_in['opt_flux_average'].loc[ii] = opt_flux_average
		data_in['opt_flux_average_err'].loc[ii] = opt_flux_average_err
		
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
		
		gam_lerr_avg = np.average(gam_lerr)
		gam_uerr_avg = np.average(gam_uerr)
		
		gam_err_avg = (gam_lerr+gam_uerr)/2.
		
		gam_flux_median = np.median(gam_flux)
		# gam_flux_median_lerr = np.median(gam_lerr)
		# gam_flux_median_uerr = np.median(gam_uerr)
		
		data_in['gam_flux_median'].loc[ii] = gam_flux_median
		# data_in['gam_flux_median_lerr'].loc[ii] = gam_flux_median_lerr
		# data_in['gam_flux_median_uerr'].loc[ii] = gam_flux_median_uerr
		
		## chi^2 variability
		chi2_var = np.sum((gam_flux[gam_flux>gam_flux_median]-gam_flux_median)**2/gam_lerr[gam_flux>gam_flux_median]**2)
		chi2_var+= np.sum((gam_flux[gam_flux<=gam_flux_median]-gam_flux_median)**2/gam_uerr[gam_flux<=gam_flux_median]**2)
		chi2_var/= len(gam_time)
		
		data_in['gam_chi2_var'].loc[ii] = chi2_var
		
		## Variability amplitude
		gam_flux_max = np.amax(gam_flux)
		gam_flux_min = np.amin(gam_flux)
		
		var_amp = 1/gam_flux_median*((gam_flux_max-gam_flux_min)**2 - gam_lerr_avg**2 - gam_uerr_avg**2)**0.5
		
		data_in['gam_var_amplitude'].loc[ii] = var_amp
		
		## doubling time
		doubling_time = (gam_time[1:]-gam_time[:-1])*np.log(2)/np.log(gam_flux[1:]/gam_flux[:-1]+1.e-30)
		doubling_time = abs(doubling_time)
		dtime_arg = np.argmin(doubling_time)
		
		doubling_time = doubling_time[dtime_arg]
		
		doubling_time_err = (doubling_time/gam_flux[1:]/np.log(gam_flux[1:]/gam_flux[:-1]+1.e-30))**2
		doubling_time_err*= gam_err_avg[1:]**2 + (gam_flux[1:]/gam_flux[:-1])**2*gam_err_avg[:-1]**2
		doubling_time_err = doubling_time_err**0.5
		
		doubling_time_err = doubling_time_err[dtime_arg]
		
		data_in['gam_doubling_time'].loc[ii] = doubling_time
		data_in['gam_doubling_time_err'].loc[ii] = doubling_time_err

	except:
		gamma_tag = 0
		
	gamma_total_tag = 1
	try:
		gamma_file = './gamma-ray/vT/object'+str(obj_num).zfill(4)+'_gam_tot.dat'
		read_in = np.loadtxt(gamma_file).T
		
		gam_flux_total = read_in[0]
		gam_lerr_total = read_in[1]
		gam_uerr_total = read_in[2]
		gam_flux_total_ns = read_in[3]
		gam_flux_total_cb = read_in[4]
		gam_flux_total_ba = read_in[5]
		gam_flux_total_err_ba = read_in[6]
		
		gam_photon_flux_total = read_in[7]
		gam_photon_lerr_total = read_in[8]
		gam_photon_uerr_total = read_in[9]
		gam_photon_flux_total_ns = read_in[10]
		gam_photon_flux_total_cb = read_in[11]
		gam_photon_flux_total_ba = read_in[12]
		gam_photon_flux_total_err_ba = read_in[13]
		
		data_in['gam_flux_total'].loc[ii] = gam_flux_total
		data_in['gam_flux_total_lerr'].loc[ii] = gam_lerr_total
		data_in['gam_flux_total_uerr'].loc[ii] = gam_uerr_total
		data_in['gam_flux_total_ns'].loc[ii] = gam_flux_total_ns
		data_in['gam_flux_total_cb'].loc[ii] = gam_flux_total_cb
		data_in['gam_flux_total_ba'].loc[ii] = gam_flux_total_ba
		data_in['gam_flux_total_err_ba'].loc[ii] = gam_flux_total_err_ba
		
		data_in['gam_photon_flux_total'].loc[ii] = gam_photon_flux_total
		data_in['gam_photon_flux_total_lerr'].loc[ii] = gam_photon_lerr_total
		data_in['gam_photon_flux_total_uerr'].loc[ii] = gam_photon_uerr_total
		data_in['gam_photon_flux_total_ns'].loc[ii] = gam_photon_flux_total_ns
		data_in['gam_photon_flux_total_cb'].loc[ii] = gam_photon_flux_total_cb
		data_in['gam_photon_flux_total_ba'].loc[ii] = gam_photon_flux_total_ba
		data_in['gam_photon_flux_total_err_ba'].loc[ii] = gam_photon_flux_total_err_ba
		
	except:
		gamma_total_tag = 0
		
	gamma_total_spectrum_tag = 1
	try:
		gamma_file = './gamma-ray/vT/object'+str(obj_num).zfill(4)+'_spectrum_tot.dat'
		read_in = np.loadtxt(gamma_file).T
		
		beta_total = read_in[12]
		beta_total_lerr = read_in[13]
		beta_total_uerr = read_in[14]
		
		beta_low_total = read_in[15]
		beta_low_total_lerr = read_in[16]
		beta_low_total_uerr = read_in[17]
		
		beta_high_total = read_in[18]
		beta_high_total_lerr = read_in[19]
		beta_high_total_uerr = read_in[20]
		
		curve_total = read_in[21] + 2.
		curve_total_lerr = read_in[22]
		curve_total_uerr = read_in[23]
		
		data_in['beta_total'].loc[ii] = beta_total
		data_in['beta_total_lerr'].loc[ii] = beta_total_lerr
		data_in['beta_total_uerr'].loc[ii] = beta_total_uerr
		
		data_in['beta_low_total'].loc[ii] = beta_low_total
		data_in['beta_low_total_lerr'].loc[ii] = beta_low_total_lerr
		data_in['beta_low_total_uerr'].loc[ii] = beta_low_total_uerr

		data_in['beta_high_total'].loc[ii] = beta_high_total
		data_in['beta_high_total_lerr'].loc[ii] = beta_high_total_lerr
		data_in['beta_high_total_uerr'].loc[ii] = beta_high_total_uerr

		data_in['curve_total'].loc[ii] = curve_total
		data_in['curve_total_lerr'].loc[ii] = curve_total_lerr
		data_in['curve_total_uerr'].loc[ii] = curve_total_uerr
		
	except:
		gamma_total_spectrum_tag = 0
	
	if optical_tag == 1 and gamma_total_tag == 1:
		F_opt_ratio = opt_flux_average*opt_factor
		F_gam_ratio = gam_flux_total
		
		fe_opt_ratio = opt_flux_average_err*opt_factor
		fe_d_gam_ratio = gam_lerr_total
		fe_u_gam_ratio = gam_uerr_total
		
		gam_opt_ratio_total = F_opt_ratio/F_gam_ratio
		ratio_total_err_u = ((fe_opt_ratio/F_gam_ratio)**2 + (F_opt_ratio/F_gam_ratio**2*fe_u_gam_ratio)**2)**0.5
		ratio_total_err_d = ((fe_opt_ratio/F_gam_ratio)**2 + (F_opt_ratio/F_gam_ratio**2*fe_d_gam_ratio)**2)**0.5
		
		data_in['opt_gam_ratio'].loc[ii] = gam_opt_ratio_total
		data_in['opt_gam_ratio_lerr'].loc[ii] = ratio_total_err_d
		data_in['opt_gam_ratio_uerr'].loc[ii] = ratio_total_err_u
	
	gamma_flux_tag = 1
	try:
		gamma_file = './gamma-ray/v'+str(gam_ver)+'/object'+str(obj_num).zfill(4)+'_gam.npy'
		read_in = np.load(gamma_file)
		gam_time_E = read_in[:,0,0]/(60*60*24) + 2451910.5
		gam_flux_E = read_in[:,:,1]
		gam_lerr_E = read_in[:,:,2]
		gam_uerr_E = read_in[:,:,3]
		gam_expo_E = read_in[:,:,4]
		
		# gam_flux_tot_E = np.average(gam_flux_E, axis=0, weights=gam_expo_E)
		# gam_lerr2_tot_E = np.average(gam_lerr_E**2, axis=0, weights=gam_expo_E)
		# gam_uerr2_tot_E = np.average(gam_uerr_E**2, axis=0, weights=gam_expo_E)
		
		# gam_flux_total_wa = np.sum(gam_flux_tot_E)
		# gam_flux_total_lerr_wa = np.sqrt(np.sum(gam_lerr2_tot_E))
		# gam_flux_total_uerr_wa = np.sqrt(np.sum(gam_uerr2_tot_E))
		
		# data_in['gam_flux_total_wa'].loc[ii] = gam_flux_total_wa
		# data_in['gam_flux_total_lerr_wa'].loc[ii] = gam_flux_total_lerr_wa 
		# data_in['gam_flux_total_uerr_wa'].loc[ii] = gam_flux_total_uerr_wa
	except:
		gamma_flux_tag = 0
	
	timescale_tag = 1
	try:
		filename = '../2001/output/bayesien_timescale/object'+str(obj_num).zfill(4)+'_timescales000.dat'
		read_in = np.loadtxt(filename)
		data_in['timescale_opt'].loc[ii] = read_in[0,0]
		data_in['timescale_opt_lerr'].loc[ii] = read_in[0,1]
		data_in['timescale_opt_uerr'].loc[ii] = read_in[0,2]
		data_in['timescale_gam'].loc[ii] = read_in[1,0]
		data_in['timescale_gam_lerr'].loc[ii] = read_in[1,1]
		data_in['timescale_gam_uerr'].loc[ii] = read_in[1,2]
	except:
		timescale_flux_tag = 0
	
	cross_corr_tag = 1
	try:
		filename = '../2001/output/cross_corr/object'+str(obj_num).zfill(4)+'_stats000.dat'
		read_in = np.loadtxt(filename)
		time = read_in[0]
		data = read_in[1:]
		
		cross_corr_t0 = data[0,time==0.]
		cross_corr_t0_lerr = data[1,time==0]
		cross_corr_t0_uerr = data[2,time==0]
		
		data_in['cross_corr_t0'].loc[ii] = data[0,time==0.]
		data_in['cross_corr_t0_lerr'].loc[ii] = data[0,time==0.] - data[1,time==0]
		data_in['cross_corr_t0_uerr'].loc[ii] = data[2,time==0] - data[0,time==0.]
		
	except:
		cross_corr_tag = 0
	
	cross_corr_err_tag = 1
	try:
		filename = '../2001/output/cross_object_gam+opt/cross_object'+str(obj_num).zfill(4)+'_all_stats_000.dat'
		read_in = np.loadtxt(filename)
		time = read_in[0]
		data = read_in[1:]
		data = data[:,time == 0.]
		data = data[:-2]
		# print(data.T)
		
		significance = np.array([0., -1., 1., -2., 2., -3., 3.])
		significance_function = interp1d(data.T[0], significance, fill_value='extrapolate')
		
		data_in['cross_corr_sig_t0'].loc[ii] = significance_function(cross_corr_t0)
		data_in['cross_corr_sig_t0_lerr'].loc[ii] = significance_function(cross_corr_t0) - significance_function(cross_corr_t0_lerr)
		data_in['cross_corr_sig_t0_uerr'].loc[ii] = significance_function(cross_corr_t0_uerr) - significance_function(cross_corr_t0)
		
	except:
		cross_corr_err_tag = 0
		
data_in['median_optical_flux'] = pd.to_numeric(data_in['median_optical_flux'], errors='coerce') 
	
data_in.to_csv('fermi_4FGL_associations_ext_GRPHorder+data.csv')