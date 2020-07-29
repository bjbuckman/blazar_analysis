import math
import sys
import numpy as np
import mpmath
# from scipy import signal
import time

from random import randint
from scipy.stats import halfnorm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

PLOT = 0
NEARBY_BIN = 3

list_of_data = []
nume=20
counter=0
time_slices = 0.0

energies = []
maxstn = []
num_signal = []
ratio_signal_outside_roi = []
num_background = []
flux_signal_total = []

##First open the cutting file, and determine which energy bins we are adding, and which we are ignoring
file = sys.argv[1]
infile = open(sys.argv[1], 'r')
for line in infile.readlines():
	params = line.split()
	energies += [float(params[0])]
	maxstn += [float(params[3])]
	num_signal += [float(params[4])]
	ratio_signal_outside_roi += [float(params[5])/float(params[4])]
	num_background += [float(params[6])]
	flux_signal_total += [float(params[7])]
infile.close()


#Load in the time series of counts, exposures and times 
mets = []
counts = []
exposures = []
energies = []
total_exposure = []
for counter in range(0,nume):
	counts_at_energy = []
	exposure_at_energy = []
	energies += [100.0*math.pow(10.0, (counter+0.5)/5.0)]
	
	infile = open('out10_lc_e' + str(counter) +'.txt', 'r')
	for line in infile.readlines():
		params = line.split()
		if(counter == 0): ##build the met file just for the first energy
			mets += [float(params[0]) + float(params[1])/2.0]
		counts_at_energy += [float(params[2])]
		exposure_at_energy += [float(params[4])]
	counts += [counts_at_energy]
	exposures += [exposure_at_energy]
	total_exposure += [np.sum(exposure_at_energy)]
	infile.close()

energies = np.array(energies)
exposures = np.array(exposures)
counts = np.array(counts)
total_exposure = np.array(total_exposure)




source_rate_step = 0.01 ##100 steps / decade
source_rate_min_coeff = 1./5.e+2
source_rate_max_coeff = 5.e+2 
source_rate_num = int(math.log10(source_rate_max_coeff/source_rate_min_coeff)/source_rate_step)

# background_rate_step = 0.01
# background_rate_min_coeff = 1./1.e1
# background_rate_max_coeff = 5.e+0
background_rate_num = 200 #int(math.log10(background_rate_max_coeff/background_rate_min_coeff)/background_rate_step)

exposure_for_signal = np.array([])
expected_source_rate = np.array([]) #constant for each energy
expected_background_rate = np.array([]) #constant for each energy
expected_total_background_counts = np.array([]) #constant for each energy

total_counts_temp = np.array([]) #total counts in energy bin
total_counts = np.array([]) #total counts in energy bin
ENERGY = np.array([]) #energy bin
ROI_FAC = np.array([]) #region of interest factor

exposures_temp = np.array([])
source_rate = np.array([])
background_rate = np.array([])

for ii in range(0,nume):
	# total_counts_temp = np.append(total_counts_temp, int(np.sum(counts[ii]))) #total counts at energy 
	if int(np.sum(counts[ii])) >= 1: # and energies[ii] > 250.:
		total_counts = np.append(total_counts, int(np.sum(counts[ii])))
		ENERGY = np.append(ENERGY, energies[ii])
		exposure_for_signal = np.append(exposure_for_signal, total_exposure[ii]/num_signal[ii])
		expected_source_rate = np.append(expected_source_rate, num_signal[ii]/total_exposure[ii])
		expected_total_background_counts = np.append(expected_total_background_counts, num_background[ii])
		expected_background_rate = np.append(expected_background_rate, num_background[ii]/total_exposure[ii])
		ROI_FAC = np.append(ROI_FAC, ratio_signal_outside_roi[ii])
		
		if len(exposures_temp) == 0:
			exposures_temp = np.array([exposures[ii]])
		else:
			exposures_temp = np.append(exposures_temp, [exposures[ii]], axis=0)
			
		source_rate_temp = (num_signal[ii]/total_exposure[ii])*np.logspace(math.log10(source_rate_min_coeff), math.log10(source_rate_max_coeff), num = source_rate_num, base=10)
		if len(source_rate) == 0:
			source_rate = np.array([source_rate_temp])
		else:
			source_rate = np.append(source_rate, [source_rate_temp], axis=0)
		
		if num_background[ii] > 50:
			background_uerr = num_background[ii] + 5.*math.sqrt(num_background[ii])
			background_lerr = num_background[ii] - 5.*math.sqrt(num_background[ii])
			background_rate_temp = (1./total_exposure[ii])*np.logspace(math.log10(background_lerr), math.log10(background_uerr), num = background_rate_num, base=10)
			if len(background_rate) == 0:
				background_rate = np.array([background_rate_temp])
			else:
				background_rate = np.append(background_rate, [background_rate_temp], axis=0)
		elif num_background[ii] <= 50 and num_background[ii] > 1.:
			background_uerr = num_background[ii] + 5.*math.sqrt(num_background[ii])
			background_lerr = num_background[ii]/10.
			background_rate_temp = (1./total_exposure[ii])*np.logspace(math.log10(background_lerr), math.log10(background_uerr), num = background_rate_num, base=10)
			if len(background_rate) == 0:
				background_rate = np.array([background_rate_temp])
			else:
				background_rate = np.append(background_rate, [background_rate_temp], axis=0)
		else:
			background_uerr = num_background[ii] + np.sqrt(num_background[ii])*5.
			background_lerr = num_background[ii]/10.
			background_rate_temp = (1./total_exposure[ii])*np.logspace(math.log10(background_lerr), math.log10(background_uerr), num = background_rate_num, base=10)
			if len(background_rate) == 0:
				background_rate = np.array([background_rate_temp])
			else:
				background_rate = np.append(background_rate, [background_rate_temp], axis=0)

# print total_counts
# print expected_source_rate 
# print expected_background_rate
# print ROI_FAC 
# print source_rate

object_name = file[:-14]

remove_time = []
if object_name == '4FGL_J0521.7+2112': #21
	remove_time = [324402217., 324488617., 324575017., 324661417., 324747817., 324834217., 324920617., 497288617., 497375017., 497461417., 497547817., 497634217., 497720617., 497807017., 497893417., 497979817., 498066217., 498152617., 498239017., 498325417., 498411817., 498498217., 498584617., 498671017., 498757417., 498843817., 498930217., 499016617., 562175017., 562261417., 562347817., 562434217., 562520617., 562607017., 562693417., 562779817.]
if object_name == '4FGL_J2236.5-1433': #42
	remove_time = [415035817.]
if object_name == '4FGL_J1058.4+0133': #46
	remove_time = [526405417., 526491817., 526837417.]
if object_name == '4FGL_J2323.5-0317': #87
	remove_time = [321291817., 352827817., 352914217., 353000617., 353087017., 353173417.]
if object_name == '4FGL J2229.7-0832': #164
	remove_time = [415035817., 415122217., 415208617., 415295017.]
if object_name == '4FGL_J1132.7+0034': #213
	remove_time = [526837417.]
if object_name == '4FGL_J1745.4-0753': #321
	remove_time = [431279017., 526837417., 526491817.]
if object_name == '4FGL_J1103.0+1157': #347
	remove_time = [431279017., 526837417., 526491817.]
if object_name == '4FGL_J2225.7-0457': #392
	remove_time = [415035817.]
if object_name == '4FGL_J1312.8-0425': #393
	remove_time = [403199017., 418146217., 418232617., 418319017., 455989417., 456075817., 456162217., 456248617.]
if object_name == '4FGL_J1120.6+0713': #476
	remove_time = [526491817., 526837417.]
if object_name == '4FGL_J0709.1+2241': #483
	remove_time = [363282217., 363368617., 363455017.]
if object_name == '4FGL_J2338.0-0230': #574
	remove_time = [352827817., 353000617., 321291817., 319823017.]
if object_name == '4FGL_J0318.7+2135': #694
	remove_time = [390152617., 390239017., 390325417.]
if object_name == '4FGL_J1050.1+0432': #715
	remove_time = [431279017., 526491817., 526837417.]
# if object_name == '':
	# remove_time = []


METS = np.array([]) ##mission elapsed time
EXPO = np.array([]) ##exposure
COUNTS = np.array([])

if len(remove_time) > 0:
	remove_time = np.array(remove_time)
	for ii in range(0,len(mets)): ##Go through every time series
		if (exposures[0][ii] > 0): ##If we looked during this period, if not, then print null
			if np.any(remove_time == mets[ii]):
				continue
			else:
				METS = np.append(METS, mets[ii])
				if len(EXPO) == 0 and len(COUNTS) == 0:
					EXPO = np.array([exposures_temp[:,ii]])
					COUNTS = np.array([counts[:,ii]])
				else:
					EXPO = np.append(EXPO, [exposures_temp[:,ii]], axis=0)
					COUNTS = np.append(COUNTS, [counts[:,ii]], axis=0)
else:
	for ii in range(0,len(mets)): ##Go through every time series
		if (exposures[0][ii] > 0): ##If we looked during this period, if not, then print null
			METS = np.append(METS, mets[ii])
			if len(EXPO) == 0 and len(COUNTS) == 0:
				EXPO = np.array([exposures_temp[:,ii]])
				COUNTS = np.array([counts[:,ii]])
			else:
				EXPO = np.append(EXPO, [exposures_temp[:,ii]], axis=0)
				COUNTS = np.append(COUNTS, [counts[:,ii]], axis=0)

EXPO_tot = np.sum(EXPO, axis=0)
counts_mets = np.sum(COUNTS, axis=1)
photon_exposure = EXPO_tot[0]/np.sum(total_counts)

# ii = len(METS)-1
# while ii >= 0:
	# if counts_mets[ii] == 0:
		# counts_bin = counts_mets[ii]
		# EXPO_add = EXPO[ii,0].copy()
		# bin_count = 1
		# bin_max = ii+1
		# ii-= 1
		# while counts_bin == 0 and ii >= 0:
			# counts_bin+= counts_mets[ii]
			# EXPO_add+= EXPO[ii,0]
			# bin_count+= 1
			# ii-= 1
		# if EXPO_add > 50.*photon_exposure and bin_count > 3.:
			# bin_min = ii+2
			# bins_del = range(bin_min,bin_max)
			# METS = np.delete(METS, bins_del) 
			# EXPO = np.delete(EXPO, bins_del, axis=0)
			# COUNTS = np.delete(COUNTS, bins_del, axis=0)
	# ii-= 1

# print EXPO_avg 
# print EXPO_tot
# print expected_source_rate*EXPO_avg
# print expected_background_rate*EXPO_avg

###
###
### Make full probalility array 
###
###

##calculate probability of r_B given n_B
P_rs_rb = np.zeros([len(ENERGY),background_rate_num])
for jj in range(0,len(ENERGY)):
	if expected_total_background_counts[jj] > 40.:
		for mm in range(0,background_rate_num):
			variance = math.sqrt(expected_total_background_counts[jj])
			P_rs_rb_temp = 1./(math.sqrt(2.*math.pi)*variance)*math.exp(-(background_rate[jj,mm]*EXPO_tot[jj]-expected_total_background_counts[jj])**2/variance**2/2.)
			P_rs_rb[jj,mm] = P_rs_rb_temp
	else:
		factorial_background_counts = math.gamma(expected_total_background_counts[jj]+1.)
		for mm in range(0,background_rate_num):
			P_rs_rb_temp = math.pow(background_rate[jj,mm]*EXPO_tot[jj], expected_total_background_counts[jj])*math.exp(-background_rate[jj,mm]*EXPO_tot[jj])/factorial_background_counts
			P_rs_rb[jj,mm] = P_rs_rb_temp

P = np.zeros([len(METS),len(ENERGY),source_rate_num]) ##full probability array
P_rb = np.zeros(background_rate_num) ##background probability array

# EXPO_min = np.percentile(EXPO, 10., axis=0)
# EXPO_one = exposure_for_signal

for ii in range(0,len(METS)):
	for jj in range(0,len(ENERGY)):
		X = int(COUNTS[ii,jj]) ##The observed counts (background+signal) in bin
		for kk in range(0,source_rate_num):
			for mm in range(0,background_rate_num):
				try:
					##First try using the math function, which is much faster - but if this fails, redo with mpmath, which works for arbitrarily small numbers
					##Using formula derived in "bayesian reasoning in data analysis" by D'Agostini. equation 7.76
					
					P_lambda_X = math.exp(-(source_rate[jj,kk]+background_rate[jj,mm])*EXPO[ii,jj])
					P_lambda_X*= math.pow((source_rate[jj,kk]+background_rate[jj,mm])*EXPO[ii,jj], X)
					P_lambda_X/= math.factorial(X)
					
					# if expected_background_rate[jj] > 5.*expected_source_rate[jj]:
						# exponent = EXPO_min[jj]/EXPO[ii,jj]
						# exponent*= math.expm1(-EXPO_min[jj]/EXPO_one[jj]*(math.floor(EXPO[ii,jj]/EXPO_min[jj])+1.))
						# exponent/= math.expm1(-EXPO_min[jj]/EXPO_one[jj])
						
						# P_lambda_X = math.pow(P_lambda_X, exponent)
					# P_lambda_X*= P_rs_rb[jj,mm]
				except:
					P_lambda_X = mpmath.exp(-(source_rate[jj,kk]+background_rate[jj,mm])*EXPO[ii,jj])
					P_lambda_X*= mpmath.power((source_rate[jj,kk]+background_rate[jj,mm])*EXPO[ii,jj], X)
					P_lambda_X/= mpmath.factorial(X)
					
					# if expected_background_rate[jj] > 5.*expected_source_rate[jj]:
						# exponent = EXPO_min[jj]/EXPO[ii,jj]
						# exponent*= mpmath.expm1(-EXPO_min[jj]/EXPO_one[jj]*(mpmath.floor(EXPO[ii,jj]/EXPO_min[jj])+1.))
						# exponent/= mpmath.expm1(-EXPO_min[jj]/EXPO_one[jj])
						
						# P_lambda_X = mpmath.power(P_lambda_X, exponent)
					# P_lambda_X*= P_rs_rb[jj,mm]
					
				P_rb[mm] = P_lambda_X
			P_rb*= P_rs_rb[jj]
			P[ii,jj,kk] = np.sum(P_rb)
		P[ii,jj]/= np.amax(P[ii,jj])
		P[ii,jj]+= 1.e-99 ## For numerical problems later
	# print 'Finished full array '+str(METS[ii])


if NEARBY_BIN == 0:
	'''
	No additional calculations
	'''
	P_final = P	

				
elif NEARBY_BIN == 3:
	'''
	Mixture of 1 and 2
	'''
	P_final = np.zeros([len(METS), len(ENERGY), source_rate_num])
	exposure_for_event = 1./expected_source_rate 
	num_events = 30.
		
	for ii in range(0,len(METS)):
		for jj in range(0,len(ENERGY)):
			P_final_temp = P[ii,jj].copy()
			EXPO_add_u = EXPO[ii,jj].copy()
			EXPO_add_l = EXPO[ii,jj].copy()
			for ll in range(1,2*len(METS)):
				if EXPO_add_u < num_events*exposure_for_event[jj]:
					if ii+ll < len(METS):
						P_final_temp*= np.power(P[ii+ll,jj], np.exp(-EXPO_add_u/exposure_for_event[jj]))
						EXPO_add_u+= EXPO[ii+ll,jj]
					elif ii+ll >= len(METS) and ((len(METS)-1)-(ll-(len(METS)-1-ii))) >= 0:
						P_final_temp*= np.power(P[((len(METS)-1)-(ll-(len(METS)-1-ii))),jj], np.exp(-EXPO_add_u/exposure_for_event[jj]))
						EXPO_add_u+= EXPO[((len(METS)-1)-(ll-(len(METS)-1-ii))),jj]
				
				if EXPO_add_l < num_events*exposure_for_event[jj]:
					if ii-ll >= 0:
						P_final_temp*= np.power(P[ii-ll,jj], np.exp(-EXPO_add_l/exposure_for_event[jj]))
						EXPO_add_l+= EXPO[ii-ll,jj]
					elif ll-ii < len(METS) and ii-ll < 0:
						P_final_temp*= np.power(P[ll-ii,jj], np.exp(-EXPO_add_l/exposure_for_event[jj]))
						EXPO_add_l+= EXPO[ll-ii,jj]
				
				if (ii+ll >= len(METS) and ((len(METS)-1)-(ll-(len(METS)-1-ii))) < 0) and (ii-ll < 0 and ll-ii >= len(METS)):
					break
					
				if EXPO_add_u >= num_events*exposure_for_event[jj] and EXPO_add_l >= num_events*exposure_for_event[jj]:
					break
					
			P_final[ii,jj] = P_final_temp
			
	
	
P_total = np.sum(P_final, axis=2)
flux = np.zeros([len(METS),len(ENERGY),7])

for ii in range(0,len(METS)):
	mean_signal_flux = []
	ll_signal_flux = []
	ul_signal_flux = []
	for jj in range(0,len(ENERGY)):
	
		cumsum = 0.0
		llfound = 0
		ulfound = 0
		meanfound = 0
		ll_value = 0.0
		mean_value = 0.0
		ul_value = 0.0
	
		for kk in range(0,source_rate_num):
			cumsum += P_final[ii,jj,kk]
			if(cumsum > 0.15865*P_total[ii,jj] and llfound == 0):
				ll_value = source_rate[jj,kk]
				llfound = 1
			if(cumsum > 0.5*P_total[ii,jj] and meanfound == 0):
				mean_value = source_rate[jj,kk]
				meanfound = 1
			if(cumsum > 0.84135*P_total[ii,jj] and ulfound == 0):
				ul_value = source_rate[jj,kk]
				ulfound = 1
				break
		
		mean_signal_flux += [mean_value * ENERGY[jj] *ROI_FAC[jj]]
		
		if mean_value == ll_value:
			ll_signal_flux += [(mean_value - 0.999*mean_value) * ENERGY[jj] *ROI_FAC[jj]]
			# photon_ll_signal_flux += [(mean_value - 0.999*mean_value) *ROI_FAC[jj]]
		else:
			ll_signal_flux += [(mean_value - ll_value) * ENERGY[jj] *ROI_FAC[jj]]
			# photon_ll_signal_flux += [(mean_value - ll_value) *ROI_FAC[jj]]
		
		if mean_value == ul_value:
			ul_signal_flux += [(1.001*mean_value - mean_value) * ENERGY[jj] *ROI_FAC[jj]]
			# photon_ul_signal_flux += [(1.001*mean_value - mean_value) *ROI_FAC[jj]]
		else:
			ul_signal_flux += [(ul_value - mean_value) * ENERGY[jj] *ROI_FAC[jj]]
			# photon_ul_signal_flux += [(ul_value - mean_value) *ROI_FAC[jj]]
		
		flux[ii,jj,0] = METS[ii]
		flux[ii,jj,1] = mean_signal_flux[jj]
		flux[ii,jj,2] = ll_signal_flux[jj]
		flux[ii,jj,3] = ul_signal_flux[jj]
		flux[ii,jj,4] = EXPO[ii,jj]
		flux[ii,jj,5] = COUNTS[ii,jj]
		flux[ii,jj,6] = EXPO[ii,jj]/exposure_for_event[jj]
	
	mean_signal_flux = np.array(mean_signal_flux)
	ll_signal_flux = np.array(ll_signal_flux)
	ul_signal_flux = np.array(ul_signal_flux)
	
	mean_flux = np.sum(mean_signal_flux[ENERGY>250.])
	ll_flux = np.sqrt(np.sum(np.power(ll_signal_flux[ENERGY>250.], 2.0)))
	ul_flux = np.sqrt(np.sum(np.power(ul_signal_flux[ENERGY>250.], 2.0)))
	print METS[ii], mean_flux, ll_flux, ul_flux		
			
object_name = file[:-14]
# print object_name+'_flux'
# flux1 = flux.copy()
np.save(object_name+'_flux', flux)
np.savetxt(object_name+'_energy.dat', ENERGY)

###
###
### CALC SPECTRA
###
###
read_in = flux.copy()

enum = len(ENERGY)
time = read_in[:,0,0]
flux = read_in[:,:,1]
gamma_lerr_data = read_in[:,:,2]
gamma_uerr_data = read_in[:,:,3]
flux_err_d = gamma_lerr_data
flux_err_u = gamma_uerr_data

# ENERGY = np.array([125.9, 199.5, 316.2, 501.2, 794.3, 1258.9, 1995.3, 3162.3, 5011.9, 7943.3, 12589.3, 19952.6, 31622.8, 50118.7, 79432.8, 125892.5, 199526.2, 501187.2, 794328.2]) #MeV
# ENERGY = ENERGY[:enum]

logENERGY = np.log(ENERGY)
logflux = np.log(flux)
logflux_err_d = abs(np.log((flux-flux_err_d)/flux))
logflux_err_u = abs(np.log((flux+flux_err_u)/flux))
lnE = logENERGY

# dlogflux = (logflux[:,1:] - logflux[:,:-1])/(lnE[1:]-lnE[:-1])
# dlogflux_err_d = np.sqrt(logflux_err_d[:,1:]**2 + logflux_err_d[:,:-1]**2)
# dlogflux_err_u = np.sqrt(logflux_err_u[:,1:]**2 + logflux_err_u[:,:-1]**2)
dlnE = np.sqrt(lnE[1:]*lnE[:-1])

###
###
### Calc spectral index by bootstrap
###
###

beta_out = np.zeros([len(time), 10])
# lnA_out = np.zeros([len(time), 7])

beta_out[:,0] = time
# lnA_out[:,0] = time

num_it = 1000

MAT_low = np.vstack([lnE[ENERGY<1.e4], np.ones(len(lnE[ENERGY<1.e4]))]).T
if len(lnE[ENERGY>=1.e4]) > 1:
	MAT_high = np.vstack([lnE[ENERGY>=1.e4], np.ones(len(lnE[ENERGY>=1.e4]))]).T
for ii in range(0,len(time)):
	
	LNA_low = np.zeros(num_it)
	LNA_high = np.zeros(num_it)
	# LNA_p = np.zeros(num_it)

	BETA_low = np.zeros(num_it)
	BETA_high = np.zeros(num_it)
	BETA_p = np.zeros(num_it)
	for kk in range(0,num_it):
	
		logflux_rand = np.zeros(len(lnE))
		for jj in range(0,len(lnE)):
			if randint(0,1) == 0:
				logflux_rand[jj] = logflux[ii,jj] - halfnorm.rvs(scale=logflux_err_d[ii,jj])
			else:
				logflux_rand[jj] = logflux[ii,jj] + halfnorm.rvs(scale=logflux_err_u[ii,jj])
				
		dlogflux_rand = (logflux_rand[1:] - logflux_rand[:-1])/(lnE[1:]-lnE[:-1])
				
		beta_low, lnA_low = np.linalg.lstsq(MAT_low, logflux_rand[ENERGY<1.e4], rcond=None)[0]
		if len(lnE[ENERGY>=1.e4]) > 1:
			beta_high, lnA_high = np.linalg.lstsq(MAT_high, logflux_rand[ENERGY>=1.e4], rcond=None)[0]
		# beta_p, dlnA = np.linalg.lstsq(MAT_p, dlogflux_rand, rcond=None)[0]
		beta_p = np.mean(dlogflux_rand)
		
		BETA_low[kk] = beta_low
		if len(lnE[ENERGY>=1.e4]) > 1:
			BETA_high[kk] = beta_high
		else:
			BETA_high[kk] = 0.
		BETA_p[kk] = beta_p
		
		# LNA_low[kk] = lnA
		# LNA_p[kk] = dlnA
		
	mean_beta_low = np.mean(BETA_low)
	beta_out[ii,1] = mean_beta_low - 2.
	beta_out[ii,2] = mean_beta_low - np.quantile(BETA_low, 0.15865)
	beta_out[ii,3] = np.quantile(BETA_low, 0.84135) - mean_beta_low
	
	mean_beta_high = np.mean(BETA_high)
	beta_out[ii,4] = mean_beta_high - 2.
	beta_out[ii,5] = mean_beta_high - np.quantile(BETA_high, 0.15865)
	beta_out[ii,6] = np.quantile(BETA_high, 0.84135) - mean_beta_high
	
	mean_beta_p = np.mean(BETA_p)
	beta_out[ii,7] = mean_beta_p
	beta_out[ii,8] = mean_beta_p - np.quantile(BETA_p, 0.15865)
	beta_out[ii,9] = np.quantile(BETA_p, 0.84135) - mean_beta_p

np.savetxt(object_name+'_spectrum.dat', beta_out)

