import math
import sys
import numpy as np
import mpmath
# from scipy import signal
import time

from random import randint
from scipy.stats import halfnorm

# import pymc3 as pm
# import theano.tensor as tt
import emcee

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

PLOT = 0
NEARBY_BIN = 0

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
for counter in range(0, nume):
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

# exposure_for_event = np.array([])
expected_source_rate = np.array([]) #constant for each energy
expected_total_background_counts = np.array([]) #constant for each energy
expected_background_rate = np.array([]) #constant for each energy

total_counts_temp = np.array([]) #total counts in energy bin
total_counts = np.array([]) #total counts in energy bin
ENERGY = np.array([]) #energy bin
ROI_FAC = np.array([]) #region of interest factor

exposures_temp = np.array([])
source_rate = np.array([])
background_rate = np.array([])
	

for ii in range(0,nume):
	# total_counts_temp = np.append(total_counts_temp, int(np.sum(counts[ii]))) #total counts at energy 
	if int(np.sum(counts[ii])) >= 1:
		total_counts = np.append(total_counts, int(np.sum(counts[ii])))
		ENERGY = np.append(ENERGY, energies[ii])
		# exposure_for_event = np.append(exposure_for_event, total_exposure[ii]/total_counts[-1])
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
			background_uerr = num_background[ii] + 5.*math.sqrt(num_background[ii]-1.)
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
		if (exposures[0][ii] > 5.e5): ##If we looked during this period, if not, then print null
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
		if (exposures[0][ii] > 5.e5): ##If we looked during this period, if not, then print null
			METS = np.append(METS, mets[ii])
			if len(EXPO) == 0 and len(COUNTS) == 0:
				EXPO = np.array([exposures_temp[:,ii]])
				COUNTS = np.array([counts[:,ii]])
			else:
				EXPO = np.append(EXPO, [exposures_temp[:,ii]], axis=0)
				COUNTS = np.append(COUNTS, [counts[:,ii]], axis=0)

EXPO_tot = np.sum(EXPO, axis=0)
COUNTS_tot = np.sum(COUNTS, axis=0)
counts_mets = np.sum(COUNTS, axis=1)
photon_exposure = EXPO_tot[0]/np.sum(total_counts)

USE_4FGL_TIME = 1
if USE_4FGL_TIME == 1:
	###
	### 4FGL extract
	time_begin = 2453951.
	time_end = 2457602.
	mets_convert = METS/(60*60*24) + 2451910.5
	ARR_METS = np.logical_and(mets_convert>=time_begin, mets_convert<=time_end)

	EXPO = EXPO[ARR_METS]
	COUNTS = COUNTS[ARR_METS]
	METS = METS[ARR_METS]

EXPO_tot_4FGL = np.sum(EXPO, axis=0)
COUNTS_tot_4FGL = np.sum(COUNTS, axis=0)
# counts_mets = np.sum(COUNTS, axis=1)
# photon_exposure = EXPO_tot[0]/np.sum(total_counts)


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
					# P_lambda_X*= P_rs_rb[jj,mm]
				except:
					P_lambda_X = mpmath.exp(-(source_rate[jj,kk]+background_rate[jj,mm])*EXPO[ii,jj])
					P_lambda_X*= mpmath.power((source_rate[jj,kk]+background_rate[jj,mm])*EXPO[ii,jj], X)
					P_lambda_X/= mpmath.factorial(X)
					# P_lambda_X*= P_rs_rb[jj,mm]
					
				P_rb[mm] = P_lambda_X
			P_rb*= P_rs_rb[jj]
			P[ii,jj,kk] = np.sum(P_rb)
		P[ii,jj]/= np.amax(P[ii,jj])
		P[ii,jj]+= 1.e-99 ## For numerical problems later
	# print 'Finished full array '+str(METS[ii])

P_final = P[0]
for ii in range(1,len(METS)):
	P_final*= P[ii]
	for jj in range(0,len(ENERGY)):
		P_final[jj]/= np.amax(P_final[jj])
	# P_final+= 1.e-99
P_total = np.sum(P_final, axis=1)
np.save(object_name+'_prob_tot', P_final)


flux = np.zeros([len(ENERGY),17])

# for ii in range(0,len(METS)):
mean_signal_flux = []
ll_signal_flux = []
ul_signal_flux = []
signal_flux = []
signal_flux_2 = []
expected_flux_arr = []
expected_flux_err_arr = []

photon_mean_signal_flux = []
photon_ll_signal_flux = []
photon_ul_signal_flux = []
signal_photon_flux = []
signal_photon_flux_2 = []
expected_photon_flux_arr = []
expected_photon_flux_err_arr = []

for jj in range(0,len(ENERGY)):

	cumsum = 0.0
	llfound = 0
	ulfound = 0
	meanfound = 0
	ll_value = 0.0
	mean_value = 0.0
	ul_value = 0.0

	for kk in range(0,source_rate_num):
		cumsum += P_final[jj,kk]
		if(cumsum > 0.15865*P_total[jj] and llfound == 0):
			ll_value = source_rate[jj,kk]
			llfound = 1
		if(cumsum > 0.5*P_total[jj] and meanfound == 0):
			mean_value = source_rate[jj,kk]
			meanfound = 1
		if(cumsum > 0.84135*P_total[jj] and ulfound == 0):
			ul_value = source_rate[jj,kk]
			ulfound = 1
			break
		
	mean_signal_flux += [mean_value * ENERGY[jj] *ROI_FAC[jj]]
	photon_mean_signal_flux += [mean_value *ROI_FAC[jj]]
	
	if mean_value == ll_value:
		ll_signal_flux += [(mean_value - 0.999*mean_value) * ENERGY[jj] *ROI_FAC[jj]]
		photon_ll_signal_flux += [(mean_value - 0.999*mean_value) *ROI_FAC[jj]]
	else:
		ll_signal_flux += [(mean_value - ll_value) * ENERGY[jj] *ROI_FAC[jj]]
		photon_ll_signal_flux += [(mean_value - ll_value) *ROI_FAC[jj]]
	
	if mean_value == ul_value:
		ul_signal_flux += [(1.001*mean_value - mean_value) * ENERGY[jj] *ROI_FAC[jj]]
		photon_ul_signal_flux += [(1.001*mean_value - mean_value) *ROI_FAC[jj]]
	else:
		ul_signal_flux += [(ul_value - mean_value) * ENERGY[jj] *ROI_FAC[jj]]
		photon_ul_signal_flux += [(ul_value - mean_value) *ROI_FAC[jj]]
	
	signal_flux += [num_signal[jj]/EXPO_tot[jj] * ENERGY[jj] *ROI_FAC[jj]]
	signal_flux_2 += [(COUNTS_tot_4FGL[jj]-expected_background_rate[jj]*EXPO_tot_4FGL[jj])/EXPO_tot_4FGL[jj] *ENERGY[jj] *ROI_FAC[jj]]
	
	signal_photon_flux += [num_signal[jj]/EXPO_tot[jj] *ROI_FAC[jj]]
	signal_photon_flux_2 += [(COUNTS_tot_4FGL[jj]-expected_background_rate[jj]*EXPO_tot_4FGL[jj])/EXPO_tot_4FGL[jj] *ROI_FAC[jj]]
	
	flux[jj,0] = mean_signal_flux[jj]
	flux[jj,1] = ll_signal_flux[jj]
	flux[jj,2] = ul_signal_flux[jj]
	flux[jj,3] = EXPO_tot[jj]
	flux[jj,4] = COUNTS_tot[jj]
	flux[jj,5] = num_signal[jj]/EXPO_tot[jj] * ENERGY[jj] *ROI_FAC[jj]
	flux[jj,6] = (COUNTS_tot_4FGL[jj]-expected_background_rate[jj]*EXPO_tot_4FGL[jj])/EXPO_tot_4FGL[jj] *ENERGY[jj] *ROI_FAC[jj]
	
	p0 = mpmath.power(expected_background_rate[jj]*EXPO_tot_4FGL[jj], COUNTS_tot_4FGL[jj])/mpmath.factorial(COUNTS_tot_4FGL[jj])
	p0_dem = 0.
	for nn in range(0,int(COUNTS_tot_4FGL[jj])+1):
		p0_dem+= mpmath.power(expected_background_rate[jj]*EXPO_tot_4FGL[jj], nn)/mpmath.factorial(nn)
	p0/= p0_dem
	
	expected_photon_flux = float( mpmath.nstr(COUNTS_tot_4FGL[jj]+1-expected_background_rate[jj]*EXPO_tot_4FGL[jj]*(1-p0), 10))/EXPO_tot_4FGL[jj] *ROI_FAC[jj]
	expected_photon_flux_err = float( mpmath.nstr((COUNTS_tot_4FGL[jj]+1+(expected_background_rate[jj]*EXPO_tot_4FGL[jj])**2*p0*(1-p0)- COUNTS_tot_4FGL[jj]*expected_background_rate[jj]*EXPO_tot_4FGL[jj]*p0), 10))**0.5/EXPO_tot_4FGL[jj] *ROI_FAC[jj]
	
	expected_photon_flux_arr += [abs(expected_photon_flux)]
	expected_photon_flux_err_arr += [abs(expected_photon_flux_err)]
	
	expected_flux_arr += [abs(expected_photon_flux *ENERGY[jj])]
	expected_flux_err_arr += [abs(expected_photon_flux_err *ENERGY[jj])]
	
	flux[jj,7] = abs(expected_photon_flux *ENERGY[jj])
	flux[jj,8] = abs(expected_photon_flux_err *ENERGY[jj])
	
	flux[jj,9] = photon_mean_signal_flux[jj]
	flux[jj,10] = photon_ll_signal_flux[jj]
	flux[jj,11] = photon_ul_signal_flux[jj]
	
	flux[jj,12] = num_signal[jj]/EXPO_tot[jj] *ROI_FAC[jj]
	flux[jj,13] = (COUNTS_tot_4FGL[jj]-expected_background_rate[jj]*EXPO_tot_4FGL[jj])/EXPO_tot_4FGL[jj] *ROI_FAC[jj]
	
	flux[jj,14] = abs(expected_photon_flux)
	flux[jj,15] = abs(expected_photon_flux_err)
	
	flux[jj,16] = ENERGY[jj]
	

## convert to numpy arrays
mean_signal_flux = np.array(mean_signal_flux)
ll_signal_flux = np.array(ll_signal_flux)
ul_signal_flux = np.array(ul_signal_flux)
signal_flux = np.array(signal_flux)
signal_flux_2 = np.array(signal_flux_2)
expected_flux_arr = np.array(expected_flux_arr)
expected_flux_err_arr = np.array(expected_flux_err_arr)

photon_mean_signal_flux = np.array(photon_mean_signal_flux)
photon_ll_signal_flux = np.array(photon_ll_signal_flux)
photon_ul_signal_flux = np.array(photon_ul_signal_flux)
signal_photon_flux = np.array(signal_photon_flux)
signal_photon_flux_2 = np.array(signal_photon_flux_2)
expected_photon_flux_arr = np.array(expected_photon_flux_arr)
expected_photon_flux_err_arr = np.array(expected_photon_flux_err_arr)

## Extract only specific energy ranges
E_1GeV_100GeV = np.logical_and(ENERGY>1.e3, ENERGY<1.e5)
E_100MeV_100GeV = np.logical_and(ENERGY>100, ENERGY<1.e5)

## Sum over energies
mean_flux = np.sum(mean_signal_flux[E_100MeV_100GeV])
ll_flux = np.sqrt(np.sum(np.power(ll_signal_flux[E_100MeV_100GeV], 2.0)))
ul_flux = np.sqrt(np.sum(np.power(ul_signal_flux[E_100MeV_100GeV], 2.0)))
mean_flux_b = np.sum(signal_flux[E_100MeV_100GeV])
mean_flux_b_2 = np.sum(signal_flux_2[E_100MeV_100GeV])

expected_flux_tot = np.sum(expected_flux_arr[E_100MeV_100GeV])
expected_flux_tot_err = np.sqrt(np.sum(np.power(expected_flux_err_arr[E_100MeV_100GeV ], 2)))

photon_mean_flux = np.sum(photon_mean_signal_flux[E_1GeV_100GeV])
photon_ll_flux = np.sqrt(np.sum(np.power(photon_ll_signal_flux[E_1GeV_100GeV], 2.0)))
photon_ul_flux = np.sqrt(np.sum(np.power(photon_ul_signal_flux[E_1GeV_100GeV], 2.0)))
photon_mean_flux_b = np.sum(signal_photon_flux[E_1GeV_100GeV])
photon_mean_flux_b_2 = np.sum(signal_photon_flux_2[E_1GeV_100GeV])

expected_photon_flux_tot = np.sum(expected_photon_flux_arr[E_1GeV_100GeV])
expected_photon_flux_tot_err = np.sqrt(np.sum(np.power(expected_photon_flux_err_arr[E_1GeV_100GeV], 2)))

print(mean_flux, ll_flux, ul_flux, mean_flux_b, mean_flux_b_2, expected_flux_tot, expected_flux_tot_err, photon_mean_flux, photon_ll_flux, photon_ul_flux, photon_mean_flux_b, photon_mean_flux_b_2, expected_photon_flux_tot, expected_photon_flux_tot_err)

object_name = file[:-14]
# print object_name+'_flux'
# flux1 = flux.copy()
np.save(object_name+'_flux_tot', flux)


###
###
### CALC SPECTRA
###
###
read_in = flux.copy()

enum = len(ENERGY)
# time = read_in[:,0,0]
flux = read_in[:,0]
gamma_lerr_data = read_in[:,1]
gamma_uerr_data = read_in[:,2]
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

# ###
# ###
# ### Calc spectral index by bootstrap
# ###
# ###

# beta_out = np.zeros(14)
# # lnA_out = np.zeros([len(time), 7])

# # beta_out[:,0] = time
# # lnA_out[:,0] = time

# num_it = 1000

# MAT = np.vstack([lnE, np.ones(len(lnE))]).T
# MAT_low = np.vstack([lnE[ENERGY<1.e4], np.ones(len(lnE[ENERGY<1.e4]))]).T
# if len(lnE[ENERGY>=1.e4]) > 1:
	# MAT_high = np.vstack([lnE[ENERGY>=1.e4], np.ones(len(lnE[ENERGY>=1.e4]))]).T

# LNA = 0.
# LNA_low = 0.
# LNA_high = 0.
# # LNA_p = np.zeros(num_it)

# BETA = np.zeros(num_it)
# BETA_low = np.zeros(num_it)
# BETA_high = np.zeros(num_it)
# BETA_p = np.zeros(num_it)
# for kk in range(0,num_it):

	# logflux_rand = np.zeros(len(lnE))
	# for jj in range(0,len(lnE)):
		# if randint(0,1) == 0:
			# logflux_rand[jj] = logflux[jj] - halfnorm.rvs(scale=logflux_err_d[jj])
		# else:
			# logflux_rand[jj] = logflux[jj] + halfnorm.rvs(scale=logflux_err_u[jj])
			
	# dlogflux_rand = (logflux_rand[1:] - logflux_rand[:-1])/(lnE[1:]-lnE[:-1])
	
	# beta, lnA = np.linalg.lstsq(MAT, logflux_rand, rcond=None)[0]
	# beta_low, lnA_low = np.linalg.lstsq(MAT_low, logflux_rand[ENERGY<1.e4], rcond=None)[0]
	# if len(lnE[ENERGY>=1.e4]) > 1:
		# beta_high, lnA_high = np.linalg.lstsq(MAT_high, logflux_rand[ENERGY>=1.e4], rcond=None)[0]
	# beta_p = np.mean(dlogflux_rand)
	
	# BETA[kk] = beta
	# BETA_low[kk] = beta_low
	# if len(lnE[ENERGY>=1.e4]) > 1:
		# BETA_high[kk] = beta_high
	# else:
		# BETA_high[kk] = 0.
	# BETA_p[kk] = beta_p
	
	# # LNA[kk] = lnA
	# # LNA_low[kk] = lnA
	# # LNA_p[kk] = dlnA
	
# mean_beta_low = np.mean(BETA_low)
# beta_out[0] = mean_beta_low - 2.
# beta_out[1] = mean_beta_low - np.quantile(BETA_low, 0.15865)
# beta_out[2] = np.quantile(BETA_low, 0.84135) - mean_beta_low

# mean_beta_high = np.mean(BETA_high)
# beta_out[3] = mean_beta_high - 2.
# beta_out[4] = mean_beta_high - np.quantile(BETA_high, 0.15865)
# beta_out[5] = np.quantile(BETA_high, 0.84135) - mean_beta_high

# mean_beta_p = np.mean(BETA_p)
# beta_out[6] = mean_beta_p
# beta_out[7] = mean_beta_p - np.quantile(BETA_p, 0.15865)
# beta_out[8] = np.quantile(BETA_p, 0.84135) - mean_beta_p

# mean_beta = np.mean(BETA)
# beta_out[9] = mean_beta - 2.
# beta_out[10] = mean_beta - np.quantile(BETA, 0.15865)
# beta_out[11] = np.quantile(BETA, 0.84135) - mean_beta

# np.savetxt(object_name+'_spectrum_tot.dat', beta_out)

# ##
# ## Bayesian
# ##

# def my_model(theta, lnE):
	# '''
	# Theta are model parameters
	# '''
	# lnA, beta = theta
	# return lnA + beta*lnE
	
	
# def my_loglike(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	# output = 0.
	
	# for jj in range(0,len(lnE)):
		# model_flux = my_model(theta, lnE[jj])
		# if ln_observed_flux[jj] >= model_flux:
			# output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_u[jj]**2) - np.log(ln_observed_sigma_u[jj]**2)
		# else:
			# output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_l[jj]**2) - np.log(ln_observed_sigma_l[jj]**2)
	# return output
	
	
# class LogLike(tt.Op):

	# """
	# Specify what type of object will be passed and returned to the Op when it is
	# called. In our case we will be passing it a vector of values (the parameters
	# that define our model) and returning a single "scalar" value (the
	# log-likelihood)
	# """
	# itypes = [tt.dvector] # expects a vector of parameter values when called
	# otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

	# def __init__(self, loglike, ln_observed_flux, lnE, ln_observed_sigma_l, ln_observed_sigma_u):
		# """
		# Initialise the Op with various things that our log-likelihood function
		# requires. Below are the things that are needed in this particular
		# example.

		# Parameters
		# ----------
		# loglike:
			# The log-likelihood (or whatever) function we've defined
		# ln_observed_flux:
			# The "observed" data that our log-likelihood function takes in
		# lnE:
			# The dependent variable (aka 'x') that our model requires
		# sigma_l:
			# The noise standard deviation that our function requires.
		# sigma_u:
			# The noise standard deviation that our function requires.
		# """

		# # add inputs as class attributes
		# self.likelihood = loglike
		# self.ln_observed_flux = ln_observed_flux
		# self.lnE = lnE
		# self.ln_observed_sigma_l = ln_observed_sigma_l
		# self.ln_observed_sigma_u = ln_observed_sigma_u
		
	# def perform(self, node, inputs, outputs):
		# # the method that is used when calling the Op
		# theta, = inputs  # this will contain my variables

		# # call the log-likelihood function
		# logl = self.likelihood(theta, self.lnE, self.ln_observed_flux, self.ln_observed_sigma_l, self.ln_observed_sigma_u)

		# outputs[0][0] = np.array(logl) # output the log-likelihood


# ln_observed_flux = logflux
# ln_observed_sigma_l = logflux_err_d
# ln_observed_sigma_u = logflux_err_u

# ndraws = 1000  # number of draws from the distribution
# nburn = 1000   # number of "burn-in points" (which we'll discard)

# logl = LogLike(my_loglike, ln_observed_flux, lnE, ln_observed_sigma_l, ln_observed_sigma_u)
# with pm.Model():
	# ## Priors for unknown model parameters
	# lnA = pm.Normal('lnA', mu=-5, sigma=10., testval=-5.)
	# beta = pm.Normal('beta', mu=0., sigma=4., testval=0.)
	
	# ## convert m and c to a tensor vector
	# theta = tt.as_tensor_variable([lnA, beta])

	# ## use a DensityDist (use a lamdba function to "call" the Op)
	# pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
	
	# trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)
# summary = pm.summary(trace)
	
# beta_out[12] = summary['mean']['beta'] - 2.
# beta_out[13] = summary['sd']['beta']
# # beta_out[ii,3] = summary['mean']['lnA']
# # beta_out[ii,4] = summary['sd']['lnA']

###
###
### Calc spectral index by bootstrap
###
###

beta_out = np.zeros(24)
# lnA_out = np.zeros([len(time), 7])

# beta_out[:,0] = time
# lnA_out[:,0] = time

num_it = 1000

MAT = np.vstack([lnE, np.ones(len(lnE))]).T
MAT_low = np.vstack([lnE[ENERGY<1.e4], np.ones(len(lnE[ENERGY<1.e4]))]).T
if len(lnE[ENERGY>=1.e4]) > 1:
	MAT_high = np.vstack([lnE[ENERGY>=1.e4], np.ones(len(lnE[ENERGY>=1.e4]))]).T

LNA = 0.
LNA_low = 0.
LNA_high = 0.
# LNA_p = np.zeros(num_it)

BETA = np.zeros(num_it)
BETA_low = np.zeros(num_it)
BETA_high = np.zeros(num_it)
BETA_p = np.zeros(num_it)
for kk in range(0,num_it):

	logflux_rand = np.zeros(len(lnE))
	for jj in range(0,len(lnE)):
		if randint(0,1) == 0:
			logflux_rand[jj] = logflux[jj] - halfnorm.rvs(scale=logflux_err_d[jj])
		else:
			logflux_rand[jj] = logflux[jj] + halfnorm.rvs(scale=logflux_err_u[jj])
			
	dlogflux_rand = (logflux_rand[1:] - logflux_rand[:-1])/(lnE[1:]-lnE[:-1])
	
	beta, lnA = np.linalg.lstsq(MAT, logflux_rand, rcond=None)[0]
	beta_low, lnA_low = np.linalg.lstsq(MAT_low, logflux_rand[ENERGY<1.e4], rcond=None)[0]
	if len(lnE[ENERGY>=1.e4]) > 1:
		beta_high, lnA_high = np.linalg.lstsq(MAT_high, logflux_rand[ENERGY>=1.e4], rcond=None)[0]
	beta_p = np.mean(dlogflux_rand)
	
	BETA[kk] = beta
	BETA_low[kk] = beta_low
	if len(lnE[ENERGY>=1.e4]) > 1:
		BETA_high[kk] = beta_high
	else:
		BETA_high[kk] = 0.
	BETA_p[kk] = beta_p
	
	# LNA[kk] = lnA
	# LNA_low[kk] = lnA
	# LNA_p[kk] = dlnA
	
mean_beta_low = np.mean(BETA_low)
beta_out[0] = mean_beta_low - 2.
beta_out[1] = mean_beta_low - np.quantile(BETA_low, 0.15865)
beta_out[2] = np.quantile(BETA_low, 0.84135) - mean_beta_low

mean_beta_high = np.mean(BETA_high)
beta_out[3] = mean_beta_high - 2.
beta_out[4] = mean_beta_high - np.quantile(BETA_high, 0.15865)
beta_out[5] = np.quantile(BETA_high, 0.84135) - mean_beta_high

mean_beta_p = np.mean(BETA_p)
beta_out[6] = mean_beta_p
beta_out[7] = mean_beta_p - np.quantile(BETA_p, 0.15865)
beta_out[8] = np.quantile(BETA_p, 0.84135) - mean_beta_p

mean_beta = np.mean(BETA)
beta_out[9] = mean_beta - 2.
beta_out[10] = mean_beta - np.quantile(BETA, 0.15865)
beta_out[11] = np.quantile(BETA, 0.84135) - mean_beta

np.savetxt(object_name+'_spectrum_tot.dat', beta_out)

##
## Bayesian
##

def my_model(theta, lnE):
	'''
	Theta are model parameters
	'''
	lnA, beta = theta
	return lnA + beta*lnE
	
	
def my_loglike(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	output = 0.
	
	for jj in range(0,len(lnE)):
		model_flux = my_model(theta, lnE[jj])
		if ln_observed_flux[jj] >= model_flux:
			output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_u[jj]**2) - np.log(ln_observed_sigma_u[jj]**2)
		else:
			output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_l[jj]**2) - np.log(ln_observed_sigma_l[jj]**2)
	return output

def log_prior(theta):
	lnA, beta = theta
	if -20. < lnA < 20. and -6. < beta < 6.0:
		return 0.0
	return -np.inf

def log_probability(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + my_loglike(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u)



ln_observed_flux = logflux
ln_observed_sigma_l = logflux_err_d
ln_observed_sigma_u = logflux_err_u

nwalkers = 10
ndim = 2
pos = np.random.uniform(low=-4, high=0, size=[nwalkers, ndim])


## ALL ENERGIES
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u))
sampler.run_mcmc(pos, 2000, progress=False);

samples = sampler.get_chain()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

mcmc = np.percentile(flat_samples, [16,50,84], axis=0)
q = np.diff(mcmc, axis=0)

beta_out[12] = mcmc[1,1] - 2.
beta_out[13] = q[0,1]
beta_out[14] = q[1,1]


##ENERGY SPLIT
## Low-energy array
energy_arr = ENERGY<9000.

ln_observed_flux_temp = ln_observed_flux[energy_arr]
ln_observed_sigma_l_temp = ln_observed_sigma_l[energy_arr]
ln_observed_sigma_u_temp = ln_observed_sigma_u[energy_arr]
lnE_temp = lnE[energy_arr]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lnE_temp, ln_observed_flux_temp, ln_observed_sigma_l_temp, ln_observed_sigma_u_temp))
sampler.run_mcmc(pos, 2000, progress=False);

samples = sampler.get_chain()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

mcmc = np.percentile(flat_samples, [16,50,84], axis=0)
q = np.diff(mcmc, axis=0)

beta_out[15] = mcmc[1,1] - 2.
beta_out[16] = q[0,1]
beta_out[17] = q[1,1]

## High-energy spectrum
energy_arr = ENERGY>7000.

if np.sum(energy_arr) >= 2:
	ln_observed_flux_temp = ln_observed_flux[energy_arr]
	ln_observed_sigma_l_temp = ln_observed_sigma_l[energy_arr]
	ln_observed_sigma_u_temp = ln_observed_sigma_u[energy_arr]
	lnE_temp = lnE[energy_arr]
	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lnE_temp, ln_observed_flux_temp, ln_observed_sigma_l_temp, ln_observed_sigma_u_temp))
	sampler.run_mcmc(pos, 2000, progress=False);
	
	samples = sampler.get_chain()
	
	flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
	
	mcmc = np.percentile(flat_samples, [16,50,84], axis=0)
	q = np.diff(mcmc, axis=0)

	beta_out[18] = mcmc[1,1] - 2.
	beta_out[19] = q[0,1]
	beta_out[20] = q[1,1]
else:
	beta_out[18] = np.nan
	beta_out[19] = np.nan
	beta_out[20] = np.nan

# print(str(object_num)+' '+str(ii)+'/'+str(len(time)))


##CURVATURE
def my_model(theta, lnE):
	'''
	Theta are model parameters
	'''
	lnA, beta1, beta2 = theta
	return lnA + beta1*lnE + beta2/2*lnE**2
	
	
def my_loglike(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	output = 0.
	
	for jj in range(0,len(lnE)):
		model_flux = my_model(theta, lnE[jj])
		if ln_observed_flux[jj] >= model_flux:
			output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_u[jj]**2) - np.log(ln_observed_sigma_u[jj]**2)
		else:
			output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_l[jj]**2) - np.log(ln_observed_sigma_l[jj]**2)
	return output

def log_prior(theta):
	lnA, beta1, beta2 = theta
	if -20. < lnA < 20. and -6. < beta1 < 6.0 and -10. < beta2 < 10.:
		return 0.0
	return -np.inf

def log_probability(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + my_loglike(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u)

nwalkers = 10
ndim = 3
pos = np.random.uniform(low=-4, high=4, size=[nwalkers, ndim])


## ALL ENERGIES
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u))
sampler.run_mcmc(pos, 4000, progress=False);

samples = sampler.get_chain()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

mcmc = np.percentile(flat_samples, [16,50,84], axis=0)
q = np.diff(mcmc, axis=0)

beta_out[21] = mcmc[1,2] - 2.
beta_out[22] = q[0,2]
beta_out[23] = q[1,2]


np.savetxt(object_name+'_spectrum_tot.dat', beta_out)





