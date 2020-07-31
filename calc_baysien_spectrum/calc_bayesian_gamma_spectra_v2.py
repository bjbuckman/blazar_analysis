import numpy as np
import pandas as pd
import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import sys

# import pymc3 as pm
# import theano.tensor as tt
import emcee

object_num = int(sys.argv[1]) ## Object number to use
gamma_file = str(sys.argv[2]) ## Input gamma file
energy_file = str(sys.argv[3]) ## Input gamma-ray energy file
	
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


##CURVATURE
def my_model2(theta, lnE):
	'''
	Theta are model parameters
	'''
	lnA, beta1, beta2 = theta
	return lnA + beta1*lnE + beta2/2*lnE**2
	
	
def my_loglike2(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	output = 0.
	
	for jj in range(0,len(lnE)):
		model_flux = my_model2(theta, lnE[jj])
		if ln_observed_flux[jj] >= model_flux:
			output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_u[jj]**2) - np.log(ln_observed_sigma_u[jj]**2)
		else:
			output+= (-(ln_observed_flux[jj] - model_flux)**2/2./ln_observed_sigma_l[jj]**2) - np.log(ln_observed_sigma_l[jj]**2)
	return output

def log_prior2(theta):
	lnA, beta1, beta2 = theta
	if -20. < lnA < 20. and -6. < beta1 < 6.0 and -10. < beta2 < 10.:
		return 0.0
	return -np.inf

def log_probability2(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	lp = log_prior2(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + my_loglike2(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u)


###
###
### MAIN
###
###

gamma_tag = 1
# try:
read_in = np.load(gamma_file)

time = read_in[:,0,0]
flux = read_in[:,:,1]
flux_err_d = read_in[:,:,2]
flux_err_u = read_in[:,:,3]

ENERGY = np.loadtxt(energy_file)
enum = len(ENERGY)
# ENERGY = np.array([125.9, 199.5, 316.2, 501.2, 794.3, 1258.9, 1995.3, 3162.3, 5011.9, 7943.3, 12589.3, 19952.6, 31622.8, 50118.7, 79432.8, 125892.5, 199526.2, 501187.2, 794328.2]) #MeV

logENERGY = np.log(ENERGY)
logflux = np.log(flux)
logflux_err_d = abs(np.log((flux-flux_err_d)/flux))
logflux_err_u = abs(np.log((flux+flux_err_u)/flux))
lnE = logENERGY

###
###
### Calc spectral index by MCMC
###
###

beta_out = np.zeros([len(time), 13])
beta_out[:,0] = time

for ii in range(0,len(time)):
	ln_observed_flux = logflux[ii]
	ln_observed_sigma_l = logflux_err_d[ii]
	ln_observed_sigma_u = logflux_err_u[ii]
	
	nwalkers = 10
	ndim = 2
	pos = np.random.uniform(low=-4, high=0, size=[nwalkers, ndim])
	# nwalkers, ndim = pos.shape
	
	## ALL ENERGIES
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u))
	sampler.run_mcmc(pos, 2000, progress=False);
	
	samples = sampler.get_chain()
	
	flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
	
	mcmc = np.percentile(flat_samples, [16,50,84], axis=0)
	q = np.diff(mcmc, axis=0)

	beta_out[ii,1] = mcmc[1,1] - 2.
	beta_out[ii,2] = q[0,1]
	beta_out[ii,3] = q[1,1]
	
	
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

	beta_out[ii,4] = mcmc[1,1] - 2.
	beta_out[ii,5] = q[0,1]
	beta_out[ii,6] = q[1,1]
	
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

		beta_out[ii,7] = mcmc[1,1] - 2.
		beta_out[ii,8] = q[0,1]
		beta_out[ii,9] = q[1,1]
		
	##CURVATURE
	nwalkers = 10
	ndim = 3
	pos = np.random.uniform(low=-4, high=4, size=[nwalkers, ndim])

	## ALL ENERGIES
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability2, args=(lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u))
	sampler.run_mcmc(pos, 4000, progress=False);

	samples = sampler.get_chain()

	flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

	mcmc = np.percentile(flat_samples, [16,50,84], axis=0)
	q = np.diff(mcmc, axis=0)

	beta_out[ii,10] = mcmc[1,2] - 2.
	beta_out[ii,11] = q[0,2]
	beta_out[ii,12] = q[1,2]

	
	# print(str(object_num)+' '+str(ii)+'/'+str(len(time)))


out_name = 'object'+str(object_num).zfill(4)+'_'
np.savetxt(out_name+'bayesian_spectrum.dat', beta_out)




