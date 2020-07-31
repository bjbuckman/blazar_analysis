import numpy as np
import pandas as pd
import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import sys

import pymc3 as pm
import theano.tensor as tt

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
	
def my_loglike_grad(theta, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	output = np.zeros(len(theta))
	
	for jj in range(0,len(lnE)):
		model_flux = my_model(theta, lnE[jj])
		if ln_observed_flux[jj] >= model_flux:
			output[0]+= (ln_observed_flux[jj] - model_flux)/ln_observed_sigma_u[jj]**2 
			output[1]+= (ln_observed_flux[jj] - model_flux)/ln_observed_sigma_u[jj]**2 *lnE[jj]
		else:
			output[0]+= (ln_observed_flux[jj] - model_flux)/ln_observed_sigma_l[jj]**2  
			output[1]+= (ln_observed_flux[jj] - model_flux)/ln_observed_sigma_l[jj]**2 *lnE[jj]
			
	return output
	
def gradients(vals, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
	"""
	Calculate the partial derivatives of a function at a set of values. 

	Parameters
	----------
	vals: array_like
		A set of values, that are passed to a function, at which to calculate
		the gradient of that function
	func:
		A function that takes in an array of values.
		
	Returns
	-------
	grads: array_like
		An array of gradients for each non-fixed value.
	"""
	grads = np.zeros(len(vals))
	grads = my_loglike_grad(vals, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u)
	
	return grads


# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

	itypes = [tt.dvector] # expects a vector of parameter values when called
	otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

	def __init__(self, loglike, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
		"""
		Initialise the Op with various things that our log-likelihood function
		requires. Below are the things that are needed in this particular
		example.

		Parameters
		----------
		loglike:
			The log-likelihood (or whatever) function we've defined
		ln_observed_flux:
			The "observed" data that our log-likelihood function takes in
		lnE:
			The dependent variable (aka 'x') that our model requires
		sigma_l:
			The noise standard deviation that our function requires.
		sigma_u:
			The noise standard deviation that our function requires.
		"""

		# add inputs as class attributes
		self.likelihood = loglike
		self.ln_observed_flux = ln_observed_flux
		self.lnE = lnE
		self.ln_observed_sigma_l = ln_observed_sigma_l
		self.ln_observed_sigma_u = ln_observed_sigma_u

		# initialise the gradient Op (below)
		self.logpgrad = LogLikeGrad(self.likelihood, self.lnE, self.ln_observed_flux, self.ln_observed_sigma_l, self.ln_observed_sigma_u)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		logl = self.likelihood(theta, self.lnE, self.ln_observed_flux, self.ln_observed_sigma_l, self.ln_observed_sigma_u)

		outputs[0][0] = np.array(logl) # output the log-likelihood

	def grad(self, inputs, g):
		# the method that calculates the gradients - it actually returns the
		# vector-Jacobian product - g[0] is a vector of parameter values 
		theta, = inputs  # our parameters 
		return [g[0]*self.logpgrad(theta)]


class LogLikeGrad(tt.Op):

	"""
	This Op will be called with a vector of values and also return a vector of
	values - the gradients in each dimension.
	"""
	itypes = [tt.dvector]
	otypes = [tt.dvector]

	def __init__(self, loglike, lnE, ln_observed_flux, ln_observed_sigma_l, ln_observed_sigma_u):
		"""
		Initialise with various things that the function requires. Below
		are the things that are needed in this particular example.

		Parameters
		----------
		loglike:
			The log-likelihood (or whatever) function we've defined
		ln_observed_flux:
			The "observed" data that our log-likelihood function takes in
		lnE:
			The dependent variable (aka 'x') that our model requires
		sigma_l:
			The noise standard deviation that our function requires.
		sigma_u:
			The noise standard deviation that our function requires.
		"""

		# add inputs as class attributes
		self.likelihood = loglike
		self.ln_observed_flux = ln_observed_flux
		self.lnE = lnE
		self.ln_observed_sigma_l = ln_observed_sigma_l
		self.ln_observed_sigma_u = ln_observed_sigma_u

	def perform(self, node, inputs, outputs):
		theta, = inputs

		# # define version of likelihood function to pass to derivative function
		# def lnlike(values):
			# return self.likelihood(theta, self.lnE, self.ln_observed_flux, self.ln_observed_sigma_l, self.ln_observed_sigma_u)

		# calculate gradients
		grads = gradients(theta, self.lnE, self.ln_observed_flux, self.ln_observed_sigma_l, self.ln_observed_sigma_u)

		outputs[0][0] = grads


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

beta_out = np.zeros([len(time), 9])
beta_out[:,0] = time

for ii in range(0,len(time)):
	ln_observed_flux = logflux[ii]
	ln_observed_sigma_l = logflux_err_d[ii]
	ln_observed_sigma_u = logflux_err_u[ii]
	
	ndraws = 400  # number of draws from the distribution
	nburn = 500   # number of "burn-in points" (which we'll discard)
	
	## Low-energy array
	energy_arr = ENERGY<9000.
	
	ln_observed_flux_temp = ln_observed_flux[energy_arr]
	ln_observed_sigma_l_temp = ln_observed_sigma_l[energy_arr]
	ln_observed_sigma_u_temp = ln_observed_sigma_u[energy_arr]
	lnE_temp = lnE[energy_arr]
	
	logl = LogLikeWithGrad(my_loglike, lnE_temp, ln_observed_flux_temp, ln_observed_sigma_l_temp, ln_observed_sigma_u_temp)
	with pm.Model():
		## Priors for unknown model parameters
		lnA = pm.Normal('lnA', mu=-5, sigma=10., testval=-5.)
		beta = pm.Normal('beta', mu=0., sigma=4., testval=0.)
		
		## convert m and c to a tensor vector
		theta = tt.as_tensor_variable([lnA, beta])
	
		## use a DensityDist (use a lamdba function to "call" the Op)
		pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
		
		starting_point = {'lnA':-5, 'beta':0}
		trace = pm.sample(ndraws, init='auto', tune=nburn, start=starting_point, progressbar=False)
	summary = pm.summary(trace)
		
	beta_out[ii,1] = summary['mean']['beta'] - 2.
	beta_out[ii,2] = summary['sd']['beta']
	beta_out[ii,3] = summary['mean']['lnA']
	beta_out[ii,4] = summary['sd']['lnA']
	
	## High-energy spectrum
	energy_arr = ENERGY>7000.
	
	if np.sum(energy_arr) >= 2:
		ln_observed_flux_temp = ln_observed_flux[energy_arr]
		ln_observed_sigma_l_temp = ln_observed_sigma_l[energy_arr]
		ln_observed_sigma_u_temp = ln_observed_sigma_u[energy_arr]
		lnE_temp = lnE[energy_arr]
		
		logl = LogLikeWithGrad(my_loglike, lnE_temp, ln_observed_flux_temp, ln_observed_sigma_l_temp, ln_observed_sigma_u_temp)
		with pm.Model():
			## Priors for unknown model parameters
			lnA = pm.Normal('lnA', mu=-5, sigma=10., testval=-5.)
			beta = pm.Normal('beta', mu=0., sigma=4., testval=0.)
			
			## convert m and c to a tensor vector
			theta = tt.as_tensor_variable([lnA, beta])
		
			## use a DensityDist (use a lamdba function to "call" the Op)
			pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
			
			starting_point = {'lnA':-5, 'beta':0}
			trace = pm.sample(ndraws, init='auto', tune=nburn, start=starting_point, progressbar=False, cores=1)
		summary = pm.summary(trace)
		# pm.traceplot(trace)
			
		beta_out[ii,5] = summary['mean']['beta'] - 2.
		beta_out[ii,6] = summary['sd']['beta']
		beta_out[ii,7] = summary['mean']['lnA']
		beta_out[ii,8] = summary['sd']['lnA']
	else:
		beta_out[ii,5] = np.nan
		beta_out[ii,6] = np.nan
		beta_out[ii,7] = np.nan
		beta_out[ii,8] = np.nan
	
	# print(summary['mean']['beta'] - 2., summary['sd']['beta'])
	# print(summary)

out_name = gamma_file[:-8]
np.savetxt(out_name+'bayesian_spectrum.dat', beta_out, fmt=['%.10e', '%.6e', '%.6e', '%.6e', '%.6e'])




