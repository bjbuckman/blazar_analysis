import numpy as np
# import pandas as pd
import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import sys

import pymc3 as pm
import theano 
import theano.tensor as tt

# theano.config.exception_verbosity='high'

obj_num = 7000 #int(sys.argv[1])
filename_err = '../source_type/dat/object7000_stats000.dat'
# filename_err = str(sys.argv[2])
# filename_gam_err = str(sys.argv[3])
out_num = 0
	
def my_model(theta, time):
	'''
	Theta are model parameters
	'''
	tau, blah, sig = theta
	return blah*np.exp(-(time-tau)**2/2./sig**2)#/np.sqrt(2*np.pi)/sig
	
def my_loglike(theta, time, cc, cc_sigma_l, cc_sigma_u):
	output = 0.
	
	for jj in range(0,len(time)):
		model_cc = my_model(theta, time[jj])
		if cc[jj] >= model_cc:
			output+= (-(cc[jj] - model_cc)**2/2./cc_sigma_l[jj]**2) - np.log(cc_sigma_l[jj]**2)
		else:
			output+= (-(cc[jj] - model_cc)**2/2./cc_sigma_u[jj]**2) - np.log(cc_sigma_u[jj]**2)
	return output
	
def my_loglike_grad(theta, time, cc, cc_sigma_l, cc_sigma_u):
	output = np.zeros(len(theta))
	tau, blah, sig = theta
	
	for jj in range(0,len(time)):
		model_cc = my_model(theta, time[jj])
		if cc[jj] >= model_cc:
			output[0]+= (cc[jj] - model_cc)/cc_sigma_l[jj]**2 *model_cc*(time[jj]-tau)/sig**2 
			output[1]+= (cc[jj] - model_cc)/cc_sigma_l[jj]**2 *model_cc/blah 
			output[2]+= (cc[jj] - model_cc)/cc_sigma_l[jj]**2 *model_cc*(time[jj]-tau)**2/sig**3 
		else:
			output[0]+= (cc[jj] - model_cc)/cc_sigma_u[jj]**2 *model_cc*(time[jj]-tau)/sig**2 
			output[1]+= (cc[jj] - model_cc)/cc_sigma_u[jj]**2 *model_cc/blah 
			output[2]+= (cc[jj] - model_cc)/cc_sigma_u[jj]**2 *model_cc*(time[jj]-tau)**2/sig**3
			
	return output
	
def gradients(vals, time, cc, cc_sigma_l, cc_sigma_u):
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
	grads = my_loglike_grad(vals, time, cc, cc_sigma_l, cc_sigma_u)
	
	return grads

# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

	itypes = [tt.dvector] # expects a vector of parameter values when called
	otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

	def __init__(self, loglike, time, cc, cc_sigma_l, cc_sigma_u):
		"""
		Initialise the Op with various things that our log-likelihood function
		requires. Below are the things that are needed in this particular
		example.

		Parameters
		----------
		loglike:
			The log-likelihood (or whatever) function we've defined
		cc:
			The "observed" data that our log-likelihood function takes in
		time:
			The dependent variable (aka 'x') that our model requires
		sigma_l:
			The noise standard deviation that our function requires.
		sigma_u:
			The noise standard deviation that our function requires.
		"""

		# add inputs as class attributes
		self.likelihood = loglike
		self.cc = cc
		self.time = time
		self.cc_sigma_l = cc_sigma_l
		self.cc_sigma_u = cc_sigma_u

		# initialise the gradient Op (below)
		self.logpgrad = LogLikeGrad(self.likelihood, self.time, self.cc, self.cc_sigma_l, self.cc_sigma_u)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		logl = self.likelihood(theta, self.time, self.cc, self.cc_sigma_l, self.cc_sigma_u)

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

	def __init__(self, loglike, time, cc, cc_sigma_l, cc_sigma_u):
		"""
		Initialise with various things that the function requires. Below
		are the things that are needed in this particular example.

		Parameters
		----------
		loglike:
			The log-likelihood (or whatever) function we've defined
		cc:
			The "observed" data that our log-likelihood function takes in
		time:
			The dependent variable (aka 'x') that our model requires
		sigma_l:
			The noise standard deviation that our function requires.
		sigma_u:
			The noise standard deviation that our function requires.
		"""

		# add inputs as class attributes
		self.likelihood = loglike
		self.cc = cc
		self.time = time
		self.cc_sigma_l = cc_sigma_l
		self.cc_sigma_u = cc_sigma_u

	def perform(self, node, inputs, outputs):
		theta, = inputs

		# define version of likelihood function to pass to derivative function
		def lnlike(values):
			return self.likelihood(theta, self.time, self.cc, self.cc_sigma_l, self.cc_sigma_u)

		# calculate gradients
		grads = gradients(theta, self.time, self.cc, self.cc_sigma_l, self.cc_sigma_u)

		outputs[0][0] = grads


###
###
### MAIN
###
###
output = np.zeros(2)
max_time = 50.

auto_opt_err_tag = 1
try:
	# filename_err = autocorr_opt_err_dir+'object'+str(obj_num).zfill(4)+'_auto_opt_stats'+str(opt_file_num).zfill(3)+'.dat'
	read_in = np.loadtxt(filename_err)

	time_opt_err = read_in[0]
	data_opt_err = read_in[1:]

	cor_mean_opt = data_opt_err[0]
	cor_err_d_opt = cor_mean_opt-data_opt_err[1]
	cor_err_u_opt = data_opt_err[2]-cor_mean_opt
	
	# cor_err_d_opt[0] = 1.e-10
	# cor_err_u_opt[0] = 1.e-10
	
	cor_mean_opt = cor_mean_opt[abs(time_opt_err)<max_time]
	cor_err_d_opt = cor_err_d_opt[abs(time_opt_err)<max_time]
	cor_err_u_opt = cor_err_u_opt[abs(time_opt_err)<max_time]
	time_opt_err = time_opt_err[abs(time_opt_err)<max_time]
	
except:
	auto_opt_err_tag = 0
	
### Calc spectral index by MCMC
if auto_opt_err_tag == 1:
	
	ndraws = 500  # number of draws from the distribution
	nburn = 500   # number of "burn-in points" (which we'll discard)
	
	logl = LogLikeWithGrad(my_loglike, time_opt_err, cor_mean_opt, cor_err_d_opt, cor_err_u_opt)
	with pm.Model():
		## Priors for unknown model parameters
		tau = pm.Normal('tau', mu=0., sigma=50., testval=1.)
		blah = pm.Normal('blah', mu=0., sigma=1., testval=1.)
		sig = pm.HalfNormal('sig', sigma=100., testval=1.)
		
		## convert m and c to a tensor vector
		theta = tt.as_tensor_variable([tau,blah,sig])
		
		## use a DensityDist (use a lamdba function to "call" the Op)
		pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
		
		starting_point = {'tau':1, 'blah':1., 'sig':10.}
		trace = pm.sample(ndraws, init='auto', tune=nburn, start=starting_point)
	print(trace)
	summary = pm.summary(trace)
	print(summary)
	
	output[0] = summary['mean']['tau']
	output[1] = summary['sd']['tau']
	
else:
	output[0] = np.nan
	output[1] = np.nan

# ###
# ###
# ### Gamma-ray timescale
# ###
# auto_gam_err_tag = 1
# try:
	# # filename_gam_err = autocorr_gam_err_dir+'object'+str(obj_num).zfill(4)+'_auto_gam_stats'+str(gam_file_num).zfill(3)+'.dat'
	# read_in = np.loadtxt(filename_gam_err)

	# time_gam_err = read_in[0]
	# data_gam_err = read_in[1:]

	# cor_mean_gam = data_gam_err[0]
	# cor_err_d_gam = cor_mean_gam-data_gam_err[1]
	# cor_err_u_gam = data_gam_err[2]-cor_mean_gam
	
	# cor_err_d_gam[0] = 1.e-10
	# cor_err_u_gam[0] = 1.e-10
	
	# cor_mean_gam = cor_mean_gam[time_gam_err<50.]
	# cor_err_d_gam = cor_err_d_gam[time_gam_err<50.]
	# cor_err_u_gam = cor_err_u_gam[time_gam_err<50.]
	# time_gam_err = time_gam_err[time_gam_err<50.]
	
# except:
	# auto_gam_err_tag = 0

# ### Calc spectral index by MCMC
# if auto_gam_err_tag == 1:
	
	# ndraws = 500  # number of draws from the distribution
	# nburn = 500   # number of "burn-in points" (which we'll discard)
	
	# logl = LogLikeWithGrad(my_loglike, time_gam_err, cor_mean_gam, cor_err_d_gam, cor_err_u_gam)
	# with pm.Model():
		# ## Priors for unknown model parameters
		# tau = pm.Normal('tau', mu=50., sigma=50., testval=1.)
		# blah = pm.Normal('blah', mu=1., sigma=0.1, testval=1.)
		
		# ## convert m and c to a tensor vector
		# theta = tt.as_tensor_variable([tau,blah])
		
		# ## use a DensityDist (use a lamdba function to "call" the Op)
		# pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
		
		# starting_point = {'tau':20, 'blah':1.}
		# trace = pm.sample(ndraws, init='auto', tune=nburn, start=starting_point)
	# summary = pm.summary(trace)
	# print(summary)
	
	# output[2] = summary['mean']['tau']
	# output[3] = summary['sd']['tau']
	
# else:
	# output[2] = np.nan
	# output[3] = np.nan

filename_out = 'object'+str(obj_num).zfill(4)+'_cc_tau'+str(out_num).zfill(3)+'.dat'
np.savetxt(filename_out, output)
