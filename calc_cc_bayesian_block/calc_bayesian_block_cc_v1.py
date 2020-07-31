import pandas as pd
import numpy as np
import os
import sys

# obj_num = int(sys.argv[1])
# filename_err = str(sys.argv[2])
out_num = 0 #int(sys.argv[3])

cc_dir = './dat/'
# cross_object_cc_dir = '/mnt/c/Users/psyko/Physics/gamma-optical/2001/output/cross_object_gam+opt/'
cc_file_num = 0

# object_nums = np.array([7020,7021,7022])
object_nums = np.array([7030,7031,7032])


# for ii in range(0,len(object_arr)):
	# obj_num = object_arr[ii]	

for ii in range(0,len(object_nums)):
	obj_num = object_nums[ii]
	
	###
	###
	### Getting full time array
	###
	###

	# cross_corr_err_tag = 1
	# try:

	filename_err = cc_dir+'object'+str(obj_num).zfill(4)+'_stats'+str(cc_file_num).zfill(3)+'.dat'
	read_in = np.loadtxt(filename_err)

	time_err = read_in[0]
	data_err = read_in[1:]

	cor_mean = data_err[0]
	cor_err_d = data_err[1]
	cor_err_u = data_err[2]
	print(cor_mean[7500:7510],cor_err_d[7500:7510], cor_err_u[7500:7510])
	# cor_err_d = cor_mean-data_err[1]
	# cor_err_u = data_err[2]-cor_mean

	cor_err = (cor_err_d + cor_err_u)/2.

	cor_err[cor_mean == -1.1] = 1.
	cor_mean[cor_mean == -1.1] = 0.

	cor_err[np.isnan(cor_mean)] = 1.
	cor_mean[np.isnan(cor_mean)] = 0.
		
		
	# except:
		# cross_corr_err_tag = 0
		# print('No cc for object '+str(obj_num))
		# exit()
		# # continue


	###
	###
	### Bayesian Block analysis
	###
	###

	## Log likelihood from SCARGLE et al. 2013 equation 41
	def fitness(y, yerr):
		a_k = 1/2*np.sum(1/yerr**2)
		b_k = -np.sum(y/yerr**2)
		return b_k**2/4/a_k

	ncp_prior = 1.32+0.577*np.log10(len(cor_mean))
	print('ncp_prior = '+str(ncp_prior))

	best = np.zeros(len(cor_mean))
	last = np.zeros(len(cor_mean))

	best[0] = fitness(cor_mean[0], cor_err[0]) - ncp_prior
	last[0] = 0

	for ii in range(1,len(cor_mean)):
		A = np.zeros(ii+1)
		for r in range(0,ii+1):
			A[r] = fitness(cor_mean[r:ii+1], cor_err[r:ii+1]) - ncp_prior
			if r == 0:
				A[r]+= 0
			else:
				A[r]+= best[r-1]
		# print(A)
		last[ii] = A.argmax()
		best[ii] = A.max()
		# print('last = '+str(last[ii]))
		# print('best = '+str(best[ii]))
		
	##find change points
	index = int(last[-1])
	change_points = np.array([index], dtype=int)
	while index > 0:
		change_points = np.append(change_points, int(last[index-1]))
		index = int(last[index-1])
		
	change_points = change_points[::-1]
	print(change_points)

	def norm_bin(y, yerr):
		a_k = 1/2*np.sum(1/yerr**2)
		b_k = -np.sum(y/yerr**2)
		# print(a_k,b_k)
		return -b_k/2/a_k

	bin_norms = np.zeros(len(change_points))
	for ii in range(0,len(change_points)):
		if ii == len(change_points)-1:
			bin_norms[ii] = norm_bin(cor_mean[change_points[ii]:], cor_err[change_points[ii]:])
		else:
			bin_norms[ii] = norm_bin(cor_mean[change_points[ii]:change_points[ii+1]], cor_err[change_points[ii]:change_points[ii+1]])

	output_arr = np.array([change_points, bin_norms])

	output_filename = './change_points/object'+str(obj_num).zfill(4)+'_cc_change_points_'+str(out_num).zfill(3)+'.dat'
	np.savetxt(output_filename, output_arr)
	print('Finished object '+str(obj_num))


