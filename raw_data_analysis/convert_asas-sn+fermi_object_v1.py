import pandas as pd
import numpy as np
import os
import shutil

fer_dir = './fermi_lat/v3/'
gam_dir = './gamma-ray/v3/'

asa_dir = './asas-sn/'
opt_dir = './optical/'

ferT_dir = './fermi_lat/vT/'
gamT_dir = './gamma-ray/vT/'

object_csv = './fermi_4FGL_associations.csv'

read_in = pd.read_csv(object_csv)

num_obj = len(read_in)
# num_obj = 3131
# num_obj = 20

camera_dict = {	'ba':1,  'bb':2,  'bc':3,  'bd':4,  'be':5,  'bf':6,  'bg':7,  'bh':8,
				'bA':9,  'bB':10, 'bC':11, 'bD':12, 'bE':13, 'bF':14, 'bG':15, 'bH':16,
				'bi':17, 'bj':18, 'bk':19, 'bl':20, 'bm':21, 'bn':22, 'bo':23, 'bp':24,
				'bq':25, 'br':26, 'bs':27, 'bt':28 }

object_numbers = []
for ii in range(0,num_obj):
	obj_num = read_in.index[ii] 
	
	FGL_name = read_in.loc[ii].name_4FGL
	FGL_name = FGL_name.replace(' ', '_')
	# asassn_name = read_in.loc[ii].ASASSN
	
	gamma_mark = 0
	try:
		shutil.copyfile(ferT_dir+FGL_name+'_fermi_tot.dat', gamT_dir+'object'+str(obj_num).zfill(4)+'_gam_tot.dat')
		shutil.copyfile(ferT_dir+FGL_name+'_flux_tot.npy', gamT_dir+'object'+str(obj_num).zfill(4)+'_gam_tot.npy')
		shutil.copyfile(ferT_dir+FGL_name+'_spectrum_tot.dat', gamT_dir+'object'+str(obj_num).zfill(4)+'_spectrum_tot.dat')
	
		shutil.copyfile(fer_dir+FGL_name+'_fermi.dat', gam_dir+'object'+str(obj_num).zfill(4)+'_gam.dat')
		shutil.copyfile(fer_dir+FGL_name+'_flux.npy', gam_dir+'object'+str(obj_num).zfill(4)+'_gam.npy')
		shutil.copyfile(fer_dir+FGL_name+'_spectrum.dat', gam_dir+'object'+str(obj_num).zfill(4)+'_spectrum.dat')
		shutil.copyfile(fer_dir+FGL_name+'_energy.dat', gam_dir+'object'+str(obj_num).zfill(4)+'_energy.dat')
		
		# print('object'+str(obj_num).zfill(4)+' = '+FGL_name)
		gamma_mark = 1
	except:
		pass
		# print('No data for object'+str(obj_num).zfill(4)+' '+FGL_name)
	
	# bayes_mark = 1
	# try:
		# shutil.copyfile(fer_dir+FGL_name+'_bayesian_spectrum.dat', gam_dir+'object'+str(obj_num).zfill(4)+'_bayesian_spectrum.dat')
	# except:
		# bayes_mark = 0
	
	
	name = read_in.loc[ii].associated_source
	name = name.replace(' ', '_')
	
	optical_mark = 0
	try:
		data_in_v = pd.read_csv(asa_dir+'FERMI/lc/rescaled/'+name+'.dat', sep='\t')
		has_v = 1
		optical_mark = 1
	except:
		has_v = 0
		# print('No V-band')
		
	try: 
		data_in_g = pd.read_csv(asa_dir+'FERMI_g/lc/rescaled/'+name+'.dat', sep='\t')
		has_g = 1
		optical_mark = 1
	except:
		has_g = 0
		# print('No g-band')
	
	drop_columns = ['mag', 'mag_err', 'FWHM']
	if has_v and has_g:
		output = data_in_v.append(data_in_g, ignore_index=True)
		output = output.sort_values(by='HJD', ascending=True)
		output = output.reset_index()
	elif has_v and not has_g:
		output = data_in_v
	elif not has_v and has_g:
		output = data_in_g
	
	if has_v or has_g:
		output.drop(drop_columns, axis=1, inplace=True)
		output[['HJD', 'flux', 'flux_err']] = output[['HJD', 'flux', 'flux_err']].apply(pd.to_numeric)
		
		CAL_FILE = 1
		try:
			cal_columns = ['cam', 'num', 'offset', 'offset_err']
			calibration = pd.read_csv(asa_dir+'cals/'+name+'.cal', sep=r'\s+', skiprows=3, header=None, names=cal_columns)
			
			# calibration.drop(calibration[calibration.offset == 0].index, inplace=True)
			# calibration.reset_index(inplace=True)

			mag_offset_dict = {calibration.cam[ii] : -calibration.offset[ii] for ii in range(0,len(calibration)) }
			mag_offset_err_dict = {calibration.cam[ii] : -calibration.offset_err[ii] for ii in range(0,len(calibration)) }
			
			output['camera_num'] = output['camera']
			output.replace({'camera_num':camera_dict}, inplace=True)
			
			output['mag_offset'] = output['camera_num']
			output.replace({'mag_offset':mag_offset_dict}, inplace=True)
			
			output['mag_offset_err'] = output['camera_num']
			output.replace({'mag_offset_err':mag_offset_err_dict}, inplace=True)
			
			output.flux_err = (output.flux_err**2+(output.flux*10**(output.mag_offset_err/2.5)-output.flux)**2)**0.5
			m_arr = output.flux != 99.99
			output.loc[m_arr,'flux'] = output.flux[m_arr] + abs(output.flux[m_arr])*(10**(output.mag_offset[m_arr]/2.5) - 1.)

			drop_columns = ['camera_num', 'mag_offset', 'mag_offset_err']
			output.drop(drop_columns, axis=1, inplace=True)
		except:
			CAL_FILE = 0
			# print('No calibration file for object'+str(obj_num).zfill(4)+' = '+name) 
		
		if gamma_mark == 1:
			output.to_csv(opt_dir+'object'+str(obj_num).zfill(4)+'_asas-sn.csv', index=False)
		# print('object'+str(obj_num).zfill(4)+' = '+name)
	else:
		pass
		# print('No data for object'+str(obj_num).zfill(4)+' '+name)
		
	if gamma_mark == 1 and optical_mark == 1 and CAL_FILE == 1:
		print('object'+str(obj_num).zfill(4)+' = '+FGL_name+' '+name)
		object_numbers+= [obj_num]
	elif gamma_mark == 1 and optical_mark == 1 and CAL_FILE == 0:
		print('object'+str(obj_num).zfill(4)+' = '+FGL_name+' '+name+' -- NO CAL FILE')
		object_numbers+= [obj_num]
	elif gamma_mark == 0 and optical_mark == 1 and CAL_FILE == 1:
		print('NO GAM -- object'+str(obj_num).zfill(4)+' = '+FGL_name+' '+name)
	elif gamma_mark == 0 and optical_mark == 1 and CAL_FILE == 0:
		print('NO GAM -- object'+str(obj_num).zfill(4)+' = '+FGL_name+' '+name+' -- NO CAL FILE')
	elif gamma_mark == 1 and optical_mark == 0:
		print('NO OPT -- object'+str(obj_num).zfill(4)+' = '+FGL_name+' '+name)
	else:
		print('NO GAM or OPT -- object'+str(obj_num).zfill(4)+' = '+FGL_name+' '+name)
	
	# if gamma_mark == 0:
		# print(str(obj_num)+' '+FGL_name)
	
out_arr = np.array(object_numbers)
out_arr = out_arr.astype(int)
np.savetxt('objects_with_data.dat', out_arr.T, fmt='%d')
# print(out_arr)
