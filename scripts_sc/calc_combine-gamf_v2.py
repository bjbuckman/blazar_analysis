import numpy as np

obj_num = int(sys.argv[1]) ## Object number to use
num_flares = int(sys.argv[2]) ## number of flares
out_num = int(sys.argv[3]) ## bookkeeping number from previous output
iout = int(sys.argv[4]) ## bookkeeping number for this output

flare_num = np.arange(0,num_flares+1) ## Array of flare numbers

## Combine correlation coefficients into one file
for ii in range(0,num_flares):
	filename = './cor_err-gam_flare/cor_err-gam_flare'+str(flare_num[ii]).zfill(4)+'_'+str(out_num).zfill(3)+'.dat'
	read_in = np.loadtxt(filename)
	
	time = read_in[0]
	data = read_in[1:]
	
	if ii == 0:
		data_all = data
	else:
		data_all = np.append(data_all, data, axis=0)

## Attach time lag array
time = np.array([time])
output_arr = np.append(time, data_all, axis=0)

## Output
output_filename = 'object'+str(obj_num).zfill(4)+'-gam_flare'+str(iout).zfill(3)+'.dat'
np.savetxt(output_filename, output_arr)

