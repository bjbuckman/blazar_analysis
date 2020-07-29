import numpy as np
import os

init_file = 'run_object'
source_filename = init_file+'.sh'

suffix = 'aa'

##############################################
##############################################
##############################################

MODEL_NUMBER = 1

if MODEL_NUMBER == 1:
	num_params = 2
	num_arr = np.zeros(num_params).astype(int)
	num_arr[0] = 10 #
	num_arr[1] = 5 #
	num_arr_tot = int(np.prod(num_arr))
	
	num_start = np.zeros(num_params).astype(int)
	num_start[0] = 10
	num_start[1] = 10
	
	## argument 1
	arr_1_min = 1
	arr_1_max = 1
	arr_1_diff = 1
	
	arr_1 = []
	for ii in range(0,num_arr[0]):
		arr_1.append(arr_1_min+arr_1_diff)
	
	## argument 2
	arr_2_min = 1.
	arr_2_max = 5000.
	arr_2_diff = np.log10(arr_2_max/arr_1_min)/(num_arr[1]-1)
	
	arr_2 = []
	for ii in range(0,num_arr[1]):
		arr_2.append(arr_2_min*10.**(arr_2_diff*ii))

new_val = []
new_val.append(arr_1)
new_val.append(arr_2)

print('# '+str(arr_1))
print('# '+str(arr_2))


linechange = [] 
newline = [] 
endline = []

#LINES PARAM1
cline = []
nline = []
eline = []

cline.append('OBJECT_NUM')
nline.append('OBJECT_NUM=') 
eline.append('')

linechange.append(cline)
newline.append(nline)
endline.append(eline)

#LINES PARAM2
cline = []
nline = []
eline = []

cline.append('DT')
nline.append('DT=') 
eline.append('')

linechange.append(cline)
newline.append(nline)
endline.append(eline)




source_file = open(source_filename,'r')

if not os.path.isdir('./'+init_file+suffix):
	os.mkdir('./'+init_file+suffix)

index=np.zeros([num_arr_tot,num_params])
index_n=index
N=np.zeros(num_params).astype(int)

for nat in range(0,num_arr_tot):	
	#Getting suffix for files
	N_index=N+num_start
	N_suffix = ''
	for s in range(0,num_params):
		N_suffix += str(N_index[s])
	
	#New file
	new_filename = init_file+suffix+N_suffix
	new_file = open('./'+init_file+suffix+'/'+new_filename+'.sh', 'w')
	
	#Writing new file
	source_file.seek(0)
	for line in source_file:
		for ii in range(0,len(linechange)):
			for ij in range(0,len(linechange[ii])):
				if line.startswith(linechange[ii][ij]):
					line = newline[ii][ij]+str(new_val[ii][N[ii]])+endline[ii][ij]+'\n'
				if line.startswith('#PBS -N'):
					line = '#PBS -N '+new_filename+'\n'
		new_file.write(line)
	new_file.close()
	
	print('qsub '+new_filename+'.sh'+'\n')

	#Update N
	N[num_params-1] += 1
	for i in (num_params-1 -np.array(range(0,num_params))):
		if N[i] >= num_arr[i]:
			N[i] = 0
			if i >= 1:
				N[i-1] += 1
	
source_file.close()


	
	
	