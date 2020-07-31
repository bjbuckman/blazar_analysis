#PBS -l nodes=1:ppn=40
#PBS -l walltime=125:39:21
#PBS -N run_bayesian_block_cc
#PBS -j oe
#PBS -A PCON0003

# export OMP_NUM_THREADS=40
# set -x

module load python/3.6-conda5.2 ## LOAD PYTHON3
export PYTHONPATH=$HOME/python3/:$HOME/python3/lib/python3.6/site-packages/

## directories
analysis_dir=$HOME/projects/gamma-optical/analysis
cc_dir=$HOME/projects/gamma-optical/analysis/objects/cross_corr
data_dir=$HOME/projects/gamma-optical/data
pyscript_dir=$HOME/projects/gamma-optical/analysis/pytools

cd $TMPDIR ## working directory
# mkdir cross_object_opt+opt ## directory with all files

# object_file=/users/PCON0003/cond0064/projects/gamma-optical/data/objects_with_data.dat
# cp $object_file .

#optical objects to go through
obj_min=0
obj_max=159

cc_file_num=0
cc_file_num3=$(printf %03d $cc_file_num)

out_num=0

num_process=0
max_process=40
for ii in $(seq $obj_min $obj_max)
do
	object_num=$(printf %04d $ii)
	
	cc_file=$cc_dir/object${object_num}_stats${cc_file_num3}.dat

	/usr/bin/time python $pyscript_dir/calc_bayesian_block_cc_v1.py $ii $cc_file $out_num &
	(( num_process += 1))
	
	if [ "$num_process" -ge "$max_process" ]; then
		wait
		num_process=0
	fi
	
done
wait

# python $pyscript_dir/cross_object_autocorr_combine_opt_v1.py $opt_min $opt_max $gam_num $iout 

##copy files to scratch
SCRATCHDATDIR=/fs/scratch/PCON0003/cond0064/bayesian_block_cc
if [ ! -d "$SCRATCHDATDIR" ]; then mkdir "$SCRATCHDATDIR"; fi
cp * "$SCRATCHDATDIR"
# cp *bayesian_spectrum.dat "$SCRATCHDATDIR"

##copy final files to folder
SAVEDIR=$analysis_dir/objects/bayesian_block_cc
if [ ! -d "$SAVEDIR" ]; then mkdir "$SAVEDIR"; fi
cp *.dat "$SAVEDIR"
