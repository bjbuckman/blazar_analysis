#PBS -l nodes=1:ppn=40
#PBS -l walltime=125:39:21
#PBS -N run_bayesian_spectrum3
#PBS -j oe
#PBS -A PCON0003

# export OMP_NUM_THREADS=40
# set -x

module load python/3.6-conda5.2 ## LOAD PYTHON3
export PYTHONPATH=$HOME/python3/:$HOME/python3/lib/python3.6/site-packages/

## directories
analysis_dir=$HOME/projects/gamma-optical/analysis
data_dir=$HOME/projects/gamma-optical/data
pyscript_dir=$HOME/projects/gamma-optical/analysis/pytools

cd $TMPDIR ## working directory
# mkdir cross_object_opt+opt ## directory with all files

# object_file=/users/PCON0003/cond0064/projects/gamma-optical/data/objects_with_data.dat
# cp $object_file .

#optical objects to go through
obj_min=320
obj_max=479

num_process=0
max_process=40
for ii in $(seq $obj_min $obj_max)
do
	object_num=$(printf %04d $ii)
	
	gam_file=$data_dir/gamma-ray/v3/object${object_num}_gam.npy
	energy_file=$data_dir/gamma-ray/v3/object${object_num}_energy.dat

	/usr/bin/time python $pyscript_dir/calc_bayesian_gamma_spectra_v2.py $ii $gam_file $energy_file &
	(( num_process += 1))
	
	if [ "$num_process" -ge "$max_process" ]; then
		wait
		num_process=0
	fi
	
done
wait

# python $pyscript_dir/cross_object_autocorr_combine_opt_v1.py $opt_min $opt_max $gam_num $iout 

##copy files to scratch
SCRATCHDATDIR=/fs/scratch/PCON0003/cond0064/gamma-ray_bayesian_spectrum
if [ ! -d "$SCRATCHDATDIR" ]; then mkdir "$SCRATCHDATDIR"; fi
cp * "$SCRATCHDIR"
# cp *bayesian_spectrum.dat "$SCRATCHDATDIR"

##copy final files to folder
SAVEDIR=$analysis_dir/objects/gamma-ray_bayesien_spectrum
if [ ! -d "$SAVEDIR" ]; then mkdir "$SAVEDIR"; fi
cp *bayesian_spectrum.dat "$SAVEDIR"
