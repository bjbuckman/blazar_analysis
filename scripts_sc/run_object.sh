#PBS -l nodes=1:ppn=20
#PBS -l walltime=155:37:24
#PBS -N run_object
#PBS -j oe
#PBS -A PCON0003

OBJECT_NUM=1
object_num=$(printf %04d $OBJECT_NUM)

export OMP_NUM_THREADS=20
set -x

cd $TMPDIR
mkdir cor_err
mkdir cor_sig

analysis_dir=$HOME/projects/gamma-optical/analysis
data_dir=$HOME/projects/gamma-optical/data
opt_file=$data_dir/optical/object${object_num}_asas-sn.csv
gam_file=$data_dir/gamma-ray/object${object_num}_gam.dat

dt=0.45
DT=60.
num_iterations=100

module load python/3.6-conda5.2
#export PYTHONPATH=$HOME/python/

pyscript_dir=$HOME/projects/gamma-optical/analysis/pytools

imin=100
imax=109
for i in $(seq $imin $imax)
do
	/usr/bin/time python $pyscript_dir/calc_corr_coeff_err_v2.py $OBJECT_NUM $opt_file $gam_file $i $dt $DT $num_iterations &
	/usr/bin/time python $pyscript_dir/calc_significance_v2.py $OBJECT_NUM $opt_file $gam_file $i $dt $DT $num_iterations &
done
wait

iout=10
python $pyscript_dir/calc_combine_v2.py $OBJECT_NUM $imin $imax $iout 

#copy to scratch
mkdir /fs/scratch/PCON0003/cond0064/object${object_num}
cp -r ./cor_err ./cor_sig /fs/scratch/PCON0003/cond0064/object${object_num}/
cp *.dat /fs/scratch/PCON0003/cond0064/object${object_num}/

#copy to folder
cp *.dat $analysis_dir/objects/
