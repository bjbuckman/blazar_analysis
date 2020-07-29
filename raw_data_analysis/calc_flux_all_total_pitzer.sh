#PBS -l nodes=1:ppn=40
#PBS -l walltime=145:39:21
#PBS -N calc_fluxes_all
#PBS -j oe
#PBS -A PCON0003

# export OMP_NUM_THREADS=40
# set -x

module load python/3.6-conda5.2
export PYTHONPATH=$HOME/python3/:$HOME/python3/lib/python3.6/site-packages/

fermi_dir=/users/PCON0003/cond0064/projects/gamma-optical/data/fermi_analysis

# Program
calc_flux() {
	local FILE=$1
	
	# echo $FILE
	local NAME=${FILE:10}
	local NAME=${NAME/.tar.gz/}
	
	cd $fermi_dir/sources
	
	mkdir $TMPDIR/work_dir${NAME}
	cp $fermi_dir/step11_calc_total_flux.py $TMPDIR/work_dir${NAME}
	cp ${NAME}.tar.gz ${NAME}_out7_rois.txt $TMPDIR/work_dir${NAME}
	cd $TMPDIR/work_dir${NAME}
	
	tar -xf ${NAME}.tar.gz
	
	python step11_calc_total_flux.py ${NAME}_out7_rois.txt >> ${NAME}_fermi_tot.dat
	cp *.dat *.npy $fermi_dir/fluxes/vT
	echo $NAME
}


cd $fermi_dir
ii=0
for file in ./sources/4FGL_J*.tar.gz
do
	if [ $ii -lt 40 ]
	then
		calc_flux ${file} &
		((ii+=1))
	else
		wait
		calc_flux ${file} &
		ii=1
	fi
done
wait




