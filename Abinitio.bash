#!/bin/bash

<<COMMENT
This is a Bash script that automatically sets the correct files to run Rosetta Abinitio followed by Clustering followed by plotting the computation result in a HPC (High Preformace Computer) that uses PBS as its job scheduler.
Written by Sari Sabban on 27-June-2017.

Required Input Files:
1. aat000_03_05.200_v1_3
2. aat000_09_05.200_v1_3
3. structure.pdb
4. structure.fasta
5. t000_.psipred_ss2

How To Use:
1. Before using this script make sure it works in your HPC by running each section individually, job chaining (what this script does) can disrupt the HPC if run incorrectly.
2. Find all nessesary paths in this script and change them using this command:
	sed -i 's^{ROSETTA}^PATH/TO/ROSETTA^g' Abinitio.bash
	sed -i 's^{PATH}^PATH/TO/FILES^g' Abinitio.bash
	sed -i 's^{HOME}^PATH/TO/HOME^g' Abinitio.bash
3. Make sure you have all the nessesary input files in the working directory.
4. Execute this script to generate all nessesary files and folders using this command:
	bash Abinitio.bash
5. Submit job for computation using this command:
	qsub abinitio.pbs
6. This script is setup to run using the PBS job scheduler, simple changes can be made to make it work on other job schedulers, but thorough understading of each job scheduler is nessesary to make these modifications.
COMMENT
#---------------------------------------------------------------------------------------------------------------
echo '-database {ROSETTA}/main/database
-in:file:frag3 {PATH}/aat000_03_05.200_v1_3
-in:file:frag9 {PATH}/aat000_09_05.200_v1_3
-in:file:fasta {PATH}/structure.fasta
-in:file:native {PATH}/structure.pdb
-psipred_ss2 {PATH}/t000_.psipred_ss2
-nstruct 1
-abinitio:relax
-use_filters true
-abinitio::increase_cycles 10
-abinitio::rg_reweight 0.5
-abinitio::rsd_wt_helix 0.5
-abinitio::rsd_wt_loop 0.5
-relax::fast
-out:file:silent {PATH}/fold_silent_${PBS_ARRAY_INDEX}.out' > flags

echo '#!/bin/bash
#PBS -N Abinitio
#PBS -q thin
#PBS -l walltime=9:00:00
#PBS -l select=1:ncpus=1
#PBS -j oe
#PBS -J 1-10

{ROSETTA}/main/source/bin/AbinitioRelax.default.linuxgccrelease @{PATH}/flags

qsub cluster.pbs -W depend=afterokarray:${PBS_JOBID}'>abinitio.pbs

cat << 'EOF' > cluster.pbs
#!/bin/bash
#PBS -N Clustering
#PBS -q thin
#PBS -l select=1:ncpus=1
#PBS -j oe

{ROSETTA}/main/source/bin/relax.default.linuxgccrelease -database {ROSETTA}/main/database -s {PATH}/structure.pdb -relax:thorough -nooutput -nstruct 100 -out:file:silent {PATH}/relax.out
grep SCORE {PATH}/relax.out | awk '{print $27 "\t" $2}' > {PATH}/relax.dat
{ROSETTA}/main/source/bin/combine_silent.default.linuxgccrelease -in:file:silent {PATH}/fold_silent_*.out -out:file:silent {PATH}/fold.out
grep SCORE {PATH}/fold.out | awk '{print $27 "\t" $2}' > {PATH}/fold.dat
tail -n +2 "{PATH}/fold.dat" > "{PATH}/fold.dat.tmp" && mv "{PATH}/fold.dat.tmp" "{PATH}/fold.dat"
mkdir {PATH}/cluster
grep SCORE {PATH}/fold.out | sort -nk +2 | head -200 | awk '{print $30}' > {PATH}/list
cat {PATH}/list | awk '{print}' ORS=" " > {PATH}/liststring
xargs {ROSETTA}/main/source/bin/extract_pdbs.linuxgccrelease -in::file::silent {PATH}/fold.out -out:pdb -in:file:tags < {PATH}/liststring
rm {PATH}/list
rm {PATH}/liststring
rm {HOME}/*.fsc
rm {PATH}/fold_silent_*
rm {PATH}/Abinitio.o*
mv S_* {PATH}/cluster
cd {PATH}/cluster
echo '-database {ROSETTA}/main/database
-in:file:fullatom
-cluster:radius 3
-nooutput
-out:file:silent {PATH}/cluster/cluster.out' > {PATH}/cluster/flags
{ROSETTA}/main/source/bin/cluster.default.linuxgccrelease @{PATH}/cluster/flags -in:file:s {PATH}/cluster/*.pdb
rm {PATH}/cluster/*.pdb
{ROSETTA}/main/source/bin/extract_pdbs.linuxgccrelease -in::file::silent {PATH}/cluster/cluster.out -out:pdb -in:file:tags
cd {PATH}
gnuplot
set terminal postscript
set output '{PATH}/plot.pdf'
set encoding iso_8859_1
set xlabel "RMSD (\305)"
set ylabel 'Score'
set yrange [:-80]
set xrange [0:20]
set title 'Abinitio Result'
plot '{PATH}/fold.dat' lc rgb 'red' pointsize 0.2 pointtype 7 title '', \
'{PATH}/relax.dat' lc rgb 'green' pointsize 0.2 pointtype 7 title ''
exit
