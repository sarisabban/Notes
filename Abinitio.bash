#!/bin/bash

<<COMMENT
This is a Bash script that automatically sets the correct files to run Rosetta Abinitio followed by Clustering on a HPC (High Preformace Computer) that uses PBS as its job scheduler.
Written by Sari Sabban on 27-June-2017

Input Files:
1. aat000_03_05.200_v1_3
2. aat000_09_05.200_v1_3
3. structure.pdb
4. structure.fasta
5. t000_.psipred_ss2

How To Use:
1. Before using this script make sure it works in your HPC by running each section individually, job chaining (what this script does) can disrupt the HPC is run incorrectly.
2. Find all nessesary paths in this script and change them using this command in vim
	:%s^{ROSETTA}^PATH/TO/ROSETTA
	:%s^{PATH}^PATH/TO/FILES
3. Make sure you have all the nessesary input files in the working directory
4. Execute this script to generate all nessesary files and folders using this command
	bash Abinitio.bash
5. Submit job for computation using this command
	qsub abinitio.pbs
COMMENT

echo '-database {ROSETTA}/main/database
-in:file:frag3 {PATH}/aat000_03_05.200_v1_3
-in:file:frag9 {PATH}/aat000_09_05.200_v1_3
-in:file:fasta {PATH}/structure.fasta
-in:file:native {PATH}/structure.pdb
-psipred_ss2 {PATH}/t000_.psipred_ss2
-nstruct 25
-abinitio:relax
-use_filters true
-abinitio::increase_cycles 10
-abinitio::rg_reweight 0.5
-abinitio::rsd_wt_helix 0.5
-abinitio::rsd_wt_loop 0.5
-relax::fast
-out:file:silent {PATH}/fold_silent_${PBS_ARRAY_INDEX}.out' > fold_flags

echo '#!/bin/bash
#PBS -N Abinitio
#PBS -q thin
#PBS -l walltime=9:00:00
#PBS -l select=1:ncpus=1
#PBS -j oe
#PBS -J 1-1000

{ROSETTA}/main/source/bin/AbinitioRelax.default.linuxgccrelease @{PATH}/fold_flags

qsub cluster.bash -W depend=afterokarray:${PBS_JOBID}'>abinitio.pbs

cat << 'EOF' > cluster.bash
#!/bin/bash
{ROSETTA}/main/source/bin/combine_silent.default.linuxgccrelease -in:file:silent fold_silent_*.out -out:file:silent fold_silent.out
grep SCORE fold_silent.out | awk '{print $27 "\t" $2}' > SCOREvsRMSD.dat
tail -n +2 "SCOREvsRMSD.dat" > "SCOREvsRMSD.dat.tmp" && mv "SCOREvsRMSD.dat.tmp" "SCOREvsRMSD.dat"
mkdir cluster
grep SCORE fold_silent.out | sort -nk +2 | head -200 | awk '{print $30}' > list
cat list | awk '{print}' ORS=" " > liststring
xargs {ROSETTA}/main/source/bin/extract_pdbs.linuxgccrelease -in::file::silent fold_silent.out -out:pdb -in:file:tags < liststring
rm list
rm liststring
rm /home/ssabban/*.fsc
rm fold_silent_*
rm Abinitio.o*
mv S_* cluster
cd cluster
echo '-database {ROSETTA}/main/database
-in:file:fullatom
-cluster:radius 3
-nooutput
-out:file:silent cluster_silent.out' > cluster_flags
{ROSETTA}/main/source/bin/cluster.default.linuxgccrelease @cluster_flags -in:file:s *.pdb
rm *.pdb
{ROSETTA}/main/source/bin/extract_pdbs.linuxgccrelease -in::file::silent cluster_silent.out -out:pdb -in:file:tags
