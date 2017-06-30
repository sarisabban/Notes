#!/bin/bash

<<COMMENT
Description:
------------
This is a Bash script that automatically sets the correct files to run Rosetta Abinitio (default for 25,000 decoys) followed by Clustering (lowest 200 scoring decoys) followed by plotting the computation result in a HPC (High Preformace Computer) that uses PBS as its job scheduler.
This script was written by Sari Sabban on 27-June-2017.

Required Input Files:
---------------------
1. aat000_03_05.200_v1_3 	---> The 3-mer fragment file
2. aat000_09_05.200_v1_3	---> The 9-mer fragment file
3. structure.pdb		---> The structure's PDB file
4. structure.fasta		---> The structure's sequence file in FASTA format
5. t000_.psipred_ss2		---> The PsiPred prediction file

How To Use:
-----------
1. Before using this script you must make sure it works in your HPC by running each section individually, job chaining (what this script does) can disrupt the HPC if run incorrectly. There are lines in this script that only works in specific supercomputers and not others.
2. It goes without saying that you need to download and compile the Rosetta modeling software [https://www.rosettacommons.org].
3. Identify the path to Rosetta and update this script using this command:
	sed -i 's^{ROSETTA}^PATH/TO/ROSETTA^g' Abinitio.bash
4. Make sure you have all the nessesary input files in the working directory.
5. Execute this script to generate all nessesary files and folders then submit the job for computation using this command:
	bash Abinitio.bash && qsub abinitio.pbs
6. This script is setup to run using the PBS job scheduler, simple changes can be made to make it work on other job schedulers, but thorough understading of each job scheduler is nessesary to make these modifications.
7. The default computation settings are for normal Abinitio computation (25,000 decoys). You will have to run the following command to setup the script for larger computations (1,000,000 decoys):
	sed -i '/#PBS -l walltime=9:00:00/d' Abinitio.bash && sed -i 's/thin/thin_1m/g' Abinitio.bash && sed -i 's/-nstruct 25/-nstruct 1000/g' Abinitio.bash
COMMENT
#---------------------------------------------------------------------------------------------------------------
cat << 'EOF' > abinitio.pbs
#!/bin/bash
#PBS -N Abinitio
#PBS -q thin
#PBS -l walltime=9:00:00
#PBS -l select=1:ncpus=1
#PBS -j oe
#PBS -J 1-1000

cd $PBS_O_WORKDIR
{ROSETTA}/main/source/bin/AbinitioRelax.default.linuxgccrelease -database {ROSETTA}/main/database -in:file:frag3 ./aat000_03_05.200_v1_3 -in:file:frag9 ./aat000_09_05.200_v1_3 -in:file:fasta ./structure.fasta -in:file:native ./structure.pdb -psipred_ss2 ./t000_.psipred_ss2 -nstruct 25 -abinitio:relax -use_filters true -abinitio::increase_cycles 10 -abinitio::rg_reweight 0.5 -abinitio::rsd_wt_helix 0.5 -abinitio::rsd_wt_loop 0.5 -relax::fast -out:file:silent ./fold_silent_${PBS_ARRAY_INDEX}.out

#qsub -W depend=afterok:${PBS_ARRAY_ID} cluster.pbs #<---- PROBLEM HERE
EOF

cat << 'EOF' > cluster.pbs
#!/bin/bash
#PBS -N Clustering
#PBS -q thin
#PBS -l walltime=9:00:00
#PBS -l select=1:ncpus=1
#PBS -j oe

cd $PBS_O_WORKDIR
{ROSETTA}/main/source/bin/relax.default.linuxgccrelease -database {ROSETTA}/main/database -s ./structure.pdb -native ./structure.pdb -relax:thorough -in:file:fullatom -nooutput -nstruct 100 -out:file:silent ./relax.out
grep SCORE ./relax.out | awk '{print $20 "\t" $2}' > ./relax.dat
{ROSETTA}/main/source/bin/combine_silent.default.linuxgccrelease -in:file:silent ./fold_silent_*.out -out:file:silent ./fold.out
grep SCORE ./fold.out | awk '{print $28 "\t" $2}' > ./fold.dat
tail -n +2 "./fold.dat" > "./fold.dat.tmp" && mv "./fold.dat.tmp" "./fold.dat"
mkdir ./cluster
grep SCORE ./fold.out | sort -nk +2 | head -200 | awk '{print $31}' > ./list
cat ./list | awk '{print}' ORS=" " > ./liststring
xargs {ROSETTA}/main/source/bin/extract_pdbs.linuxgccrelease -in::file::silent ./fold.out -out:pdb -in:file:tags < ./liststring
rm ./list
rm ./liststring
rm ./*.fsc
rm ./fold_silent_*
rm ./Abinitio.o*
mv *_*.pdb ./cluster
cd ./cluster
{ROSETTA}/main/source/bin/cluster.default.linuxgccrelease -database {ROSETTA}/main/database -in:file:fullatom -cluster:radius 3 -nooutput -out:file:silent ./cluster.out -in:file:s ./*.pdb
rm ./*.pdb
{ROSETTA}/main/source/bin/extract_pdbs.linuxgccrelease -in::file::silent ./cluster.out -out:pdb -in:file:tags
cd ..
echo "set terminal postscript
set output './plot.pdf'
set encoding iso_8859_1
set term post eps enh color
set xlabel 'RMSD (\305)'
set ylabel 'Score'
set yrange [:-80]
set xrange [0:20]
set title 'Abinitio Result'
plot './fold.dat' lc rgb 'red' pointsize 0.2 pointtype 7 title '', \
'./relax.dat' lc rgb 'green' pointsize 0.2 pointtype 7 title ''
exit" > gnuplot_sets
gnuplot < gnuplot_sets
rm gnuplot_sets
EOF
