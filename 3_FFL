#!/bin/bash

<<COMMENT
Input:
1. 3mer file
2. 9mer file
3. psipred_ss2 file
4. motif.pdb
5. scaffold.pdb
COMMENT

grep " B " structure.pdb > scaffold_raw.pdb
perl {PATH}/sequentialPdbResSeq.pl -pdbfile scaffold_raw.pdb > scaffold.pdb
rm scaffold_raw.pdb
{ROSETTA}/tools/perl_tools/getFastaFromCoords.pl -pdbfile scaffold.pdb -chain B > scaffold.fasta

echo '
LOOP 61 66 0 0.0 X
'>input.loop

echo '
-s {PATH}/scaffold.pdb
-loops::loop_file {PATH}/input.loop
-loops::frag_sizes 9 3 1
-in:file:frag3 {PATH}/aat000_03_05.200_v1_3
-in:file:frag9 {PATH}/aat000_09_05.200_v1_3
-fold_from_loops::add_relax_cycles 2
-fold_from_loops::swap_loops {PATH}/motif.pdb
-fold_from_loops:res_design_bs 1 6
-in::file::psipred_ss2 {PATH}/t000_.psipred_ss2
-fold_from_loops::loop_mov_nterm 2
-fold_from_loops::loop_mov_cterm 2
-fold_from_loops::ca_rmsd_cutoff 1.5
-abinitio::steal_3mers
-abinitio::steal_9mers
-fold_from_loops::native_ca_cst
-fold_from_loops::ca_csts_dev 3.0
-out::prefix FFL
-out::nstruct 10
-out::file::silent_struct_type binary
-out::file::silent {PATH}/FFL_silent_${PBS_ARRAY_INDEX}.out
-mute all
'>flags

echo '
#!/bin/bash
#PBS -N FFL
#PBS -q fat_1m
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1
#PBS -j oe
#PBS -J 1-1000

{ROSETTA}/main/source/bin/fold_from_loops_devel.default.linuxgccrelease @{PATH}/flags
'>FFL.pbs
