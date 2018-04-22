#!/bin/bash
<<COMMENT
Required Files:
1. This script 
2. The .pdb protein structure file
COMMENT
#------------------------------------------------------------------
#Prepare The Protein File
mkdir -p Gromacs-Simulation/RawData
mv *.pdb protein.pdb
mv protein.pdb Gromacs-Simulation/RawData
cd Gromacs-Simulation/RawData

#Construct The .mdp Files
echo '
integrator	= steep					; Algorithm (steep = steepest descent minimization)
emtol		= 1000.0				; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      	= 0.01					; Energy step size
nsteps		= 50000					; Maximum number of (minimization) steps to perform
; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist		= 1					; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet
ns_type		= grid					; Method to determine neighbor list (simple, grid)
coulombtype	= PME					; Treatment of long range electrostatic interactions
rcoulomb	= 1.0					; Short-range electrostatic cut-off
rvdw		= 1.0					; Short-range Van der Waals cut-off
pbc		= xyz					; Periodic Boundary Conditions (yes/no)
'>>ions.mdp

echo '
integrator	= steep					; Algorithm (steep = steepest descent minimization)
emtol		= 1000.0				; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep     	= 0.01					; Energy step size
nsteps		= 50000					; Maximum number of (minimization) steps to perform
; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist		= 1					; Frequency to update the neighbor list and long range forces
cutoff-scheme   = Verlet	
ns_type		= grid					; Method to determine neighbor list (simple, grid)
coulombtype	= PME					; Treatment of long range electrostatic interactions
rcoulomb	= 1.0					; Short-range electrostatic cut-off
rvdw		= 1.0					; Short-range Van der Waals cut-off
pbc		= xyz					; Periodic Boundary Conditions (yes/no)
'>>minim.mdp

echo '
define			= -DPOSRES			; position restrain the protein
; Run parameters
integrator		= md				; leap-frog integrator
nsteps			= 50000				; 2 * 50000 = 100 ps
dt			= 0.002				; 2 fs
; Output control
nstxout			= 500				; save coordinates every 1.0 ps
nstvout			= 500				; save velocities every 1.0 ps
nstenergy		= 500				; save energies every 1.0 ps
nstlog			= 500				; update log file every 1.0 ps
; Bond parameters
continuation	        = no				; first dynamics run
constraint_algorithm    = lincs				; holonomic constraints 
constraints		= all-bonds			; all bonds (even heavy atom-H bonds) constrained
lincs_iter		= 1				; accuracy of LINCS
lincs_order		= 4				; also related to accuracy
; Neighborsearching
cutoff-scheme   	= Verlet			
ns_type		   	= grid				; search neighboring grid cells
nstlist		    	= 10				; 20 fs, largely irrelevant with Verlet
rcoulomb	    	= 1.0				; short-range electrostatic cutoff (in nm)
rvdw		    	= 1.0				; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype	    	= PME				; Particle Mesh Ewald for long-range electrostatics
pme_order	    	= 4				; cubic interpolation
fourierspacing		= 0.16				; grid spacing for FFT
; Temperature coupling is on
tcoupl			= V-rescale			; modified Berendsen thermostat
tc-grps			= Protein Non-Protein		; two coupling groups - more accurate
tau_t			= 0.1	  0.1			; time constant, in ps
ref_t			= 300 	  300			; reference temperature, one for each group, in K
; Pressure coupling is off
pcoupl			= no				; no pressure coupling in NVT
; Periodic boundary conditions
pbc			= xyz				; 3-D PBC
; Dispersion correction
DispCorr		= EnerPres			; account for cut-off vdW scheme
; Velocity generation
gen_vel			= yes				; assign velocities from Maxwell distribution
gen_temp		= 300				; temperature for Maxwell distribution
gen_seed		= -1				; generate a random seed
'>>nvt.mdp

echo '
define			= -DPOSRES			; position restrain the protein
; Run parameters
integrator		= md				; leap-frog integrator
nsteps			= 50000				; 2 * 50000 = 100 ps
dt		    	= 0.002				; 2 fs
; Output control
nstxout			= 500				; save coordinates every 1.0 ps
nstvout			= 500				; save velocities every 1.0 ps
nstenergy		= 500				; save energies every 1.0 ps
nstlog			= 500				; update log file every 1.0 ps
; Bond parameters
continuation	        = yes				; Restarting after NVT 
constraint_algorithm    = lincs				; holonomic constraints 
constraints		= all-bonds			; all bonds (even heavy atom-H bonds) constrained
lincs_iter		= 1				; accuracy of LINCS
lincs_order		= 4				; also related to accuracy
; Neighborsearching
cutoff-scheme   	= Verlet			
ns_type		   	= grid				; search neighboring grid cells
nstlist		   	= 10				; 20 fs, largely irrelevant with Verlet scheme
rcoulomb	    	= 1.0				; short-range electrostatic cutoff (in nm)
rvdw		    	= 1.0				; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype	    	= PME				; Particle Mesh Ewald for long-range electrostatics
pme_order	    	= 4				; cubic interpolation
fourierspacing		= 0.16				; grid spacing for FFT
; Temperature coupling is on
tcoupl			= V-rescale			; modified Berendsen thermostat
tc-grps			= Protein Non-Protein		; two coupling groups - more accurate
tau_t			= 0.1	  0.1			; time constant, in ps
ref_t			= 300 	  300			; reference temperature, one for each group, in K
; Pressure coupling is on
pcoupl		        = Parrinello-Rahman		; Pressure coupling on in NPT
pcoupltype	        = isotropic			; uniform scaling of box vectors
tau_p		        = 2.0				; time constant, in ps
ref_p		        = 1.0				; reference pressure, in bar
compressibility     	= 4.5e-5			; isothermal compressibility of water, bar^-1
refcoord_scaling    	= com				
; Periodic boundary conditions
pbc			= xyz				; 3-D PBC
; Dispersion correction
DispCorr		= EnerPres			; account for cut-off vdW scheme
; Velocity generation
gen_vel			= no				; Velocity generation is off 
'>>npt.mdp

echo '
; Run parameters
integrator		= md				; leap-frog integrator
nsteps			= 500000			; 2 * 500000 = 1000 ps (1 ns)
dt		    	= 0.002				; 2 fs
; Output control
nstxout		        = 5000				; save coordinates every 10.0 ps
nstvout		        = 5000				; save velocities every 10.0 ps
nstenergy	        = 5000				; save energies every 10.0 ps
nstlog		        = 5000				; update log file every 10.0 ps
nstxout-compressed  	= 5000				; save compressed coordinates every 10.0 ps
; nstxout-compressed replaces nstxtcout
compressed-x-grps   	= System			; replaces xtc-grps
; Bond parameters
continuation	        = yes				; Restarting after NPT 
constraint_algorithm    = lincs				; holonomic constraints 
constraints		= all-bonds			; all bonds (even heavy atom-H bonds) constrained
lincs_iter		= 1				; accuracy of LINCS
lincs_order		= 4				; also related to accuracy
; Neighborsearching
cutoff-scheme   	= Verlet			
ns_type		    	= grid				; search neighboring grid cells
nstlist		    	= 10				; 20 fs, largely irrelevant with Verlet scheme
rcoulomb	    	= 1.0				; short-range electrostatic cutoff (in nm)
rvdw		    	= 1.0				; short-range van der Waals cutoff (in nm)
; Electrostatics
coulombtype	    	= PME				; Particle Mesh Ewald for long-range electrostatics
pme_order	    	= 4				; cubic interpolation
; Temperature coupling is on
fourierspacing		= 0.16				; grid spacing for FFT
tcoupl			= V-rescale			; modified Berendsen thermostat
tc-grps			= Protein Non-Protein		; two coupling groups - more accurate
tau_t			= 0.1	  0.1			; time constant, in ps
ref_t			= 300 	  300			; reference temperature, one for each group, in K
; Pressure coupling is on
pcoupl		        = Parrinello-Rahman		; Pressure coupling on in NPT
pcoupltype	        = isotropic			; uniform scaling of box vectors
tau_p		        = 2.0				; time constant, in ps
ref_p		        = 1.0				; reference pressure, in bar
compressibility     	= 4.5e-5			; isothermal compressibility of water, bar^-1
; Periodic boundary conditions
pbc			= xyz				; 3-D PBC
; Dispersion correction
DispCorr		= EnerPres			; account for cut-off vdW scheme
; Velocity generation
gen_vel			= no				; Velocity generation is off 
'>>md.mdp

#Prepare The System
printf %s\\n 15 | gmx pdb2gmx -f protein.pdb -o protein.gro -water spce 
echo '---------------------------'
echo 'Does the system have extra charge?'
read -p 'Number of extra negative charge>' PC
read -p 'Number of extra positive charge>' NC
gmx editconf -f protein.gro -o protein_newbox.gro -c -d 1.0 -bt cubic
gmx solvate -cp protein_newbox.gro -cs spc216.gro -o protein_solv.gro -p topol.top
gmx grompp -f ions.mdp -c protein_solv.gro -p topol.top -o ions.tpr
printf %s\\n 13 | gmx genion -s ions.tpr -o protein_solv_ions.gro -p topol.top -pname NA -nname CL -nn $NC -np $PC
gmx grompp -f minim.mdp -c protein_solv_ions.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em
gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt
gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -v -deffnm npt
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr

#Run Simulation
echo "
#!/bin/bash
#PBS -N Gromacs
#PBS -q fat
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -j oe

cd $PBS_O_WORKDIR

gmx mdrun -v -deffnm md

printf %s\\n 0 | gmx trjconv -s em.tpr -f md_noPBC.xtc -o Simulation.pdb
mv Simulation.pdb Gromacs-Simulation
printf %s\\n 10 0 | gmx energy -f em.edr -o potential.xvg
sed -i 's/@/#/g' potential.xvg
printf %s\\n 15 0 | gmx energy -f nvt.edr -o temperature.xvg
sed -i 's/@/#/g' temperature.xvg
printf %s\\n 15 0 | gmx energy -f npt.edr -o pressure.xvg
sed -i 's/@/#/g' pressure.xvg
printf %s\\n 22 0 | gmx energy -f npt.edr -o density.xvg
sed -i 's/@/#/g' density.xvg
printf %s\\n 0 | gmx trjconv -s md.tpr -f md.xtc -o md_noPBC.xtc -pbc mol -ur compact
printf %s\\n 4 \n 4 | gmx rms -s md.tpr -f md_noPBC.xtc -o rmsd.xvg -tu ns
sed -i 's/@/#/g' rmsd.xvg
printf %s\\n 4 \n 4 | gmx rms -s em.tpr -f md_noPBC.xtc -o rmsd_xtal.xvg -tu ns
sed -i 's/@/#/g' rmsd_xtal.xvg
printf %s\\n 4 | gmx gyrate -s md.tpr -f md_noPBC.xtc -o gyrate.xvg
sed -i 's/@/#/g' gyrate.xvg
">>simulation.pbs

#qsub simulation.pbs
#gmx mdrun -deffnm md_0_1 -nb gpu	#to run on GPU
<<COMMENT
gnuplot
plot 'potential.xvg' w l
plot 'temperature.xvg' w l
plot 'pressure.xvg' w l
plot 'density.xvg' w l
plot 'rmsd.xvg' w l
plot 'rmsd_xtal.xvg' w l
plot 'gyrate.xvg' w l
COMMENT
