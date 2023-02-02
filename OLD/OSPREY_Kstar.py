'''
# Install OSPREY Python:
# ======================
apt install python3-venv
python3 -m venv myenv
source myenv/bin/activate
mkdir software
cd software
wget -q https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
tar -xf openjdk-17.0.2_linux-x64_bin.tar.gz
rm openjdk-17.0.2_linux-x64_bin.tar.gz
touch .bash_profile
echo 'export JAVA_HOME=$HOME/software/jdk-17.0.2' >> $HOME/.bash_profile
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> $HOME/.bash_profile
source $HOME/.bash_profile
java -version
git clone https://github.com/donaldlab/OSPREY3.git
cd OSPREY3/
sed -i s/'"--user", "--editable",'/'"--editable",'/ ./buildSrc/src/main/kotlin/osprey/python.kt
./gradlew assemble
./gradlew pythonDevelop
cd
'''

'''
# apt install gromacs

P=2RL0

echo '
integrator           = steep               ; Algorithm (steep = steepest descent minimization)
emtol                = 1000.0              ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep               = 0.01                ; Energy step size
nsteps               = 50000               ; Maximum number of (minimization) steps to perform
; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist              = 1                   ; Frequency to update the neighbor list and long range forces
cutoff-scheme        = Verlet
ns_type              = grid                ; Method to determine neighbor list (simple, grid)
coulombtype          = PME                 ; Treatment of long range electrostatic interactions
rcoulomb             = 1.0                 ; Short-range electrostatic cut-off
rvdw                 = 1.0                 ; Short-range Van der Waals cut-off
pbc                  = xyz                 ; Periodic Boundary Conditions (yes/no)
'>>ions.mdp

echo '
integrator           = steep               ; Algorithm (steep = steepest descent minimization)
emtol                = 1000.0              ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep               = 0.01                ; Energy step size
nsteps               = 50000               ; Maximum number of (minimization) steps to perform
; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist              = 1                   ; Frequency to update the neighbor list and long range forces
cutoff-scheme        = Verlet
ns_type              = grid                ; Method to determine neighbor list (simple, grid)
coulombtype          = PME                 ; Treatment of long range electrostatic interactions
rcoulomb             = 1.0                 ; Short-range electrostatic cut-off
rvdw                 = 1.0                 ; Short-range Van der Waals cut-off
pbc                  = xyz                 ; Periodic Boundary Conditions (yes/no)
'>>minim.mdp

printf %s\\n 3 | gmx pdb2gmx -f $P.pdb -o $P.gro -water spce -ignh
gmx editconf -f $P.gro -o $P\_box.gro -c -d 1.0 -bt dodecahedron
gmx solvate -cp $P\_box.gro -cs spc216.gro -o $P\_solv.gro -p topol.top
gmx grompp -f ions.mdp -c $P\_solv.gro -p topol.top -o ions.tpr -maxwarn 1
printf %s\\n 13 | gmx genion -s ions.tpr -o $P\_solv\_ions.gro -p topol.top -pname NA -nname CL -neutral
gmx grompp -f minim.mdp -c $P\_solv_ions.gro -p topol.top -o em.tpr
gmx mdrun -deffnm em
printf %s\\n 1 | gmx trjconv -s em.tpr -f em.gro -pbc mol -o $P\_\H\_min.pdb
rm *.gro *.edr *.log *.tpr *.trr *.mdp *.itp *.top \#topol*
'''

import osprey
osprey.start()

filename = '2RL0_H_min.pdb'
epsilon  = 0.999
CPUs     = 4

ffparams = osprey.ForcefieldParams()
mol = osprey.readPdb(filename)
templateLib = osprey.TemplateLibrary(ffparams.forcefld)

# define the protein strand
protein = osprey.Strand(mol, templateLib=templateLib, residues=['G648', 'G654'])
protein.flexibility['G649'].setLibraryRotamers(osprey.WILD_TYPE, 'TYR', 'ALA', 'VAL', 'ILE', 'LEU').addWildTypeRotamers().setContinuous()
protein.flexibility['G650'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
protein.flexibility['G651'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
protein.flexibility['G654'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()

# define the ligand strand
ligand = osprey.Strand(mol, templateLib=templateLib, residues=['A155', 'A194'])
ligand.flexibility['A156'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['A172'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['A192'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()
ligand.flexibility['A193'].setLibraryRotamers(osprey.WILD_TYPE).addWildTypeRotamers().setContinuous()

proteinConfSpace = osprey.ConfSpace(protein)
ligandConfSpace = osprey.ConfSpace(ligand)
complexConfSpace = osprey.ConfSpace([protein, ligand])
parallelism = osprey.Parallelism(cpuCores=CPUs)
ecalc = osprey.EnergyCalculator(complexConfSpace, ffparams, parallelism=parallelism)
kstar = osprey.KStar(
	proteinConfSpace,
	ligandConfSpace,
	complexConfSpace,
	epsilon=epsilon,
	writeSequencesToConsole=True,
	writeSequencesToFile='kstar.results.tsv')
for info in kstar.confSpaceInfos():
	eref = osprey.ReferenceEnergies(info.confSpace, ecalc)
	info.confEcalc = osprey.ConfEnergyCalculator(info.confSpace, ecalc, referenceEnergies=eref)
	emat = osprey.EnergyMatrix(info.confEcalc, cacheFile='emat.%s.dat' % info.id)
	def makePfunc(rcs, confEcalc=info.confEcalc, emat=emat):
		return osprey.PartitionFunction(
			confEcalc,
			osprey.AStarTraditional(emat, rcs, showProgress=False),
			osprey.AStarTraditional(emat, rcs, showProgress=False),
			rcs)
	info.pfuncFactory = osprey.KStar.PfuncFactory(makePfunc)
scoredSequences = kstar.run(ecalc.tasks)
analyzer = osprey.SequenceAnalyzer(kstar)
for scoredSequence in scoredSequences:
	print("result:")
	print("\tsequence: %s" % scoredSequence.sequence)
	print("\tK* score: %s" % scoredSequence.score)
	numConfs = 10
	analysis = analyzer.analyze(scoredSequence.sequence, numConfs)
	print(analysis)
	analysis.writePdb(
	'seq.%s.pdb' % scoredSequence.sequence,
	'Top %d conformations for sequence %s' % (numConfs, scoredSequence.sequence))
