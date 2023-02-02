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

import osprey
osprey.start()

filename = '2RL0.pdb'
epsilon  = 0.99
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
