import Bio.PDB , math , os , random , time
from pyrosetta import *
from pyrosetta.toolbox import *
init()
#-------------------------------------------------------------------
#Functions:
def DSSP():
	''' Measures the secondary structures of each amino acid in a protein '''
	''' [0] = Loop, [1] = Helix, [2] = Strand '''
	pose.dump_pdb('temp.pdb')
	structure = Bio.PDB.PDBParser().get_structure('X' , 'temp.pdb')
	DSSP1 = Bio.PDB.DSSP(structure[0] , 'temp.pdb' , acc_array = 'Wilke')
	SS = list()
	for ss in DSSP1:
		if ss[2] == 'S' or ss[2] == 'T' or ss[2] == '-':	#Loop
			SS.append('L')
		if ss[2] == 'G' or ss[2] == 'H' or ss[2] == 'I':	#Helix
			SS.append('H')
		if ss[2] == 'E' or ss[2] == 'B':			#Strand
			SS.append('S')
	Sec = ''.join(SS)
	Loop = SS.count('L')
	Helix = SS.count('H')
	Strand = SS.count('S')
	os.remove('temp.pdb')
	return((Sec , Loop , Helix , Strand))

def Decision(before , after , kt):
	''' Metropolis Criterion, P = 1 to accept all structures when final score is lower than starting score, P closer to 0 = accept less and less but sometimes accept to escape local energy minima '''
	''' Returns a string with either Accept or Reject '''
	E = after - before
	if E == 0:
		E = -1
		negE = (math.fabs(E)) * -1
		e = math.e
		P = e**(negE/kt)
	elif E < 0:
		negE = (math.fabs(E)) * -1
		e = math.e
		P = e**(negE/kt)
	elif E > 0:
		P = 1
	if (random.random() < P) == True:
		return('Accept')
	else:
		return('Reject')
#-------------------------------------------------------------------
#Start with initial configuration
TheFile = 'SimulatedAnealing.pdb'
pose = pose_from_pdb(TheFile)
'''
T = 10.0						#Initial temperature Value
SS1 , L1 , H1 , S1 = DSSP()				#Measure Original DSSP
print('Start\t{}\nL={}|H={}|S={}'.format(SS1 , L1 , H1 , S1))
print('----------')
pose_copy = Pose()
while T > 0:
	#Change phi/psi angles
	pose_copy.assign(pose)



	#Measure DSSP after change
	SS2 , L2 , H2 , S2 = DSSP()
	before = H1 + S1
	after = H2 + S2
	if Decision(before , after , T) == 'Accept':
		#print('\x1b[32m[+] Accept\x1b[0m')
		pose.assign(pose_copy)			#Accept by updating the original pose
		print('T={}\t{}\nL={}|H={}|S={}'.format(round(T , 3) , SS2 , L2 , H2 , S2))
		print('----------')
	else:
		#print('\x1b[31m[-] Reject\x1b[0m')
		continue
	#Lower temperature
	T -= 0.03
'''
#Movement Testing
pymover = PyMOLMover()
pymover.apply(pose)



