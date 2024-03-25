#!/usr/bin/python3

#Imports
import sys , subprocess , time
from pyrosetta import *
from pyrosetta.toolbox import *
init()

#Inputs
RECEPTOR	= '1V74'
CHAIN		= 'A'
LIGAND		= 'ETQ'

jump_num 	= 1
rotation 	= 8
translation 	= 3
scorefxn 	= get_fa_scorefxn()
#--------------------------------------------------------------------
#Fundtions

#1 - Get Receptor and Ligand Inputs
class GET():
	#1.1 Get Receptor
	def receptor(receptor , chain):
		''' Downloads the receptor's .pdb file and cleans it in preparation for the docking protocol '''
		''' Generates the receptor.pdb '''
		os.system('wget http://www.rcsb.org/pdb/files/' + receptor + '.pdb')
		TheReceptor = open(receptor + '.pdb' , 'r')
		TheFile = open('receptor.pdb' , 'w')
		for line in TheReceptor:
			search = line.split()
			if search[0] == 'ATOM':			#Choose only protein atoms
				if search[4] == chain:		#Choose chain
					TheFile.write(line)	#Save to file
		os.remove(receptor + '.pdb')			#Delete RCSB file
	#1.2 Get Ligand
	def ligand(ligand):
		''' Downloads a ligand's .sdf file and prepares for the docking protocol '''
		''' Generates the ligand.pdb '''
		os.system('wget files.rcsb.org/ligands/view/' + ligand + '_ideal.sdf')

#2 Global Search
def Global(pose , repeats):
	score = list()
	for nterm in range(repeats):
		pyrosetta.rosetta.protocols.docking.setup_foldtree(pose , 'A_B', Vector1([1]))		#Make chain A ridgid and chain B move
		pert_mover = pyrosetta.rosetta.protocols.rigid.RigidBodyPerturbMover(jump_num , rotation , translation)
		pert_mover.apply(pose)

		randomize_upstream = pyrosetta.rosetta.protocols.rigid.RigidBodyRandomizeMover(pose , jump_num , pyrosetta.rosetta.protocols.rigid.partner_upstream)
		randomize_upstream.apply(pose)

		randomize_downstream = pyrosetta.rosetta.protocols.rigid.RigidBodyRandomizeMover(pose , jump_num , pyrosetta.rosetta.protocols.rigid.partner_downstream)
		randomize_downstream.apply(pose)

		slide = pyrosetta.rosetta.protocols.docking.FaDockingSlideIntoContact(jump_num)
		slide.apply(pose)

		movemap = MoveMap()
		movemap.set_jump(jump_num , True)
		minmover = pyrosetta.rosetta.protocols.simple_moves.MinMover()
		minmover.movemap(movemap)
		minmover.apply(pose)

		score.append(scorefxn(pose))

	score.sort()
	for value in score:
		print(value)
#--------------------------------------------------------------------
#Protocol

#GET.receptor(RECEPTOR , CHAIN)
#GET.ligand(LIGAND)
pose = pose_from_rcsb(RECEPTOR)
os.remove(RECEPTOR+'.pdb')
os.remove(RECEPTOR+'.clean.pdb')

Global(pose , 3)
