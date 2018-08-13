#!/usr/bin/python

'''
Required programs to install:
Debian:		sudo apt-get install autodock-vina openbabel
Arch:		sudo pacman -S openbabel && yaourt -S autodock-vina

Run using these commands:
* To run script and docking protocol:
python AutoDock.py PROTEIN.pdb LIGAND.sdf CX CY CZ LX LY LZ
'''

import os
import sys
import pymol
from pymol.cgo import *

#-------------------------------------------------------------------------------
def Box(pX, pY, pZ, x, y, z):
	pymol.cmd.pseudoatom('Position', pos=[pX, pY, pZ])
	([X, Y, Z], [a, b, c]) = pymol.cmd.get_extent('Position')
	pymol.cmd.show('spheres', 'Position')
	minX = X+float(x)
	minY = Y+float(y)
	minZ = Z+float(z)
	maxX = X-float(x)
	maxY = Y-float(y)
	maxZ = Z-float(z)
	CONFIG = open('setup.config', 'w')
	CONFIG.write('receptor =\t\tprotein.pdbqt\n')
	CONFIG.write('ligand =\t\tligand.pdbqt\n')
	CONFIG.write('out =\t\t\tdock.pdbqt\n')
	CONFIG.write('exhaustiveness =\t10\n')
	CONFIG.write('center_x =\t\t'+pX+'\n')
	CONFIG.write('center_y =\t\t'+pY+'\n')
	CONFIG.write('center_z =\t\t'+pZ+'\n')
	CONFIG.write('size_x =\t\t'+x+'\n')
	CONFIG.write('size_y =\t\t'+y+'\n')
	CONFIG.write('size_z =\t\t'+z+'\n')
	CONFIG.close()
	boundingBox = [BEGIN, LINES,
		VERTEX, minX, minY, minZ,
		VERTEX, minX, minY, maxZ,
		VERTEX, minX, maxY, minZ,
		VERTEX, minX, maxY, maxZ,
		VERTEX, maxX, minY, minZ,
		VERTEX, maxX, minY, maxZ,
		VERTEX, maxX, maxY, minZ,
		VERTEX, maxX, maxY, maxZ,
		VERTEX, minX, minY, minZ,
		VERTEX, maxX, minY, minZ,
		VERTEX, minX, maxY, minZ,
		VERTEX, maxX, maxY, minZ,
		VERTEX, minX, maxY, maxZ,
		VERTEX, maxX, maxY, maxZ,
		VERTEX, minX, minY, maxZ,
		VERTEX, maxX, minY, maxZ,
		VERTEX, minX, minY, minZ,
		VERTEX, minX, maxY, minZ,
		VERTEX, maxX, minY, minZ,
		VERTEX, maxX, maxY, minZ,
		VERTEX, minX, minY, maxZ,
		VERTEX, minX, maxY, maxZ,
		VERTEX, maxX, minY, maxZ,
		VERTEX, maxX, maxY, maxZ,
		END]
	boxName = 'Box'
	pymol.cmd.load_cgo(boundingBox, boxName)
	return(boxName)

def ligand(filename):
	'''
	Prepares the ligand by converting the .sdf or .pdb molecule file
	into a .pdbqt file
	'''
	os.system('babel {} ligand.pdbqt -p'.format(filename))

def receptor(filename, CX, CY, CZ, LX, LY, LZ):
	'''
	Prepares the protein by first removing all the water molecules from
	the protein's structure, then adds only the polar hydrogens, then
	it exports the resulting structure and converts it to a .pdbqt file. It
	also preps the search location box and generates the setup.config file
	that will be used to run the docking algorithm.
	'''
	cmd.load(filename)
	cmd.remove('resn HOH')
	cmd.h_add(selection='acceptors or donors')
	cmd.save('receptor.pdb')
	os.system('babel receptor.pdb temp.pdbqt -xh')
	os.system('grep ATOM temp.pdbqt > protein.pdbqt')
	os.remove('temp.pdbqt')
	CX = str(CX)
	CY = str(CY)
	CZ = str(CZ)
	LX = str(LX)
	LY = str(LY)
	LZ = str(LZ)
	cmd.load('receptor.pdb')
	pymol.cmd.extend('Box', Box(CX, CY, CZ, LX, LY, LZ))
	os.remove('receptor.pdb')

def main():
	ligand(sys.argv[2])
	CX = 11#sys.argv[3]
	CY = 90.5#sys.argv[4]
	CZ = 57.5#sys.argv[5]
	LX = 15#sys.argv[6]
	LY = 15#sys.argv[7]
	LZ = 15#sys.argv[8]
	receptor(sys.argv[1], CX, CY, CZ, LX, LY, LZ)
	os.system('vina --config setup.config > dock.log')
	os.system('pymol protein.pdbqt dock.pdbqt')

if __name__ == '__main__':
	main()
