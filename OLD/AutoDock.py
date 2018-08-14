#!/usr/bin/python

'''
# Required programs to install:
Debian:
sudo apt-get install autodock-vina openbabel

Arch:
sudo pacman -S openbabel && yaourt -S autodock-vina



# Run using these commands:
* To download the ligand database from ChemDB (http://cdb.ics.uci.edu/):
python AutoDock.py download

* To setup multiple ligands:
python AutoDock.py ligand miltiple LIGAND.sdf

* To setup a single ligand:
python AutoDock.py ligand single LIGAND.sdf

* To setup the protein:
python AutoDock.py protein PROTEIN.pdb CX CY CZ LX LY LZ

* To run the docking protocol on multiple ligands:
python AutoDock.py dock multiple LIGAND_DIRECTORY

* To run the docking protocol on a single ligand:
python AutoDock.py dock single
'''

import os
import sys
import pymol
from pymol.cgo import *

#-------------------------------------------------------------------------------
def Box(pX, pY, pZ, x, y, z):
	'''
	Sets up the search box within the protein, the dimentions are then
	added to a file named setup.config which is used in the docking
	protocol
	'''
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

def download(items):
	'''
	Downloads around 4,000,000+ small molecules form the ChemDB databse
	(http://cdb.ics.uci.edu/), then combines all the downloaded files into
	one large .sdf file
	'''
	count = items
	while count != 0:
		numb = '{0:02d}'.format(count)
		URL = 'ftp://ftp.ics.uci.edu/pub/baldig/cdbDownload'
		os.system('wget {}/isomer3d.{}00000.sdf.gz'.format(URL, numb))
		os.system('gzip -d isomer3d.{}00000.sdf.gz'.format(numb))
		count -= 1
	os.system('cat isomer3d.* > Molecules.sdf')
	os.system('rm isomer3d.*')

def ligand(filename, protocol):
	'''
	Prepares the ligand by converting the .sdf or .pdb molecule file
	into a .pdbqt file
	'''
	if protocol == 'single':
		os.system('babel {} ligand.pdbqt -p'.format(filename))
	elif protocol == 'multiple':
		os.mkdir('ligands')
		os.system('babel {} ligands.pdbqt -p'.format(filename))
		command = 'vina_split --input ligands.pdbqt --ligand'
		os.system('{} ligands/ligand_'.format(command))
		os.remove('ligands.pdbqt')
	else:
		print('Error: bad command argument')

def protein(filename, CX, CY, CZ, LX, LY, LZ):
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
	if sys.argv[1] == 'download':
		download(72)

	elif sys.argv[1] == 'ligand' and sys.argv[2] == 'single':
		ligand(sys.argv[3], 'single')

	elif sys.argv[1] == 'ligand':
		ligand(sys.argv[3], 'multiple')

	elif sys.argv[1] == 'protein':
		CX = sys.argv[3]
		CY = sys.argv[4]
		CZ = sys.argv[5]
		LX = sys.argv[6]
		LY = sys.argv[7]
		LZ = sys.argv[8]
		protein(sys.argv[2], CX, CY, CZ, LX, LY, LZ)

	elif sys.argv[1] == 'dock' and sys.argv[2] == 'single':
		os.system('vina --config setup.config 2>&1 | tee dock.log')
		os.system('pymol protein.pdbqt dock.pdbqt')

	elif sys.argv[1] == 'dock' and sys.argv[2] == 'multiple':
		directory = sys.argv[3]
		
		os.system('pymol protein.pdbqt dock.pdbqt')

	else:
		print('Error: bad command argument')

if __name__ == '__main__':
	main()
