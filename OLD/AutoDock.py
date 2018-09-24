#!/usr/bin/python

"""
INSTRUCTIONS:
* Install programs: sudo pacman -S openbabel pymol
* Download receptor
* Choose receptor search box: pymol AutoDock.py FILENAME.pdb
* Export receptor as .pdbqt file: python AutoDock.py receptor FILENAME.pdb
* Download ligand database from ZINC15 website [zinc15.dock.com]
* Segment ligands into individual files: python AutoDock.py segment DIRECTORY
* Run docking:
	#!/bin/bash
	#PBS -N dock
	#PBS -q thin_1m
	#PBS -l select=1:ncpus=24:mpiprocs=24
	#PBS -j oe
	#PBS -J 1-82
	cd $PBS_O_WORKDIR
	for i in `ls -lv ../Ligands/${PBS_ARRAY_INDEX}`; do
		echo $i
		./vina --receptor receptor.pdbqt --ligand ../Ligands/${PBS_ARRAY_INDEX}/$i --out docked_$i --exhaustiveness 10 --seed 1991540417 --center_x 8 --center_y -19 --center_z -7 --size_x 18 --size_y 9 --size_z 17
		echo ''
		echo ''
	done
	echo '--------------------------------------------------'
	echo 'DONE'
* Analyse output: python AutoDock.py analyse DIRECTORY
"""

import os
import sys
import time
import pymol
import pathlib
import itertools
from pymol.cgo import *

def Box(pX, pY, pZ, x, y, z):
	'''
	Sets up the search box within the protein, which is
	used in the docking protocol
	'''
	pymol.cmd.pseudoatom('Position', pos=[pX, pY, pZ])
	([X, Y, Z],[a, b, c]) = pymol.cmd.get_extent('Position')
	pymol.cmd.show('spheres', 'Position')
	minX = X+float(x)
	minY = Y+float(y)
	minZ = Z+float(z)
	maxX = X-float(x)
	maxY = Y-float(y)
	maxZ = Z-float(z)
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

def protein(filename):
	'''
	Prepares the protein by first removing all the water
	molecules from the protein's structure, then adds only
	the polar hydrogens, then it exports the resulting
	structure and converts it to a .pdbqt file
	'''
	cmd.load(filename)
	cmd.remove('resn HOH')
	cmd.h_add(selection='acceptors or donors')
	cmd.save('receptor.pdb')
	os.system('babel receptor.pdb temp.pdbqt -xh')
	os.system('grep ATOM temp.pdbqt > receptor.pdbqt')
	os.remove('temp.pdbqt')
	os.remove('receptor.pdb')

def ligand(filename):
	'''
	Prepares a single ligand by converting the .sdf file or
	.pdb file into a .pdbqt file
	'''
	os.system('babel {} ligand.pdbqt -p'.format(filename))

def analyse(filename):
	'''
	Extracts the lowest binding affinity from the .pdbqt
	file that results from a docking run
	'''
	dockfile = open(filename, 'r')
	check = False
	for line in dockfile:
		if line.split(':')[0] == 'REMARK VINA RESULT'\
				and check is False:
			check = True
			d = line.split()
			f = filename.split('_')[0]
			newline = '{:19} {:10} {:9} {:10}'\
				.format(f, d[3], d[4], d[5])
			return(newline)

def analyse_run(directory):
	'''
	This function analyses multiple docking output collected
	in a directory then organises all the binding affinities
	into a file and sorts them from lowest values to largest
	'''
	thefiles = sorted(os.listdir(directory))
	for f in thefiles:
		print('Analysed {}'.format(f))
		line = analyse('{}/{}'.format(directory, f))
		temp = open('temp', 'a')
		temp.write(line)
		temp.write('\n')
		temp.close()
	results = open('Results', 'a')
	results.write('Molecule           |Affinity  |Dist from|Best mode\n')
	results.write('                   |(kcal/mol)|RMSD l.b.|RMSD u.b.\n')
	results.write('-------------------+----------+---------+---------\n')
	results.close()
	os.system('cat temp | sort -nk 2 >> Results')
	os.system('rm temp')

def prep(directory):
	'''
	This function collects all the downloaded structures from
	the ZINC15 database and unzipps them in preparation for
	parsing and segmentation by split_multi()
	'''
	os.mkdir('ligs')
	os.system("find %s -name '*.gz' -exec mv {} ligs \;" % directory)
	dr = os.listdir('ligs')
	for f in dr:
		print(f)
		os.system('gunzip ligs/{}'.format(f))

def lines_from_files(names):
	''' Function for split_multi() '''
	for name in names:
		with open(str(name)) as f:
			yield from f

def split_multi(lines, dire, prefix):
	'''
	This function loops through a directory with multiple
	.pdbqt files each with multiple molecules and separates
	each file's molecules into a single directory.
	'''
	count = 0
	in_dir_count = 0
	limit = 100
	dircount = 0
	for dircount in itertools.count():
		for line in lines:
			x = line.split()
			if len(x) == 2 and x[0] == 'MODEL' and x[1].isdigit():
				directory = os.path.join(dire, '{}'.format(dircount+1))
				os.makedirs(directory, exist_ok=True)
				out = os.path.join(directory, '{}_{}.pdbqt'.format(prefix, count+1))
				with open(out, 'w') as outfile:
					for line in lines:
						if line.strip() == 'ENDMDL':
							break
						outfile.write(line)
				count += 1
				in_dir_count += 1
				if in_dir_count >= limit:
					in_dir_count = 0
					print('Finished directory {}'.format(directory))
					break
		else:
			break
	print('----------\nDone')

def main():
	if sys.argv[1] == 'receptor':
		protein(str(sys.argv[2]))
	elif sys.argv[1] == 'ligand':
		ligand(str(sys.argv[2]))
	elif sys.argv[1] == 'analyse':
		analyse_run(sys.argv[2])
	elif sys.argv[1] == 'segment':
		prep(sys.argv[2])
		time.sleep(1)
		lines = lines_from_files(pathlib.Path.cwd().glob('ligs/*.pdbqt'))
		split_multi(lines, 'Ligands', 'ligand')

if __name__ != "__main__":
	pymol.cmd.load(str(sys.argv[2]))
	pymol.cmd.extend('Box', Box)

if __name__ == "__main__": main()
