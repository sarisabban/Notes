#!/usr/bin/python

'''
Instructions:
-------------

This script uses Python 3.6+ and requires openbabel and PyMOL 2.2.
Follow this precise order of steps.

0. Use the command python3 AutoDock.py -h for the help menu.
1. Update system and install required programs:
	sudo apt update && sudo apt full-upgrade && sudo apt install openbabel pymol
2. Choose the search space:
	pymol AutoDock.py -b FILENAME.pdb
	select an amino acid (has to be an amino acid only) where you want the center of the search box to be
	then in PyMOL command terminal type Box("sele",1,1,1) where 1, 1, 1 are the x y z distances
	from the center of the box. Adjust the x y z distances until the box is of satisfactory
	size, you have to delete the Box and Position objects everytime before adjusting the numbers.
	When finished the last line that is printed in the pymol terminal (Ca center of selection is)
	will be the Center_X Center_Y Center_Z values. The Size_X Size_Y Size_Z values are the 
	values that you have adjusted. Use Box_fine(pX, pY, pZ, x, y, z) to fine adjust the center of the box,
	where pX, pY, pZ are the values from Ca center of selection (the center of the box).
3. Prepare and convert the protein receptor from PDB to PDBQT:
	python3 AutoDock.py -r FILENAME.pdb
4. Get Ligands from ZINC15 database.
5. Download the ligands and combine them into a file:
	python3 AutoDock.py -d FILENAME.wget
6. Split ligands file for virtual screaning:
	python3 AutoDock.py -s FILENAME.pdbqt 300000
7. Generate a PBS job submission file for a high performance computer:
	python3 AutoDock.py -j Center_X Center_Y Center_Z Size_X Size_Y Size_Z Seed Exhaustiveness Output CPUs Array Email
8. Download Autodock vina from the following link:
	http://vina.scripps.edu/download.html
9. Combine computed files and sort them to see which ligand binds strongest:
	python3 AutoDock.py -c DIRECTORY
10. Generate PBS job script:
	python3 AutoDock.py -p pX pY pZ x y z seed exhaust out CPU array email
		pX pY pZ x y z -> these values are the box dementions
		seed           -> choose a random number for a starting seed
		exhaust        -> how exhaustive you want the search
		out            -> true or false (true will keep the output for each search - I recommend false)
		CPU            -> number of CPUs you have available per node
		array          -> number of array to repeat the code
		email          -> your email to be notified when the job completes

Here is a video explaning how to perform virtual screaning using AutoDock Vina
and how to use this script:

For running on a local computer use this command:
for file in ./Ligands/*/*; do tmp=${file%.pdbqt}; name="${tmp##*/}"; ./vina --receptor receptor.pdbqt --ligand "$file" --out $name_out.pdbqt --log $name.log --exhaustiveness 10 --center_x 0 --center_y 0 --center_z 10 --size_x 15 --size_y 15 --size_z 15; awk '/^[-+]+$/{getline;print FILENAME,$0}' $name.log >> temp; done; sort temp -nk 3 > Results; rm temp; mkdir logs; mv *.log *.out logs
--------------------------------------------------------------------------------
MIT License

Copyright (c) 2018 Sari Sabban

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import sys
import math
import numpy
import pymol
import argparse
import itertools
from pymol.cgo import *

def center_of_box(selection):
	'''
	output the avarage Ca coordinate of the given selection
	'''
	selection = selection + " and n. CA"
	model = querying.get_model(selection)
	coords = model.get_coord_list()
	coords_matrix = numpy.array(coords)
	coords_ave = coords_matrix.mean(axis=0)
	coords_ave = coords_ave.tolist()
	print("Ca coordinates of selection are:", coords)
	print("Ca center of selection is:", coords_ave)
	cmd.extend("center_of_box", center_of_box)
	return coords_ave

def Box(selection, x, y, z):
	'''
	Sets up the search box within the protein, which is
	used in the docking protocol
	'''
	pX, pY, pZ = center_of_box(selection)
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

def Box_fine(pX, pY, pZ, x, y, z):
	'''
	Fine selection of the search box within the protein, which is
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

def download(filename):
	'''
	Download, unzip, combine, renumber ligands
	'''
	with open(filename, 'r')as infile:
		for line in infile:
			try:
				namegz = line.split()[-1]
				name = line.split()[-1].split('gz')[0][:-1]
				get = line.split()[1]
				#wget = 'wget {} -O {}'.format(get, namegz)
				wget = line.strip()
				gunzip = 'gunzip {}'.format(namegz)
				cat = 'cat {} >> temp'.format(name)
				os.system(wget)
				os.system(gunzip)
				with open(name) as f:
					first = f.readline()
				if first.split()[0] == 'MODEL':
					os.system(cat)
				else:
					os.system('echo "MODEL        1" >> temp')
					os.system(cat)
					os.system('echo "ENDMDL" >> temp')
			except:
				with open('error', 'a') as e:
					e.write(line)
	count = 0
	with open('temp', 'r') as infile:
		with open('temp2', 'a') as outfile:
			for line in infile:
				if line.startswith('MODEL'):
					count += 1
					outfile.write('MODEL {:15}\n'.format(count))
				else:
					outfile.write(line)
	#os.system('ls *.pdbqt | grep -v receptor.pdbqt | xargs rm')
	os.system("rm -R `ls -1 -d */`")
	os.remove('temp')
	os.rename('temp2', 'ZINC15.pdbqt')

def receptor(filename):
	'''
	Prepares the receptor by first removing all the water molecules from
	the protein's structure, then adds only the polar hydrogens, then
	it exports the resulting structure and converts it to a .pdbqt file.
	'''
	cmd.load(filename)
	cmd.remove('resn HOH')
	cmd.h_add(selection='acceptors or donors')
	cmd.save('protein.pdb')
	#os.system('babel protein.pdb temp.pdbqt -xh')	#OLD
	os.system('obabel protein.pdb -O temp.pdbqt -xh')
	os.system('grep ATOM temp.pdbqt > receptor.pdbqt')
	os.remove('temp.pdbqt')
	os.remove('protein.pdb')

def ligand(filename):
	name = filename[:-4]
	cmd.load(filename)
	cmd.remove('resn HOH')
	cmd.h_add()
	cmd.save('{}.pdb'.format(name))
	os.system('obabel {}.pdb -O {}.pdbqt -xh'.format(name, name))
	os.remove('{}.pdb'.format(name))

def split(filename, direct, prefix, limit):
	'''
	Separates a .pdbqt file with multiple molecules into separate files with
	singles molecules segmented over sub directories.
	'''
	with open(filename) as infile:
		count = 0
		in_dir_count = 0
		dircount = 0
		for dircount in itertools.count():
			for line in infile:
				#if line.strip() == 'MODEL{:16}'.format(count+1):
				if line.strip().split()[0] == 'MODEL' and line.strip().split()[1] == '{}'.format(count+1):
					directory = os.path.join(direct, '{}'.format(dircount+1))
					os.makedirs(directory, exist_ok=True)
					name = '{}_{:09}.pdbqt'.format(prefix, count+1)
					out = os.path.join(directory, name)
					with open(out, 'w') as outfile:
						for line in infile:
							if line.strip() == 'ENDMDL':
								break
							if line.split()[0] == 'REMARK' and\
											line.split()[1] == 'Name':
								NewName = os.path.join(directory,\
											'{}.pdbqt'.format(line.split()[3]))
							outfile.write(line)
					os.rename(out, NewName)
					count += 1
					in_dir_count += 1
					if in_dir_count >= limit:
						in_dir_count = 0
						print('[+] Finished directory {}'.format(directory))
						break
			else: break
	print('----------\n[+] Done')

def PBS(pX, pY, pZ, x, y, z, seed, exhaust, out, CPU, array, email):
	'''
	Write a PBS file for HPC virtual screening
	'''
	if out == 'True' or out == 'true':
		output = '"out_$n"'
	elif out == 'False' or out == 'false':
		output = '/dev/null'
	with open('dock.pbs', 'w') as dock:
		dock.write('#!/bin/bash\n\n')
		dock.write('#PBS -N Docking\n')
		dock.write('#PBS -m e\n')
		dock.write('#PBS -M {}\n'.format(email))
		dock.write('#PBS -q thin_1m\n')
		dock.write('#PBS -l select=1:ncpus=24:ompthreads=24\n')
		dock.write('#PBS -j oe\n')
		dock.write('#PBS -J 1-{}\n\n'.format(array))
		dock.write('cd $PBS_O_WORKDIR\n\n')
		dock.write('mkdir -p ../Ligands_Completed/${PBS_ARRAY_INDEX}\n')
		dock.write('process() { local n=${1##*/}\n')
		dock.write('\t./vina \\\n')
		dock.write('\t\t--receptor receptor.pdbqt \\\n')
		dock.write('\t\t--ligand "$1" \\\n')
		dock.write('\t\t--out {} \\\n'.format(output))
		dock.write('\t\t--log "log_$n" \\\n')
		dock.write('\t\t--exhaustiveness {} \\\n'.format(exhaust))
		dock.write('\t\t--cpu {} \\\n'.format(CPU))
		dock.write('\t\t--seed {} \\\n'.format(seed))
		dock.write('\t\t--center_x {} \\\n'.format(pX))
		dock.write('\t\t--center_y {} \\\n'.format(pY))
		dock.write('\t\t--center_z {} \\\n'.format(pZ))
		dock.write('\t\t--size_x {} \\\n'.format(x))
		dock.write('\t\t--size_y {} \\\n'.format(y))
		dock.write('\t\t--size_z {} \\\n'.format(z))
		dock.write('''\t\t| awk -v name="$n" '$1 == "1" {print name "\\t" $0;exit}' >> Docks_${PBS_ARRAY_INDEX} \\\n''')
		dock.write('\t\t&& rm log_$n \\\n')
		dock.write('\t\t&& mv "$1" ../Ligands_Completed/${PBS_ARRAY_INDEX}\n')
		dock.write('}\n')
		dock.write('export -f process\n')
		dock.write('''find ../Ligands/${PBS_ARRAY_INDEX}/ -type f -print0 | xargs -0 -P 24 -I{} bash -c 'process "$1"' _ {}''')

def Kd_to_dG(Kd):
	Kd = float(Kd)
	dG = 0.0019872036*298*numpy.log(Kd)
	print('{} Kcal/mol'.format(round(dG, 2)))

def dG_to_Kd(dG):
	dG = float(dG)
	Kd = math.e**(dG/(0.0019872036*298))
	print('{:0.2e} dG'.format(Kd))

parser = argparse.ArgumentParser(description='Prep ligands for AutoDock Vina')
parser.add_argument('-r',
					'--receptor',
					nargs='+',
					help='Prep and convert protein receptor from PDB to PDBQT')
parser.add_argument('-b',
					'--box',
					nargs='+',
					help='Draw search box')
parser.add_argument('-d',
					'--download',
					nargs='+',
					help='Download, unzip, renumber, combine ligands')
parser.add_argument('-s',
					'--split',
					nargs='+',
					help='Split a file with multiple models into single files\
							segmented into directories')
parser.add_argument('-j',
					'--job',
					nargs='+',
					help='Write the PBS file for HPC virtual screaning')
parser.add_argument('-c',
					'--combine',
					nargs='+',
					help='Sort and combine the docking results into a file')
parser.add_argument('-Kd',
					'--Kd_to_dG',
					nargs='+',
					help='Convert Kd to delta G')
parser.add_argument('-dG',
					'--dG_to_Kd',
					nargs='+',
					help='Convert delta G to Kd')
parser.add_argument('-p',
					'--pbs',
					nargs='+',
					help='Generates an HPC PBS job script')
args = parser.parse_args()

def main():
	if args.receptor:
		receptor(sys.argv[2])
	elif args.box:
		pymol.cmd.load(str(sys.argv[2]))
		pymol.cmd.extend('Box', Box)
	elif args.download:
		download(sys.argv[2])
	elif args.split:
		split(sys.argv[2], 'Ligands', 'model', int(sys.argv[3]))
	elif args.job:
		PBS(sys.argv[2],	# pX
			sys.argv[3],	# pY
			sys.argv[4],	# pZ
			sys.argv[5],	# x
			sys.argv[6],	# y
			sys.argv[7],	# z
			sys.argv[8],	# Seed
			sys.argv[9],	# Exhaustiveness
			sys.argv[10],	# Output
			sys.argv[11],	# CPUs			
			sys.argv[12],	# Array
			sys.argv[13])	# Email
	elif args.combine:
		os.system('cat {}/Docks_* | sort -nk 3 > Result'.format(sys.argv[2]))
	elif args.Kd_to_dG:
		Kd_to_dG(sys.argv[2])
	elif args.dG_to_Kd:
		dG_to_Kd(sys.argv[2])
	elif args.pbs:
		PBS(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13])

if __name__ == '__main__': main()
