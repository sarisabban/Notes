#!/usr/bin/python


import os
import sys
#import pymol
import argparse
import itertools
#from pymol.cgo import *

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

def download(filename):
	'''
	Download, unzip, combine, renumber ligands
	'''
	with open(filename, 'r')as infile:
		for line in infile:
			namegz = line.split()[-1]
			name = line.split()[-1].split('gz')[0][:-1]
			get = line.split()[1]
			wget = 'wget {} -O {}'.format(get, namegz)
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
	count = 0
	with open('temp', 'r') as infile:
		with open('temp2', 'a') as outfile:
			for line in infile:
				if line.startswith('MODEL'):
					count += 1
					outfile.write('MODEL {:15}\n'.format(count))
				else:
					outfile.write(line)
	os.system('rm *.pdbqt')
	os.remove('temp')
	os.rename('temp2', 'ZINC15.pdbqt')

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
				if line.strip() == 'MODEL{:16}'.format(count+1):
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

def PBS(pX, pY, pZ, x, y, z, seed, exhaust, array, email):
	'''
	Write a PBS file for HPC virtual screening
	'''
	with open('dock.pbs', 'w') as TheFile:
		TheFile.write('#!/bin/bash\n\n')
		TheFile.write('#PBS -N Docking\n')
		TheFile.write('#PBS -m e\n')
		TheFile.write('#PBS -M {}\n'.format(email))
		TheFile.write('#PBS -q thin_1m\n')
		TheFile.write('#PBS -l select=1:ncpus=24:ompthreads=24\n')
		TheFile.write('#PBS -j oe\n')
		TheFile.write('#PBS -J 1-{}\n\n'.format(array))
		TheFile.write('cd $PBS_O_WORKDIR\n\n')
		TheFile.write('mkdir -p ../Ligands_Completed/${PBS_ARRAY_INDEX}\n')
		TheFile.write('process() { local n=${1##*/}\n')
		TheFile.write('\t./vina \\\n')
		TheFile.write('\t\t--receptor receptor.pdbqt \\\n')
		TheFile.write('\t\t--ligand "$1" \\\n')
		TheFile.write('\t\t--out /dev/null \\\n')
		TheFile.write('\t\t--log "log_$n" \\\n')
		TheFile.write('\t\t--exhaustiveness {} \\\n'.format(exhaust))
		TheFile.write('\t\t--cpu 1 \\\n')
		TheFile.write('\t\t--seed {} \\\n'.format(seed))
		TheFile.write('\t\t--center_x {} \\\n'.format(pX))
		TheFile.write('\t\t--center_y {} \\\n'.format(pY))
		TheFile.write('\t\t--center_z {} \\\n'.format(pZ))
		TheFile.write('\t\t--size_x {} \\\n'.format(x))
		TheFile.write('\t\t--size_y {} \\\n'.format(y))
		TheFile.write('\t\t--size_z {} \\\n'.format(z))
		TheFile.write('''\t\t| awk -v name="$n" '$1 == "1" {print name "\\t" $0;exit}' >> Docks_${PBS_ARRAY_INDEX} \\\n''')
		TheFile.write('\t\t&& rm log_$n \\\n')
		TheFile.write('\t\t&& mv "$1" ../Ligands_Completed/${PBS_ARRAY_INDEX}\n')
		TheFile.write('}\n')
		TheFile.write('export -f process\n')
		TheFile.write('''find ../Ligands/${PBS_ARRAY_INDEX}/ -type f -print0 | xargs -0 -P 24 -I{} bash -c 'process "$1"' _ {}''')

parser = argparse.ArgumentParser(description='Prep ligands for AutoDock Vina')
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
parser.add_argument('-p',
					'--pbs',
					nargs='+',
					help='Write the PBS file for HPC virtual screaning')
parser.add_argument('-c',
					'--combine',
					nargs='+',
					help='Sort and combine the docking results into a file')
args = parser.parse_args()

def main():
	if args.box:
		pymol.cmd.load(str(sys.argv[2]))
		pymol.cmd.extend('Box', Box)
	elif args.download:
		download(sys.argv[2])
	elif args.split:
		split(sys.argv[3], 'Ligands', 'model', int(sys.argv[2]))
	elif args.pbs:
		PBS(sys.argv[2],	# pX
			sys.argv[3],	# pY
			sys.argv[4],	# pZ
			sys.argv[5],	# x
			sys.argv[6],	# y
			sys.argv[7],	# z
			sys.argv[8],	# Seed
			sys.argv[9],	# Exhaustiveness
			sys.argv[10],	# Array
			sys.argv[11])	# Email
	elif args.combine:
		os.system('{}/cat Docks_* | sort -nk 3 > temp'.format(sys.argv[2]))

if __name__ == "__main__": main()
