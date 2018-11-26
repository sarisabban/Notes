#!/usr/bin/python

import os
import sys
import numpy as np
import biotite
import biotite.structure as struc
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb
import biotite.structure.io as strucio

def Nano(angstrom):
	'''Convert angstrom to nanometer'''
	nano = angstrom / 10
	return(nano)

def DiameterA(TheFile):
	'''
	Find the diameter of a protein's structure accuratly, requires lots
	of memory and crashes for big structures. Here we broadcast the array
	against itself, calculating all pairwise distances between points.
	This is a bad idea, because we have N*(N-1) = (1e6)**2 = 1 trillion
	pairs! This will raise a MemoryError for N=1 million, as it requires
	half a million gigabytes!!
	'''
	# Get atom coordinates
	atom_array = strucio.load_structure(TheFile)
	# Remove all non-amino acids atoms
	atom_array = atom_array[struc.filter_amino_acids(atom_array)]
	# Coordinates as a NumPy array
	coord = atom_array.coord
	# Calculate all pairwise difference vectors
	diff = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
	# Calculate absolute of difference vectors -> square distances
	sq_dist = np.sum(diff*diff, axis=-1)
	# Get maximum distance
	maxdist = np.max(sq_dist)
	# Maximum distance is diameter
	diameter = np.sqrt(np.max(sq_dist))
	return(round(diameter, 3))

def Diameter(TheFile):
	'''
	Find the diameter of a protein's structure approximately, requires less
	memory thus good for big structures
	'''
	# Get atom coordinates
	atom_array = strucio.load_structure(TheFile)
	# Remove all non-amino acids atoms
	atom_array = atom_array[struc.filter_amino_acids(atom_array)]
	# Coordinates as a NumPy array
	coord = atom_array.coord
	# Find geometric center
	center = np.mean(coord, axis=0)
	# Find largest distance from center -> diameter
	diameter = 2*np.sqrt(np.sum((coord - center)**2, axis=-1)).max()
	return(round(diameter, 3))

def main():
	directory = sys.argv[1]
	filelist = os.listdir(directory)
	for File in filelist:
		try:
			diameter = DiameterA('{}/{}'.format(directory, File))
			diameternano = round(Nano(diameter), 3)
			print('{} = {} A\t{} nm'.format(File, diameter, diameternano))
		except:
			diameter = Diameter('{}/{}'.format(directory, File))
			diameternano = round(Nano(diameter), 3)
			print('{} = {} A\t{} nm'.format(File, diameter, diameternano))

if __name__ == '__main__': main()
