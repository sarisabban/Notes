#!/usr/bin/python

import pymol
from pymol.cgo import *

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
	CONFIG = open('config.txt', 'w')
	CONFIG.write('receptor =\t\treceptor.pdbqt\n')
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

pymol.cmd.extend('Box', Box)
