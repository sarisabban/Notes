#!/usr/bin/python

import os
import sys

def Melip(filename, C, M, L):
	'''
	Takes a Circular Dichroism text file in milli degrees and converts it
	into molar ellipticity, then plots the values, the following is the
	equation used: m°*M/(10*L*C)
	C is concentration in g/L
	M is average molecular weight (g/mol)
	L is path length of cell (cm)
	Then plots the results using GnuPlot, and exports the plot as a PDF
	'''
	name = filename.split('.')[0]
	C = float(C)
	M = float(M)
	L = float(L)
	data = open(filename, 'r')
	for line in data:
		line = line.strip().split()
		try:
			float(line[0])
			X = float(line[0])
			mDeg = float(line[1])
			Y = (mDeg*M)/(10*L*C)
			newline = '{}\t{}\n'.format(X, Y)
			temp = open('temp', 'a')
			temp.write(newline)
			temp.close()
		except:
			pass
	gnu = open('gnu', 'w')
	gnu.write('''
	set terminal postscript
	set output './{}_plot.pdf'
	set term post eps enh color
	set xlabel'Wavelength (nm)'
	set ylabel'Molar Ellipticity'
	set xrange [190:250]
	set yrange [:]
	set title 'Circular Dichroism'
	plot './temp' lc rgb 'red' smooth csplines title '{}'
	exit'''.format(name, name))
	gnu.close()
	os.system('gnuplot < gnu')
	os.remove('gnu')
	os.remove('temp')

def main():
	filename = sys.argv[1]
	conc = sys.argv[2]
	mw = sys.argv[3]
	length = sys.argv[4]
	Melip(filename, conc, mw, length)

if __name__ == '__main__':
	main()
