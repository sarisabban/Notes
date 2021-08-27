**#Rosetta Ligand-Receptor Docking**
[https://www.rosettacommons.org/demos/latest/tutorials/ligand_docking/ligand_docking_tutorial]
--------------------------------------------------
**#Protocol**

#Prepare Receptor
1. Download receptor PDB crystal structure.
2. Clean structure and name receptor.pdb
3. Export receptor.pdb FASTA sequence and name receptor.fasta

#Prepare Ligand
4. Download from RCSB the IDEAL.SDF ligand file and rename ligand_original.sdf: RCSB.org > Advanced Search > Choose A Query Type > Chemical Compound > Chemical Name > Name of Ligand > Submit Query
5. Generate a confirmation library of the ligand:
	* Open the ligand_original.sdf with PyMol and save as .mol2 file
	* Go to: http://bioserv.rpbs.jussieu.fr/cgi-bin/Frog2
	* Input drug description > choose mol2
	* Upload the .mol2 file
	* Output format sdf
	* Run
6. Download the .zip file and rename the Frog.sdf library file to ligand.sdf
7. Generate the param file: {ROSETTA}/main/source/scripts/python/public/molfile_to_params.py -n ligand -p ligand --conformers-in-one-file ligand.sdf
8. This command will result in three files: ligand.param, ligand.pdb, and ligand_conformers.pdb

#Docking
9. Combine the ligand.pdb file with the receptor.pdb file: cat receptor.pdb ligand.pdb > receptor_ligand.pdb
10. Move the ligand to the aproximate binding location using PyMol and save the .pdb file again
11. Additional required files: dock.xml and flags
12. Run computation: {ROSETTA}/main/source/bin/rosetta_scripts.linuxgccrelease @flags

#Analysis
13. Choose the lowest scoring decoy
14. Use this decoy as your native structure and measure all other decoys' RMSD from it
15. Plot the Score vs RMSD values













# To Visualise The Docked Ligand
<USESES PYTHON 2>
#!/usr/bin/python

import pymol

def PyMol(filename):
	''' Takes a .pdb file that contains a protein with a docked ligand and sets up PyMOL visualisations to view the ligand '''
	''' Outputs a .pse file of the PyMOL visualisation settings '''
	''' Reference: From Rosetta Ligand Docking tutorial https://www.rosettacommons.org/demos/latest/tutorials/ligand_docking/ligand_docking_tutorial '''
	#Imports
	pymol.finish_launching(['pymol', '-qc'])
	#Load file
	pymol.cmd.delete('docked')
	pymol.cmd.load(filename, 'docked')
	#Make selections of ligand, binding pocket (all residues with an atom within 4 angstroms of ligand) and their polar contacts
	pymol.cmd.select(name='ligand', selection='docked and hetatm')
	pymol.cmd.select(name='pocket', selection='ligand around 4 and docked')
	pymol.cmd.select(name='pocket', selection='byres pocket')
	pymol.cmd.distance('contacts', 'ligand', 'pocket', 4, 2)
	pymol.cmd.color(4,'contacts')	
	#Stylistic changes
	pymol.preset.ligand_cartoon('all')
	pymol.cmd.delete('docked_pol_conts')
	pymol.cmd.show('cartoon','all')
	pymol.cmd.hide('lines')
	pymol.cmd.bg_color('black')
	pymol.cmd.set('cartoon_fancy_helices', 1)
	pymol.cmd.show('sticks','pocket & (!(n;c,o,h|(n. n&!r. pro)))')
	pymol.cmd.show('sticks','ligand')
	pymol.cmd.hide('lines', 'hydro')
	pymol.cmd.hide('sticks', 'hydro')
	#Save session
	filename=os.path.splitext(filename)[0]
	outfile=filename+'.pse'
	pymol.cmd.save(outfile)

filename = '2.pdb'
PyMol(filename)





# To Calculate The RMSD
#!/usr/bin/python

import pymol , glob
pymol.finish_launching(['pymol' , '-qc'])

def Load():
	''' Load the .pdb files into PyMol '''
	pymol.cmd.load('receptor_ligand.pdb' , 'Native')
	count = 0
	for filename in glob.glob('receptor_ligand_*.pdb'):
		count += 1
		pymol.cmd.load(filename , 'Docked' + str(count))
	return(count)

def RMSD(native , docked):
	''' Calculate the RMSDs '''
	Total_RMSD = pymol.cmd.rms_cur(native , docked)
	print(Total_RMSD)
#----------------------------------------------------------------
number = Load()
count = 0
for loop in range(number):
	count += 1
	RMSD('Native' , 'Docked'+str(count))
