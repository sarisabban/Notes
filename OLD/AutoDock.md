**AutoDock Vina**
--------------------------------------------------------------------
# Install
sudo apt-get install autodock-vina openbabel
sudo pacman -S openbabel && yaourt -S autodock-vina
# Protocol
## Ligand
1. Get the ligand from RCSB: 		`wget files.rcsb.org/ligands/view/FILENAME_ideal.sdf`
2. Convert sdf to pdbqt: 		`babel FILENAME.sdf ligand.pdbqt -p`
## Receptor
3. Get the protein receptor RCSB: 	`wget files.rcsb.org/download/FILENAME.pdb` Make sure only 1 chain is present without any ligands
4. Open PyMol
5. Remove all water:			`receptor > action > remove waters`
6. Add Hydrogen:			`receptor > action > hydrogens > add polar`
7. Export molecule receptor.pdb
8. Convert pdb to pdbqt:		`babel receptor.pdb receptortemp.pdbqt -xh`
9. Get only the ATOMS:			`grep ATOM receptortemp.pdbqt > receptor.pdbqt`
## Search Space and Generate the config.text file
10. From the terminal:			`pymol receptor.pdb`
11. From PyMol:				`run Box.py`
12. From PyMol:				`Box Center_X, Center_Y, Center_Z, Length_X, Length_Y, Length_Z`
## Required Computation Files:
* receptor.pdbqt
* ligand.pdbqt
* conf.txt
## The Run Computation:
13. vina --config config.txt
or without the config.txt file
vina --receptor receptor.pdbqt --ligand ligand.pdbqt --out dock.pdbqt --exhaustiveness 10 --seed 1991540417 --center_x 8 --center_y -19 --center_z -7 --size_x 18 --size_y 9 --size_z 17
## Analysis
14. To view result			`pymol receptor.pdbqt dock.pdbqt` 
15. Lowest kcal and usually the first structure is the best binding structure.
## Example:
* ligand:	STI.sdf
* protein:	1OPJ.pdb
* Box.py:	Box 11, 90.5, 57.5, 15, 15, 15

## Loop:
To loop over many ligands:
for i in 'ls ligands'; do
	echo $i
	~/vina --receptor protein.pdbqt --ligand ligands/$i --out docked_$i --exhaustiveness 10 --seed 1991540417 --center_x 8 --center_y -19 --center_z -7 --size_x 18 --size_y 9 --size_z 17
	rm docked_$i
	echo ''
	echo ''
done
