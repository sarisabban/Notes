**AutoDock Vina**
------------------------------------------------------------------------------------------------------------------------------------------
# Install
sudo apt-get install autodock-vina openbabel

# Protocol
## Ligand
1. Get the ligand from RCSB: `wget files.rcsb.org/ligands/view/FILENAME_ideal.sdf`
2. Convert sdf to pdbqt: babel FILENAME.sdf ligand.pdbqt -p
## Receptor
3. Get the protein receptor RCSB: `wget files.rcsb.org/download/FILENAME.pdb`
4. Open PyMol
4. Remove all water:	receptor > action > remove waters
5. Add Hydrogen:	receptor > action > hydrogens > add polar
6. Save molecule receptor.pdb
7. Convert pdb to pdbqt: babel receptor.pdb receptortemp.pdbqt -xh
8. Get only the ATOMS: grep ATOM receptortemp.pdbqt > receptor.pdbqt
## Search Space and Generate the config.text file
8. From the terminal:	pymol receptor.pdb
9. From PyMol:		run Box.py
10. From PyMol:		Box Center_X , Center_Y , Center_Z , Length_X , Length_Y , Length_Z
# Required Computation Files:
receptor.pdbqt
ligand.pdbqt
conf.txt
# The Run Computation:
vina --config config.txt
# Analysis
11. Lowest kcal and usually the first structure is the best binding structure.
