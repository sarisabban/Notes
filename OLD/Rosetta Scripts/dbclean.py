#!/usr/bin/python3

'''
This script is used to prepare a PDB database for the Epigrafting protocol
To use:
python3 dbclean.py
'''
current = os.getcwd()
import gzip , os

#Download Database
os.system('rsync -rlpt -v -z --delete --port=33444 rsync.wwpdb.org::ftp/data/structures/divided/pdb/ ./DATABASE')
current = os.getcwd()
os.mkdir('PDBDatabase')
os.system('mv ./DATABASE/*/*.ent.gz ./PDBDatabase')
os.system('rm -r ./DATABASE')

#Organise Database
pdbfilelist = os.listdir('PDBDatabase')
for thefile in pdbfilelist:
	#Open File
	TheFile = current + '/PDBDatabase/' + thefile
	TheName = TheFile.split('.')
	#Extract Each Chain  and Save As Different Files
	InFile = gzip.open(TheFile, 'rb')
	for line in InFile:
		line = line.decode()
		if line.startswith('ATOM') or line.startswith('ANISOU'):
			chain = line[21]
			output = open(TheName[0] + chain + '.pdb' , 'a')
			output.write(line)
			output.close()
	os.remove(TheFile)
	#Compress Each File
	filelist = os.listdir('PDBDatabase')
	for files in filelist:
		filename = current + '/PDBDatabase/' + files
		if filename.split('.')[1] == 'pdb':
			try:
				f_in = open(filename , 'r')
				s = f_in.read()
				f_in.close()
				f_out = gzip.GzipFile(filename + '.gz', 'wb')
				f_out.write(s.encode())
				f_out.close()
				os.remove(filename)
			except:
				pass

#Delete Non-Protein Files
filelist = os.listdir('PDBDatabase')
for files in filelist:
	File = gzip.open('PDBDatabase/' + files, 'rb')
	for line in File:
		chain = line.decode()[17:21]
		if chain == ' DA ' or chain == ' DC ' or chain == ' DG ' or chain == ' DI ' or chain == ' DT ' or chain == '  A ' or chain == '  C ' or chain == '  G ' or chain == '  U ' or chain == '  I ':
			try:
				os.remove(current + '/PDBDatabase/' + files)
			except:
				pass
		else:
			pass

#Generate The PDB List
filelist = os.listdir('PDBDatabase')
for files in filelist:
	filename = current + '/PDBDatabase/' + files
	TheList = open('pdb.list' , 'a')
	TheList.write(filename + '\n')
	TheList.close()

#Remove All Structures Larger Than 150 Amino Acids
Script = open('script' , 'w')
Script.write('''
for file in *.pdb; do
	CHAINAnumb=`grep ATOM $file | awk '{print $5 "\t" $6}' | grep A | tail -n 1 | awk '{print $2}'`
	CHAINBnumb=`grep ATOM $file | awk '{print $5 "\t" $6}' | grep B | tail -n 1 | awk '{print $2}'`
	[[ $CHAINBnumb = *[!0-9]* || $CHAINAnumb = *[!0-9]* ]] && continue
	AminoAcids=$((CHAINBnumb-CHAINAnumb))
	echo $AminoAcids
	if (( $AminoAcids \< 150 ))
		then
			rm $file ;
	fi
done''')
os.system('bash script')
os.remove('script')

#Clean Database
result = []
for root , dirs , files in os.walk('/'):
	if 'score_jd2.default.linuxgccrelease' in files:
		result.append(os.path.join(root))
Rosetta = result[0] + '/' + 'score_jd2.default.linuxgccrelease -l pdb.list'
os.system(Rosetta)
