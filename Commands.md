**#Summery of Commands:**
--------------------------------------------------
**#Linux**

##Ubuntu Gnome
###For University - MUST first open web browser and put in password
sudo apt update && sudo apt full-upgrade
sudo apt install weechat vim ffmpeg pymol gnuplot tmux openconnect python3-pip git htop dssp -y && sudo python3 -m pip install numpy biopython tweepy bs4 praw zeep
* Rosetta
* PyRosetta

##Programs:
* weechat
* pymol
* gromacs
* autodock-vina
* autodocktools
* ffmpeg
* tmux
* owncloud-client
* gnuplot
* zlib1g-dev scons build-essential
* openconnect
* network-manager-openconnect-gnome
* openvpn
* network-manager-openvpn-gnome
* dssp
* git
* python3-pip
* lynx
* htop
* gtk-recordmydesktop
* firmware-b43-installer
* nmap
* grace
* tty-clock
* whois
* xscreensaver xscreensaver-gl-extra xscreensaver-data-extra

##For Debian
su
vi /etc/apt/sources.list
add after main on each line that has the word: contrib non-free
apt-get update
apt-get upgrade
apt-get install weechat vim pymol gnuplot tmux openconnect git htop dssp python3-pip firmware-b43-installer -y
python3 -m pip install numpy biopython tweepy bs4 praw
vi /etc/apt/sources.list
deb http://www.deb-multimedia.org jessie main non-free
deb-src http://www.deb-multimedia.org jessie main non-free
apt-get install ffmpeg
name: Open Terminal
command: gnome-terminal
then click Add then click crtl+alt+t
Creat Live Install USB:
su
lsblk
dd if=/home/acresearch/Desktop/ubuntu-gnome-16.04.2-desktop-amd64.iso of=/dev/sdb
--------------------------------------------------
**#MARKDOWN.md**

#Header1

##Header2

###Header3

[title](http://link)

> Block text like in quotation

*Italics*
**Bold**

* Bullet point * or + or - 

1. Number

`Code`

Body

Horisantal separation ------------
--------------------------------------------------
**#SSH**

#To Connect
ssh user@IP
Example: ssh pi@192.168.1.10

#To Retrieve File
cd to location where you want file to download (example cd Downloads)
sftp user@IP
type password
cd to location
get file_name (mget *.pdf -> to get all PDF files)

#To Send File
cd to location where you want file to upload from
sftp user@IP
type password
cd to location
put file_name (mput *.pdf -> to upload all PDF files)

#To transfer a directory
cd to location where you want file to upload from 
sftp user@IP
type password
cd to location
mkdir directory_name
put -r directory_name

For some reason mput -r needs the same directory name in the destination. But mget does not need a destination folder.

NO SFTP - If you do not want to use sftp you can simply copy files using:
To HPC:		scp /home/acresearch/Desktop/* ssabban@10.113.16.5:/fefs1/generic/ssabban/Allergy_Vaccine_pex/refine_remodel/1Q1FA/design_1Q1FA
From HPC:	scp ssabban@10.113.16.5:/fefs1/generic/ssabban/Allergy_Vaccine_pex/refine_remodel/1Q1FA/refinement_* /home/acresearch/Desktop
--------------------------------------------------
**#Networking**

#Connect through cisco VPN
sudo apt-get install openconnect
sudo openconnect IPADRESS
Enter USERNAME and PASSWORD

#nmap
sudo nmap -sP 192.168.1.*

#See which IPs I am currently connected to, and which applications are going online
sudo netstat -natp

#To find the host name
sudo netstat -Watp

#All running processes and their number
ps -a
--------------------------------------------------
**#freenode**

#Search lists:
/msg alis list
/msg alis list * -topic

#Find someone's timezone
/time <nick>
--------------------------------------------------
**#Convert Images**
#Convert a PDF file to a PNG image
convert FILENAME.pdf FILENAME.png		#<--- work best with PNG only
or for better quality
gs -sDEVICE=jpeg -sOutputFile=FILENAME.jpg -r1000 FILENAME.pdf			#<--- increase r to increase quality
gs -sDEVICE=pngalpha -sOutputFile=FILENAME.png -r1000 FILENAME.pdf

#Invert colours of image
convert -negate FILENAME.gif NEWFILENAME.gif
--------------------------------------------------
**#owndrive**
[https://my.owndrive.com]
ac.research@icloud.com

#what is your dream job?
Academia

#In what city did your parents meet?
Prague

#what was the first album you purchased?
The
--------------------------------------------------
**#Raspberry Pi On-Screen Keyboard**

sudo apt-get install matchbox-keyboard
sudo reboot
Menu>Preferences>Main Menu Editor> de-select then select
--------------------------------------------------
**#Virtual keyboard**

sudo apt-get install matchbox-keyboard -y
--------------------------------------------------
**#Audio Recording**

arecord FILENAME.wav <--- Bad Quality
--------------------------------------------------
**#ASCII TEXT ART**

[http://patorjk.com/software/taag/]
[http://www.network-science.de/ascii/]
--------------------------------------------------
**#MPI**

mpiexec --hostfile /home/pi/machinefile -np 3 ./home/pi/MY_COMMAND_FILE
where -np 3 is the total number of CPU cores in the cluster (each node has a 4 core CPU therefore in 3 node cluster = 12 cores)
--------------------------------------------------
**#TMUX**

#To start a new session
tmux new -s NAMEOFSESSION

#To exit and end a session
exit

#To exit a session without shutting it down
ctrl-B D 

#To devide a window by veritcal half
ctrl-B %

#To devide a window by horizontal half
ctrl-B "

#To change between windows
ctrl-B left or right arrow

#To scroll page up/down
crtl-B [ then use up and down arrows to scroll
q to quit

#To list all active sessions
tmux list-sessions

#To log into a session
tmux attach -t NAMEOFSESSION

#Clock:
ctrl-B t

#List Sessions:
tmux ls

#Close Session:
tmux kill-session -t aziz

#Aziz Supercomputer Environment Setup:
tmux new -s aziz -n aziz \; split-window -h -p25 \; split-window -v -p75 \; split-window -v -p50
tmux new -s aziz -n aziz \; split-window -h -p25 \; split-window -v -p50
--------------------------------------------------
**#Terminal Commands - Bash**

#Bulk Rename Sequentially Command
cd to directory where files are found
N=1; ls -tr -1 | while read file; do mv "$file" "FILENAME $N"; N=$((N+1)); done

#Rename files sequencially
N=1; ls -tr -1 | while read file; do mv "$file" "$N"; N=$((N+1)); done

#Rename all files starting at number 417 (not starting at 1)
n=417 ; for f in *.png ; do printf -v t 'video%04d.png' $((n++))  ; mv -vi -- "$f" "$t"; done

#To Shutdown
sudo poweroff

#To Reboot
sudo reboot

#To blur an image
higher number = more blur
convert FILENAME.jpg -blur 0x5 NEWFILENAME.jpg

#To download PDB files through command line
cd to location
wget http://www.rcsb.org/pdb/files/XXXX.pdb

#To connect to the NCBI ftp server
ftp ftp.ncbi.nih.gov
username is anonymous
password is any email address

#To download multiple files
wget -r -A "example0*.pdf" URL

#Find difference between 2 files
diff -u file2 file3

#To choose the option 15 after a command
printf %s\\n 15 | THEcommand

#To measure how long a command takes to finish exceuting
time <command>

#To list a directiry content with file sizes in MB
ls -lh

#This command with give the (Disk Usage) total size of each file or directory in MB
du -sh *

#Disk space usage of HDD - HDD is the one mounted on root /
df -h
or
sudo apt-get install pydf
pydf

#To copy a directiry - not just cp like other files:
cp -a FROM TO

#To measure CPU temperature
cat /sys/class/thermal/thermal_zone0/temp

#To continuously watch CPU temperature
watch /opt/vc/bin/vcgencmd measure_temp

#To find how many CPU cores the computer has
cat /proc/cpuinfo | grep processor | wc -l

#Check to see if a program is installed
sudo dpkg -l | grep PROGRAM_NAME

#Checksum
md5sum FILENAME

#To compile a C program
gcc -o NEW_FILE_NAME FILE_NAME.c

#Compress a directory
tar czvf NEWFILENAME.tgz NAME_OF_FILE_OR_DIRECTORY_TO_COMPRESS
	-c: Create an archive.
	-z: Compress the archive with gzip.
	-v: verbose mode.
	-f: Allows you to specify the filename of the archive.
	-x: uncompress and tar

#Uncompress a file
tar xzvf FILENAME.tgz

#Compress directory + encrypt
tar czvf - NAME | gpg -c -o NAME.tgz.gpg

#Uncompress + decrypt
gpg -d NAME.tgz.gpg | tar xzvf -

#To navigate to a connected HDD
df
to find out where it is mounted (look at the right where it says mounted on)
cd to that path

#Count items in direcotry
CD to derectory
ls | wc -l

#Making an alias
ln -s PATH/FROM PATH/TO

#Find all files with extention .pdb in current directory
find -name "*.pdb"

#List in order
ls | sort -n
ls -lv

#To terminal install a package that is not found in the APT repository
wget URL.deb
sudo dpkg -i FILENAME.deb


#Find local router IP
netstat -nr | awk '$1 == "0.0.0.0"{print$2}'

**CAT**
display content
ctl-d to end editing

**REDIRECTION**
input > output replace
input >> output append (add)

**MORE**
q to exit

**LESS**
/search downward n for next
?search upward n for next
G document end
g document begining
q to exit
down key for 1 line
space for down 1 page
up key for up 1 line
b for up 1 page

**FILE**
detect type of file content
file index -> HTML document text

**HEAD**
-n 2 to print the first 2 lines
-2 -2 to print everything except the last 2 lines

**TAIL**
-n 2 to print only the last 2 lines
-2 +2 to print everything except the first 2 lines

**PIPES**
| to pass command to another command (use output of one command as input for another command)

**FIND**
. for current directory
-name "" search by file name (case sensitive)
-iname "" search by file name (not case sensitive)
-type f search by files
-print print directory path
find . -name '*.pdb' -exec mv {} NEWDIRECTORY \; -> find all file names then execute move on all found comand
find . -name '*.sc' -exec cat {} + > NEWFILE -> find all file names then execute copy on all found comand

**LOCATE**
locate filename to show path to file

**WC**
word count
-l number of lines
-w number of words
ls | wc -l counts how many lines (files) in current directory

**UNIQ**
removes all but 1 copy of successive repeated lines
-c count occurences of each repeated line
-d only display lines that are repeated
-u only display non repeated lines 

**PERMISSIONS**
ls -l
1 - normal file, d directory, b block special file, c character special file
234 permission for file owner
567 permission for group
789 permission for everybody else
r read w write x execute
number of links to the file
name of file owner
name of group
size in bytes
date
name
change permission by chmod
to change permission for group to be able to execute
chmod g+x
chmod g-x to take away permission
o+rwx to make file r w x for everybody
u for owner of file
ug user and group
ugo-rwx take away everything from everyone
u+rwx,g+rx,o+r
user	: is the owner of the file
group	: is users that belong to a certain group (generic, biology, etc...)
everyone: other users from other groups

**COMMAND NESTING**
`` backtics
nests a command within a command
cat `ls -l` will run ls -l first then the result will be used as an argument for cat
more better is to use $()

**ECHO**
x=10 -> echo $x
echo "striong" -> string
echo "string" > file.txt creates and saves string into file
-e uses \n and as new line and \t and tab -> echo -e "Tecmint \nis \na \ncommunity \nof \nLinux \nNerds" 
Tecmint
is
a
community
nof
Linux
Nerds
echo -e "Tecmint \tis \ta \tcommunity \tof \tLinux \tNerds" 
Tecmint 	is 	a 	community 	of 	Linux 	Nerds

**SORT**
sort alphabbetically
-r reverese alphabbetically
-n numeric sort
-k sort in location
-nk +2 in a table column 2 -n for sorting numbers as a whole instead of separatly 1 11 111 2 22 222 3 33 etc -k for column
-M sort months
-o output and save file of sorted arrgument (in place of > which also works)
-Vk natural sort of (version) numbers within text. file_1 file_2 file_3 instead of file_1 file_10 file_100 

**GREP**
search and find then print a pattern
grep "PATTERN" FILENAME
-n gives line number
-E "PATTERN" allows search for regular expressions (search PA will show all words that has pa in it) "Pa(r|i)" search r or i
-w search exact word
-i remove case sensitive

**SED**
filtering and transforming text
sed 's///' FILENAME -> s for substitute
sed 's/from/to/' will change only the first instance in each line
sed 's/from/to/g' -> g for global (substitute everywhere)
sed 's/from//g' -> removes all searched pattern
-i edit in line (edit and save the same original file)
sed 's/from/to/g;s/again/another/g' -> replace more than one word
sed 's///Ig' -> ignore case apply globally
sed 's/\bnew\b/to/g' -> \b a boundery before and after a word means match only this word new will not match newer
sed '/from/d' -> remove whole line if pattern in found in it
sed '1!{/^#/d;}' -> remove all lines that start with # (remove all comment from a script) except first line #!/bin/bash
sed 's/^from/to/g' -> ^ replace only first instance in a line if line begins with it
sed 's/from$/to/g' -> $ replace only last instance in a line if line ends with it, if line does not end with it, it won't be replaced
sed 's/[0-9]/*/g' -> replace all numeric characters with *, also [a-z] case sensitive
sed 's/[^0-9]/*/g' -> replace everything that is NOT a number
[a-z][A-Z] -> small letter followed by capital letter
[a-zA-Z] or [A-z] -> any letter small or capital
sed 's/[0-9]/(&)/g' -> & = match search pattern, put () around every searched number 0-9 (will not worj in 2 digit numbers)
sed 's/[0-9][0-9]*/(&)/g' -> * = or, search [0-9] or [0-9][0-9] (works with 2 digit numbers)
sed 's/\/w/d/g' -> \ = ignore the / in front of it and treat as part of searched string rather than sed command, replace all /w with d
any delimiter can be used -> sed 's///' FILENAME or sed 's|||' FILENAME or sed 's:::' FILENAME as long as they are consistant

**AWK**
programming language that uses grep, sed, cut, tr
awk '{print}' FILENAME -> look for all lines, print all these lines
awk '/pattern/ {print}' FILENAME -> look for lines that include pattern, print only these lines
awk '$3>1 {print}' FILENAME -> look for lines that include a number greater than 3, print only these lines
awk '{print $2 "\t" $3 }' FILENAME -> feild 2 insert a tab feild 3 (second word seperated by space and third word seperated by space)
awk '{print $2 "," $3 }' FILENAME -> feild 2 insert a comma feild 3
awk '$3>1 {print $2}' FILENAME -> look for lines that include a number greater than 3, print only second feild of these lines
BEGIN {print "THIS IS A HEADER"} -> insert header above extracted words and tables
END {print "THIS IS A FOOTER"} -> insert footer after extracted words and tables
FS=""	Feild Seperator			
OFS=""	Output Field Separator		
RS=""	Record Separator		
ORS=""	Output Record Separator		
NR=""	Number of Records		
NF=""	Number of Fields		
FNR=""	Number of Records		
awk '{print}' ORS=" " -> remove all newlines between lines (merge all lines into one continues line)

**CRON**
crontab -e

minute	hour	day	month	week	COMMAND		>>	log file	2>&1
00	15	*	*	*	python3		>>	filename.log	2>&1
--------------------------------------------------
**#Using Aziz Supercomputer**

#Specs
Red Hat
230 TFLOPS
495 nodes (each node has 2 CPUs with 12 cores each) therefore each node has 24 cores.
run script and computation from the following directory because the file system is customised for HPC and therefore is very fast:
cd /fefs1/generic/ssabban

#Show available modules
module avail


#To Compute - Write the following .pbs file
	#!/bin/bash
	#PBS -N Rosetta					Name of the computation
	#PBS -q thin					Computation type
	#PBS -l select=1:ncpus=24:mpiprocs=24		Number of nodes to use select=2 is 2 nodes to use etc...
	#PBS -l walltime=1:00:00			computation limit for testing computation (using less than 6 hour = front of queue)
	#PBS -e	/home/ssabban/				Path to error log	- no path there saved in home directory
	#PBS -o	/home/ssabban/				Path to output log	- no path there saved in home directory
	BASH SCRIPT

#Computation Types
thin	380 nodes	computation limit 48 hours	each node space 96GB
fat	112 nodes	computation limit 48 hours	each node space 256GB
thin_1m	380 nodes	computation limit 1 month	each node space 96GB
fat_1m	112 nodes	computation limit 1 month	each node space 256GB

#To run the .pbs script
qsub sample.pbs

#To view all running computations:
qstat
R = running	Q = queued	H = held	E = exited after having run	not in list = computation finished
T = moved to new location	W = waiting for execution	S = suspended

#To view my running computations
qstat -u sabban

#To Delete a Job
qdel job_identifier
to forcefully delete a job that refuses to stop: qdel -Wforce job_identifier

#To submit array computing (submitting the same job multiple times) use the # --array=1-100 (1-100 nodes when available) and change the output file name into a variable name so it does not get over written:
	#PBS --array=1-100

	XXXX_${SLURM_ARRAY_TASK_ID}.output but make sure the output is outside the flags file so as to incorporate the variable job id

	in PBS it will be 
	#PBS -J 1-100
	.${PBS_ARRAY_INDEX}
--------------------------------------------------
**#PSIPRED**
[http://bioinf.cs.ucl.ac.uk/web_servers/web_services/]

#Required Programs
sudo python3 -m pip install zeep

#Inspect WSDL file:
python3 -mzeep http://bioinf.cs.ucl.ac.uk/psipred_api/wsdl

#To run psipred
import zeep

client = zeep.Client('http://bioinf.cs.ucl.ac.uk/psipred_api/wsdl')

sequence = "MQTIQMIVTHPHLPRALTTQITVQKNQNLAAKDLAGLIASQMESLRGITDEDLRKFKQAHAIVYFKNGTKLTLTLKSGTHTANLFNAKDIKSIQVNAD"
email = "psipred@cs.ucl.ac.uk"
title = "testing"

#Submit
print(client.service.PsipredSubmit(sequence, email, title, "True", "False", "False", "all"))

#Get Result
print(client.service.PsipredResult('6320b648-46fd-11e7-8a89-00163e110593'))
--------------------------------------------------
**#Rosetta Compilation Commands**

#Rosetta
[https://www.rosettacommons.org/software/academic]
Academic_User
Xry3x4

wget https://www.rosettacommons.org/download.php?token=a64t4Q78m&file=rosetta_src_3.7_bundle.tgz
wget https://www.rosettacommons.org/downloads/academic/3.6

sudo apt-get install zlib1g-dev scons build-essential -y
tar -xvzf rosetta_src_3.7_bundle.tgz
cd {ROSETTA}/main/source
./scons.py mode=release bin

#Ubuntu 17.04
in home directory .bashrc add at the end of the line:
export LD_LIBRARY_PATH=/home/acresearch/rosetta_src_2017.08.59291_bundle/main/source/build/external/release/linux/4.10/64/x86/gcc/6.3/default/:$LD_LIBRARY_PATH

#Cannot Compile on AZIZ straighforward because python -V -> Python 2.6.6
#To compile first load python 2.7.9 by these two commands
module use /app/utils/modules
module load python-2.7.9
Then compile normally with ./scons.py mode=release bin
--------------------------------------------------
**#GitHub**

***COMMANDS ONLY WORK WHEN IN THE GIT DIRECTORY IN MY COMPUTERs

#To get a project
git clone PASTE CLONE LINK
will download the project and make a directory with its name in the location you specify

#To see a project's updates and which of my files are not yet sent (or newly sent) to the repository
git status

#To upload a file to the server repository
git add FILENAME
git add . -> everything in my directory

#Commit new file to the repository
git commit -m "a message"

#Sync everything in my directory to the online repository
git push -> to send
git pull -> to get

#Summery
git clone URL		Clones a project into your computer
git pull 		Update local directory (musy be in directory)
git status 		Shows what is different between computer and server directory
git add FILENAME	Add newly generated file so GitHub can track it (must then be commited then pushed)
git commit FILENAME	Commit changes to computer directory
git push FILENAME	Push to server
--------------------------------------------------
**#GPG**

#Generate public/private key
gpg --gen-key

#Find all key IDs
gpg --fingerprint

#Send public key to server
gpg --keyserver pgp.mit.edu --send-keys KEY_ID

#Revokation certificate
gpg --gen-revoke KEY_ID

#Revoke key
gpg --import REVOKE_CERTIFICATE_TEXT_FILE.txt
gpg --keyserver pgp.mit.edu --send-keys KEY_ID

#Encrypt asymetric
gpg --encrypt --sign --armor -r KEY_ID (of recepient)
type message
end by crt-d

#Decrypt
gpg -d
paste encrypted message
crt-d

#Search key server
gpg --keyserver pgp.mit.edu --search-keys SEARCH_PARAMETER

#Export key
gpg --export -a KEY_ID
gpg --export-secret-keys -a KEY_ID

#List all keys
gpg --list-keys

#Delete key
gpg --delete-secret-keys KEY_ID
gpg --delete-keys KEY_ID

#Import someone's key from server
gpg --keyserver pgp.mit.edu --recv-keys KEY_ID

#Encypt symmetric
gpg -c -a
write message
crt-d
--------------------------------------------------
**#FFMPEG**

#Re-encode .mkv to .mp4 To Work With Samsung TV
ffmpeg -i 1.mkv -vcodec libx264 -r 25 -crf 23 -ab 384k -acodec ac3 3.mp4
#loop
for i in *.mkv; do ffmpeg -i "$i" -vcodec libx264 -r 25 -crf 23 -ab 384k -acodec ac3 "${i%.mkv}.mp4"; done

#No lost quality
ffmpeg -i input.mkv -c copy -sn -movflags faststart out.mp4

#4k to 1080p only video
ffmpeg -i input.mkv -c:v libx264 -preset veryfast -level 41 -crf 21 -vf scale=1920:1080 -c:a copy -sn out.mp4

#4k to 1080p and change audio to different format to work with Sumsung TV
ffmpeg -i input.mkv -c:v libx264 -preset veryfast -level 41 -crf 21 -vf scale=1920:1080 -c:a aac -b:a 192k -ac 2 -sn out.mp4

ffmpeg -i input.mkv		#input
-c:v libx264			#to encode in H264
-preset veryfast		#compress libx264 quickly (slower presets compress better but take longer - default is medium slowest is placebo)
-level 41			#set level to 4.1. as in tells the encoder to limit memory requirements (https://en.wikipedia.org/wiki/H.264/MPEG-4_AVC#Levels)
-crf 21				#constant quality (lower numbers look better, higher numbers give - default=23)
-vf scale=1920:1080		#control scale and resolution 
-c:a aac			#copy all video stream Advanced Audio Coding (AAC) encoder
-b:a 192k			#audio quality
-ac 2				#2 audio channels
-sn				#disable subtitles in output
out.mp4				#output
ffmpeg -i input.mkv -c:v libx264 -preset placebo -level 41 -crf 21 -vf scale=1920:1080 out.mp4

#Loop
for i in *.mkv; do ffmpeg -i "$i" -vcodec libx264 -r 25 -crf 23 -ab 384k -acodec ac3 "${i%.mkv}.mp4"; done

#Trim from second 13 for 3 steps
ffmpeg -ss 13 -t 3 -i IMG_1669.mov -vcodec copy -acodec copy Shake.mov

#Merg video files
ffmpeg -f concat -safe 0 -i <(for f in ./*.mov; do echo "file '$PWD/$f'"; done) -c copy output.mov

#Record Desktop Screen - Lossless quality but VERY LARGE FILE SIZES
ffmpeg -f x11grab -s 1280x800 -r 30 -i :0.0 -c:v ffv1 X.mkv

#Record Desktop Screen - Lossless quality but SMALL SIZE
ffmpeg -f x11grab -s 1280x800 -framerate 30 -i :0.0 -c:v libx264rgb X.mkv
--------------------------------------------------
**#Python**

#Python Terminal Colours
[https://en.wikipedia.org/wiki/ANSI_escape_code]
print('\x1b[31m'+'Hello World'+'\x1b[0m')

Text			Background
0	Back To Normal	0
30	Black		40
31	Red		41
32	Green		42
33	Yellow		43
34	Blue		44
35	Magenta		45
36	Cyan		46
37	White		47

#Compile a python script
python3 -m py_compile FILENAME.py
Give is .pyc that can be run with python3 FILENAME.pyc
renaming it will not affect it.
it can also made executable with chmod 775 FILENAME.pyc ---> 500 is best
--------------------------------------------------
**#TWEEPY**
#search a hashtag
hash = tweepy.Cursor(api.search , q = '#.......').items(1)
loop to get info from it

#Methods
.retweet()				retweets a tweet
.favorite()				favitres a tweet
.user.follow('@...')			follows a user
.update_status(status='...')		tweets

#Prints
tweet.user.screen_name			prints the username @...
tweet.text				prints the actual tweet

#About Users
x = api.get_user('@...')		get user info, loop to print info
x = api.user_timeline(screen_name = '@....' , count=200)
--------------------------------------------------
**#GNUPLOT**
sudo apt-get install gnuplot
Enter:
gnuplot

#Quick References To Commands
[http://gnuplot.sourceforge.net/docs_4.0/gpcard.pdf]
* test									to see what each flag does.
* plot 'FILENAME.dat'							plots the first column as x and second column as y (scatter).
* plot 'FILENAME.dat','FILENAME2.dat'					plots first file and second file.
* plot 'FILENAME.dat'lc rgb 'red','FILENAME2.dat' lc rgb 'black'		first file with red dots second file with black dots.
* set title 'example'							sets the title of the plot, excecute before plotting.
* set ylabel 'example'							sets y axis label, excecute before plotting.
* set xlabel 'example'							sets x axis label, excecute before plotting.
* '{/symbol a}'								use greek symbol alpha.
* unset key								remove plot key (which value has which symbol/colour).
* set key									bring back key.
* set grid								add grid.
* save 'FINENAME.gp'							save file+format into file. chmod a+x makes it ./ executable
* load 'FINENAME.gp'							to open saved file.
* set pointsize								set the size of each point

#Within Command:
* title 'TITLENAME'							title in key instead of filename (shortened to t 'TITLENAME').
* plot [0:3] [4:5]							set axis range x-axis 0-3 y-axis 4-5.
* using 1:2								using clumn 1 over column 2 of data in file (u).
* with lines								plot graph of line connecting points (shortened to w l). 
* with linespoint								plot graph of line connecting points and show points (w lp).
* smooth unique w l							if line plot is jumbled makes sure it connects line correct.
* bezier 									curvy lines.
* lc rgb 'black'								change point colour into black.
* with yerrorbars								add error bars to the y axis.
* with xerrorbars								add error bars to the x axis.
* pointtype 2								set shape of points 1=+ 2=x 5=square
* set yrange [min:max]							Max and min y axis value
* set xrange [min:max]							Max and min x axis value

#To set angstrom symbol
set encoding iso_8859_1
set xlabel 'RMSD ({/E \305})'

##For Abinitio Only Red
gnuplot
set xlabel 'RMSD'
set ylabel 'Scores'
set yrange [:-80]
set xrange [0:20]
set title 'Ab Initio of 3Q4HB'
plot '1' lc rgb 'red' pointsize 0.2 pointtype 7 title ' '

#FOR FINAL RESULTS ---> Ubuntu
##Generate two files (2= the ab-initio file minus the lowest 200 scores, 3= the lowest 200 scores):
cat SCOREvsRMSD.dat | sort -nk +2 | tail -999800 > 1.dat
cat SCOREvsRMSD.dat | sort -nk +2 | head -200 > 2.dat

gnuplot
set encoding iso_8859_1
set xlabel 'RMSD (\305)'
set ylabel 'Scores'
set yrange [:-80]
set xrange [0:20]
set title 'Abinitio of The REX3 Structure'
plot '1.dat' lc rgb 'red' pointsize 0.2 pointtype 7 title '', \
'2.dat' lc rgb 'blue' pointsize 0.2 pointtype 7 title ''

#For RedHat:
set encoding iso_8859_1
set term post eps enh color
set xlabel 'RMSD (\305)'


###Quicky
gnuplot
set yrange [:-60]
plot 'SCOREvsRMSD.dat'
--------------------------------------------------
**#PYMOL**
[http://www-cryst.bioc.cam.ac.uk/members/zbyszek/figures_pymol]

#Save FASTA Sequence
save FILENAME.fasta

#Video Making
setup scenes
Movie > Program > Scene Loop > Steady > 4 seconds each

##for pymol 1.8.6 use these two commands
set ray_trace_frames
mpng video, width=2400

##for pymol before 1.8.6:
make a .py file with the following script
	cmd.load('structure.pse')
	cmd.viewport (2400 , 2400)
	cmd.set('ray_trace_mode' , 0)
	cmd.frame(2250)
	cmd.mpng('video')

then open from terminal in code mode
pymol -c FILENAME.py

iOS video convert command
ffmpeg -f image2 -i video%4d.png -r 30 -vcodec libx264 -pix_fmt yuv420p -acodec libvo_aacenc -ab 128k -profile:v high -level 4.2 video.mp4

iOS simpler video convert command
ffmpeg -f image2 -i video%4d.png -pix_fmt yuv420p -vcodec libx264 -profile:v high -level 4.2 video.mp4

#Measure distance between CA of different residues
distance i. 10 and n. CA, i. 40 and n. CA

#Save only selected into new .pdb file
save FILENAME.pdb, sele

#Trace structure with black line - works after trace 2400
set ray_trace_mode,1

#Select residues
* select all alenine:	sele resn ala
* select by type:	selec resn arg+his+lys
* select by number:	select nterm, resi 7+10+11+14

#Isolate sequence into new file
Action > Extract Object
save FILENAME.pdb, obj01

#Find Cavities
set cavity_cull
set cavity_cull, <NUMBER> <--- lower number = more sensitive

#Colour all structures by rainbow
unset ribbon_color, (*)
set ribbon_color, default
for name in cmd.get_object_list(): \
  cmd.spectrum(selection=name + " and polymer")

#Trace structure with black line - works after trace 1000,1000
set ray_trace_mode, 1

##For Publication:
1. Align all to designed structure.

2. Copy/Paste the following:
set cartoon_fancy_helices, 1
cartoon loop, (c.0.*)
set cartoon_highlight_color, grey50
unset ribbon_color, (*)
set ribbon_color, default
for name in cmd.get_object_list(): \
  cmd.spectrum(selection=name + " and polymer")
set ray_opaque_background, off
set antialias, 1
set ray_trace_mode, 1

3. Set background to white.
4. ray 2400
5. save FILENAME.png

**Pymol API**

#Run
	pymol FILENAME.py		-----> With GUI
	pymol -c FILENAME.py		-----> Without GUI

#Always add
	pymol.finish_launch()		-----> To finish launching GUI before running any pymol commands

#Load
	cmd.load('STRUCTURE.pdb')

#Quit
	cmd.quit()

#Show
	cmd.cartoon('automatic' , 'selection name') 	-----> selection name can be the selection (Core, Surf, Bound) or the filename without the .pdb

	cmd.show('cartoon' , 'selection name')		-----> Show (cartoon, stick etc...)
	cmd.show_as('cartoon' , 'selection name')	-----> Show As (cartoon, stick etc...)
	cmd.show_as('cartoon')				-----> Show As for all selections

#Colour
	cmd.color('colour name' , 'selection name')

#Distance
	x = cmd.distance('/structure///1/CA' , '/structure///2/CA')
	print(x)

#Cavity
	cmd.set('cavity_cull', 0)

**OTHER COMMANDS**
#Convert GROMACS Images Into A Movie
1. Run following commands in pymol:
python
os.mkdir('Pics')
os.chdir('Pics')
for x in range(501):
	cmd.frame(x+1)
	cmd.ray(1000,1000)
	cmd.png(str(x+1) + ".png")
python end

2. CD to image location then run following commands in terminal:
rename 's/\d+/sprintf("%05d",$&)/e' *.png
ffmpeg -i %05d.png Movie.avi

***To convert between video codecs:
ffmpeg -i FILENAME.mkv -vcodec copy -acodec copy FILENAME.avi
http://www.zamzar.com/convert/mpg-to-mp4/#

#Save FASTA Sequence
save /home/computer/Desktop/FILENAME.fasta

#Export secondary strcutres to a text file (exports each amino acid and which secondary structure it belongs to)
iterate n. CA, print resi + ':' + ss
iterate n. CA, print ss

`ATOM is for DNA or Protein atoms`
`HETATM is for small molicule atoms such as ligands ions etc...`
`TER separates chains to indicate they are not physically connected`
--------------------------------------------------
**#PYROSETTA**
#Download
[http://www.pyrosetta.org]
[http://graylab.jhu.edu/download/PyRosetta4/archive/release/]
levinthal
paradox

#Install
1. tar -vjxf PYROSETTA_FILENAME.tar.bz2
2. cd to PYROSETTA_DIRECTORY
3. cd setup && sudo python3 setup.py install

#Test if installed correctly by typing the following
1. python3 FILENAME
2. import pyrosetta; pyrosetta.init()

#Tutorials
[http://www.pyrosetta.org/tutorials]

#Heading and Imports
from pyrosetta import *			#Import everything in pyrosetta
from pyrosetta.toolbox import *		#Import everything in toolbox
init()					#Imports the rosetta database
pose = pose_from_pdb('1YY8.clean.pdb')	#pose

#Download From RCSB And Cleans The File
pose_from_rcsb('1YY8')

#Clean PDB
cleanATOM('FILENAME.pdb')

#Examine File
x = pose_from_pdb('CLEANFILE.pdb')		#Parses the file
print(x)					#Prints all pose information
print(pose.sequence())				#Prints only FASTA sequence (will combine multiple chains)
print(pose.total_residue())			#Prints number of residues (with all chains combines)
print(pose.residue(500).name())			#Prints the name of the 500th residue in tri letters (with all chains combined, assuming that the first residue is 1)
print(pose.pdb_info().chain(500))		#Print the chain that has the 500th residue
print(pose.pdb_info().number(500))		#Print the actual residue number within the chain of the 500th residue of the whole combined chains
print(pose.pdb_info().pdb2pose('A', 100))	#Print the Rosetta internal number for residue 100 of chain A

#Get Secondary Sstructure of Each Residue
get_secstruct(pose)

#Angles φ,ψ,χ1 of A Certain Residue
print(pose.phi(5))
print(pose.psi(5))
print(pose.chi(1,5))					#chi 1 i.e Carbon alpha - Ca

#Bond Leangth in Angrstom
N  = AtomID(1, 5)					#Nitrogen of residue 5
Ca = AtomID(2, 5)					#Ca of residue 5
C  = AtomID(3, 5)					#C of residue 5

print(pose.conformation().bond_length(N, Ca))		#Length between N and Ca
print(pose.conformation().bond_length(Ca, C))		#Length between C and Ca
print(pose.conformation().bond_length(N, Ca, C))	#Length between N and C and Ca

#Mutate A Residue
mutate_residue(pose, 108, 'A')

#Change Bond Lengths
pose.conformation().set_bond_length(N, Ca, 1.5)		#Change to the final value of 3.5

#Bond Angles in Radians
print(pose.conformation().bond_angle(N, Ca, C))

#Change Bond Angles in Radians
pose.set_phi(5, -60)
pose.set_psi(5, -43)
pose.set_chi(1, 5, 180)

pose.conformation().set_bond_angle(N, Ca, C, 110./180.*3.14159)

#Save Modified PDB
pose.dump_pdb('FILENAME.pdb')

#Visualise Changes in PYMOL
1. cd to location of PyMOL-RosettaServer.py
2. start pymol by typing pymol
3. in the pymol terminal type the following: run PyMOL-RosettaServer.py
4. now pymol is listening to rosetta
5. in the python script add the following lines after the pose_from_pdb() function:
pymol = PyMOLMover()
pymol.apply(pose)
6. run a loop and include pymol.apply(pose) within the loop to update the pymol picture
7. run script

##example1:
from pyrosetta import *
from pyrosetta.toolbox import *
init()

pose = pose_from_pdb('1YY8.clean.pdb')

pymol = PyMOLMover()
pymol.apply(pose)

N  = AtomID(1, 5)
Ca = AtomID(2, 5)
C  = AtomID(3, 5)

length=1

while True:
	pose.conformation().set_bond_length(N, Ca, length)
	print(pose.conformation().bond_length(N, Ca))
	length=length+1
	pymol.apply(pose)

##example2:
from pyrosetta import *
from pyrosetta.toolbox import *
init()

pro='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'

pose = pose_from_sequence(pro,'fa_standard')
pymol = PyMOLMover()
pymol.apply(pose)

aa=1
for x in range(len(pro)):
	pose.set_phi(aa, 45); pose.set_psi(aa, 45)
	print(pose.phi(aa)); print(pose.psi(aa))
	aa=aa+1
	pymol.apply(pose)

pose.dump_pdb('helix.pdb')

#Scoring
0. USE HEADING FROM PERVIOUS SECTION

1. LOAD THE DEFAULT FULL-ATOM ENERGY TERMS
scorefxn = get_fa_scorefxn()			#print(scorefxn) to see these energy terms and their weights

2. SCORE PROTEIN:
print(scorefxn(ras))				#prints the total score for the portein
scorefxn.show(ras)				#prints a breakdown showing the value of each energy term


#Score for A Particular Amino Acid
print(ras.energies().show(24))				#shows each enery term for this particular amino acid number 24 - must include everything previously
print(pose.energies().residue_total_energy(5))		#shows total score for a specified residue

#Get All H-Bonds
hbond_set = get_hbonds(ras)
hbond_set.show(ras)

#Get Particular Amino Acid's H-Bond
hbond_set.show(ras, 24)

#Change Scoring Weights
scorefxn = get_fa_scorefxn()			#Load scoring terms
scorefxn2 = ScoreFunction()			#declares the object scorefxn2 as part of the ScoreFunction class
scorefxn2.set_weight(fa_atr, 1.0)		#Change this particular term's weight (SCORE TERM, NEW WEIGHT)





#Movers
```python
#!/usr/bin/python3

import subprocess , time

from pyrosetta import *
from rosetta.protocols.moves import *
from rosetta.protocols.simple_moves import *
init()

#1 - Find Path To PyRosetta
def Find(Filename):
	''' Find path to PyRosetta '''
	''' Returns string with path '''
	result = []
	for root , dirs , files in os.walk('/'):
		if Filename in files:
			result.append(os.path.join(root))
	return(result[0]+'/')

#3.1 - Setup PyMOL For Visualisation
def PyMol_Start():
	''' Starts PyMOL and allows it to listen to PyRosetta ''' 
	''' Opens PyMOL '''
	PATH = Find('PyMOL-RosettaServer.py')
	Temp = open('temp.py' , 'w')
	Temp.write("import pymol\npymol.finish_launching()\ncmd.show_as('cartoon')\ncmd.set('cavity_cull', 0)\n")
	Temp.write("cmd.do('run " + PATH + "PyMOL-RosettaServer.py')")
	Temp.close()
	new_terminal = "x-terminal-emulator -e pymol temp.py".split()
	processes = [subprocess.Popen(new_terminal)]
	time.sleep(5)
	AddPyMOLObserver(test, True)
	os.remove('temp.py')

#3.2 - Live PyMOL Visualisation:
def PyMol(pose):
	''' Transmits what is happening to a structure in PyRosetta to PyMol '''
	''' Shows structure in PyMol or updates PyMol with new structure '''
	pymol = PyMOLMover()
	pymol.apply(pose)
	pymol.send_ss(pose , ss = '')
	pymol.keep_history(True)
#--------------------------------------------------------------------------------------------------
#Generate pose from .pdb file
pose = pose_from_pdb('structure.pdb')

#Make COPY of pose and call it start
start = pose_from_pdb('structure.pdb')
test = Pose()
test.assign(start)

#Rename the two poses (original = start and copy = test)
start.pdb_info().name('start')
test.pdb_info().name('test')

#Visualise both poses in PyMol
PyMol_Start()
PyMol(start)
PyMol(test)
#--------------------------------------------------------------------------------------------------
#The Rosetta Protocols:
for x in dir(rosetta.protocols): print(x)
Here is a list of all the rosetta protocols that can be used in PyRosetta dir() into each one to see its sub functions and how to use them. Sometimes if you just called them on their own like rosetta.core.scoring.packstat.compute_packing_score() it will tell you the arguments it take

#List of All The PyRosetta Movers:
rosetta.protocols.moves
you must import spesifically the class in order to use it, importing all of rosetta does not work, example: from rosetta.protocols.vip import * to import all the VIP_Mover() class and all the functions under it

#The Rosetta Core:
rosetta.core
Has the rosetta core functions like TaskFactory and all the scoring functions

#List of functions you can apply to a pose:
for x in dir(pose): print(x)
#--------------------------------------------------------------------------------------------------
#Small and Shear Moves
kT = 1.0
n_moves = 1
#Move Map that says make all backbone torsion angles free to change
movemap = MoveMap()
movemap.set_bb(True)
#Setup Movers: use the MoveMap with kT for metropolis criterion and make 1 move
small_mover = SmallMover(movemap, kT, n_moves)
shear_mover = ShearMover(movemap, kT, n_moves)
#Apply Small or Shear Move
#small_mover.apply(test)
shear_mover.apply(test)
#Print Data
print(small_mover)
print(shear_mover)



#Minimisation Mover
min_mover = MinMover()
#Requires a Move Map and Score Function
movemap = MoveMap()
scorefxn = get_fa_scorefxn()
#attach these objects to the mover
min_mover.movemap(movemap)
min_mover.score_function(scorefxn)
#Apply minimisation
min_mover.apply(test)
#Print Data
print(min_mover)



#Monte Carlo Mover
kT = 1.0
scorefxn = get_fa_scorefxn()
#Setup Mover
mc = MonteCarlo(test, scorefxn, kT)
#Make several moves with movers
#Then
#Show whether the moves are accepted or regected (True = accept False = reject) if rejected the pose goes back to the way it was
print(mc.boltzmann(test) , scorefxn(test))
#Show details after loop
mc.show_scores()	#Shows lowest accepted score
mc.show_counters()	#Number of tials (iterations) and probability of accepted moves in total
mc.show_state()		#Both?
#Example:
#kT = 1.0
#scorefxn = get_fa_scorefxn()
#mc = MonteCarlo(test, scorefxn, kT)
#for attempt in range(10):
#	shear_mover.apply(test)
#	small_mover.apply(test)
#	shear_mover.apply(test)
#	small_mover.apply(test)
#	print(mc.boltzmann(test) , scorefxn(test))



#Trial Mover
#Combines a mover and MonteCarlo i.e: moves the pose and then takes desicion whether that move is good or bad
#Setup MonteCarlo Mover
kT = 1.0
scorefxn = get_fa_scorefxn()
mc = MonteCarlo(test, scorefxn, kT)
#Setup Small Mover (Done Previously)
#Setup Trial Mover
trial_mover = TrialMover(small_mover, mc)
trial_mover.apply(pose)
#Data
print(trial_mover.num_accepts())	#Print number of accepted moves
print(trial_mover.acceptance_rate())	#Print rate of acceptance
#Example:
#kT = 1.0
#scorefxn = get_fa_scorefxn()
#mc = MonteCarlo(test, scorefxn, kT)
#trial_mover = TrialMover(small_mover, mc)
#for i in range (10):
#	trial_mover.apply(test)
#print(trial_mover.num_accepts())
#print(trial_mover.acceptance_rate())



#Sequence Mover
#Combines movers in succession (one after the other)
seq_mover = SequenceMover()
seq_mover.add_mover(small_mover)
seq_mover.add_mover(shear_mover)
seq_mover.add_mover(min_mover)
#Make into Trial Mover
#trialmover = TrialMover(seq_mover, mc)



#Repeat Mover
#Loops (repeats) a number of times
repeat_mover = RepeatMover(trialmover, 2)
repeat_mover.apply(test)

#Classic Relax Mover
from rosetta.protocols.relax import *
relax = ClassicRelax()
relax.set_scorefxn(scorefxn)
relax.apply(pose)

#Packing Mover
#Setup PackerTask (Similar to MoveMap - specifies degrees of freedom)
task_pack = standard_packer_task(pose)			#call PackerTask 
task_pack.restrict_to_repacking()			#Remove for RosettaDesign (becuase without it = allows any amino acid residue to be swapped in for another. With it - only allows rotamers from the current residue at current position to be used)
task_pack.temporarily_fix_everything()			#Add to prevent all amino acids from being repacked
task_pack.temporarily_set_pack_residue(47, True)	#Add to move only one spesific amino acid
print(task_pack)					#Confirm that only residue 49 will be acted apon
#Construct Packing Mover (requires a score function)
scorefxn = get_fa_scorefxn()
pack_mover = PackRotamersMover(scorefxn, task_pack)
#Apply Move
pack_mover.apply(pose)
#Example: Total structure refinement
task_pack = standard_packer_task(pose)
task_pack.restrict_to_repacking()
print(task_pack)
#Construct Packing Mover (requires a score function)
scorefxn = get_fa_scorefxn()
pack_mover = PackRotamersMover(scorefxn, task_pack)
#Apply Move
print(scorefxn(pose))
pack_mover.apply(pose)
print(scorefxn(pose))



#Design Mover
#Design calculations can be accomplished simply by packing side chains with a rotamer set that includes all amino acid types
#Generate Resfile
#generate_resfile_from_pdb('structure.pdb' , 'structure.resfile')	#To generate a Resfile from a .pdb file
generate_resfile_from_pose(pose , 'structure.resfile')			#To generate a Resfile from a pose
#Change NATRO to NATAA
import os , time
with open('structure.resfile') as filein:
    with open('structure.resfile2' , 'w') as fileout:
        for line in filein:
            fileout.write(line.replace('NATRO' , 'NATAA'))
time.sleep(0.1)
os.remove('structure.resfile')
os.rename('structure.resfile2' , 'structure.resfile')
#Design from Resfile
task_design = rosetta.core.pack.task.TaskFactory.create_packer_task(pose)
rosetta.core.pack.task.parse_resfile(pose, task_design , 'structure.resfile')
pack_mover = PackRotamersMover(scorefxn, task_design)
#Appy move
print(task_design)
print(pose.sequence())
pack_mover.apply(pose)
print(pose.sequence())
pose.dump_pdb('1.pdb')
######Better Design (Previous from tutorial does not work, gives errors, not sure why, tutorial seems to be missing crucial information)
task_pack = standard_packer_task(pose)
print(task_pack)
#Construct Packing Mover (requires a score function)
scorefxn = get_fa_scorefxn()
pack_mover = PackRotamersMover(scorefxn, task_pack)
#Apply Move
print(pose.sequence())
pack_mover.apply(pose)
print(pose.sequence())
pose.dump_pdb('1.pdb')

```
