**#Linux**
#Ubuntu
sudo apt update && sudo apt full-upgrade && sudo apt install weechat vim ffmpeg pymol gnuplot tmux openconnect python3-pip git htop dssp python3-tk -y && sudo pip3 install biopython bs4 scikit-learn numpy pandas matplotlib
##Temp command on weechat
/set buflist.look.enabled off
--------------------------------------------------
**#Convert Images**
#Convert a PDF file to a PNG image - increase r to increase quality
gs -sDEVICE=jpeg -sOutputFile=FILENAME.jpg -r1000 FILENAME.pdf
#Invert colours of image
convert -negate FILENAME.gif NEWFILENAME.gif
--------------------------------------------------
**#TMUX**
#Aziz Supercomputer Environment Setup
tmux new -s aziz -n aziz \; split-window -h -p25 \; split-window -v -p50
--------------------------------------------------
**#FFMPEG**
#Re-encode .mkv to .mp4 To Work With Samsung TV
for i in *.mkv; do ffmpeg -i "$i" -vcodec libx264 -r 25 -crf 23 -ab 384k -acodec ac3 "${i%.mkv}.mp4"; done
#Record Desktop Screen
ffmpeg -f x11grab -s 1280x800 -framerate 30 -i :0.0 -c:v libx264rgb Screen.mkv
--------------------------------------------------
**#Rosetta**
#Download
[https://www.rosettacommons.org/software/academic]
Academic_User
Xry3x4
#Compile
sudo apt install zlib1g-dev scons build-essential -y
tar -xvzf {ROSETTA}.tgz
cd {ROSETTA}/main/source
./scons.py mode=release bin
#Cannot compile on AZIZ straighforward because python -V -> Python 2.6.6 therefore to compile first load python 2.7.9 by this command. Then compile normally with ./scons.py mode=release bin
module use /app/utils/modules && module load python-2.7.9
--------------------------------------------------
**#PYROSETTA**
#Download
[http://graylab.jhu.edu/download/PyRosetta4/archive/release/]
levinthal
paradox
#Compile
tar -vjxf {PYROSETTA}.tar.bz2
cd {PYROSETTA}/setup
sudo python3 setup.py install
