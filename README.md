```
**#COMMANDS**
===============
**#Ubuntu Environment Setup**
wget -O - https://raw.githubusercontent.com/sarisabban/Notes/master/Scripts/MintEnv_22.sh | bash
------------------------------
**#Live USB**
sudo fdisk -l
sudo dd bs=4M if=FILENAME.iso of=/dev/sdb status=progress oflag=sync iflag=fullblock
------------------------------
**#TMUX**
tmux new -s work -n work \; split-window -d -t 0 -v -l "30%" weechat \; split-window -h -l "47%" htop \; split-window -v -l "68%" vlc -I ncurses --no-video https://www.youtube.com/watch?v=jfKfPfyJRdk \; split-window -v -l "68%"
tmux new -s work -n work \; split-window -d -t 0 -v -l "30%" weechat \; split-window -h -l "47%" htop \; split-window -v -l "66%"
------------------------------
**#SSH**
ssh-keygen -t rsa -b 4096 -C USERNAME
copy/paste public key to ~/.ssh/authorized_keys
chmod 600 KEY.prv
------------------------------
**#VIM**
'[,']:w !nc termbin.com 9999
------------------------------
**#My Public IP Address**
curl ifconfig.me
------------------------------
**#FFMPEG**
for i in *.mkv; do ffmpeg -i "$i" -vcodec libx264 -r 25 -crf 23 -ab 384k -acodec ac3 "${i%.mkv}.mp4"; done
ffmpeg -i in.mp4 -c:v libx264 -crf 18 -preset veryslow -c:a copy out.mp4 # Reduce size
------------------------------
**#Convert PDF to IMAGE**
pdfimages FILENAME.pdf x
convert x-000.ppm FILENAME.jpg
------------------------------
**#Compress PDF**
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH -sOutputFile=output.pdf input.pdf
------------------------------
**#Conda**
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
conda create -n myenv
conda activate myenv
conda install ...
source deactivate myenv
```
