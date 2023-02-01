<<COMMENT
# Install OSPREY:
# ===============
mkdir software
cd software
echo 'Downloading openjdk-17.0.2'
wget -q https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
echo 'Extracting openjdk-17.0.2'
tar -xf openjdk-17.0.2_linux-x64_bin.tar.gz
rm openjdk-17.0.2_linux-x64_bin.tar.gz
touch .bash_profile
echo 'export JAVA_HOME=$HOME/software/jdk-17.0.2' >> $HOME/.bash_profile
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> $HOME/.bash_profile
echo '' >> $HOME/.bash_profile
source $HOME/.bash_profile
echo ''
echo $JAVA_HOME
java -version
echo ''
echo 'Downloading Miniconda'
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
	# Enter
	# yes
	# $HOME/software/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
echo 'export PATH="$HOME/software/miniconda3/bin:$PATH"' >> $HOME/.bash_profile
echo '' >> $HOME/.bash_profile
source $HOME/.bash_profile
conda config --set auto_activate_base false
source miniconda3/etc/profile.d/conda.sh # Need to run everytime to activate conda after reboot
conda create --name AmberTools22
conda activate AmberTools22
conda install -c conda-forge ambertools=22 compilers
	# Follow instructions
antechamber -h
wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
tar -xf julia-1.8.5-linux-x86_64.tar.gz
rm julia-1.8.5-linux-x86_64.tar.gz
echo 'export JULIA_HOME=$HOME/software/julia-1.8.5' >> $HOME/.bash_profile
echo 'export PATH=$PATH:$JULIA_HOME/bin' >> $HOME/.bash_profile
echo '' >> $HOME/.bash_profile
source $HOME/.bash_profile
julia --version
wget -q https://github.com/donaldlab/OSPREY3/releases/download/3.3-resistor/osprey-3.3.tar
tar --file osprey-3.3.tar --extract
rm osprey-3.3.tar
echo 'export OSPREY_HOME=$HOME/software/osprey-3.3' >> $HOME/.bash_profile
echo 'export PATH=$PATH:$OSPREY_HOME/bin' >> $HOME/.bash_profile
source $HOME/.bash_profile
osprey --help
COMMENT

# Install OSPREY Python:
# ======================
python3 -m venv myenv
source myenv/bin/activate
git clone https://github.com/donaldlab/OSPREY3.git
cd OSPREY3/
sed -i s/'"--user", "--editable",'/'"--editable",'/ ./buildSrc/src/main/kotlin/osprey/python.kt
./gradlew assemble
./gradlew pythonDevelop
