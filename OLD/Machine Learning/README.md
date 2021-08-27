<h1>GPU Setup For Deep Learning</h1>
<h2>Use a GPU with Google in Colaboratory (Colab)</h2>
<h3>Setup Gmail Account</h3>

* Go to Google Drive

* Right Click > +More > Search > Colaboratory > Connect

* Right Click > Creat New Colaboratory

* Rename notebook

* Edit > Notebook Settings > python3 and GPU


<h3>Get a file from Google</h3>
* Run the following python script

`!pip install -U -q PyDrive`

`from pydrive.auth import GoogleAuth`

`from pydrive.drive import GoogleDrive`

`from google.colab import auth`

`from oauth2client.client import GoogleCredentials`

`auth.authenticate_user()`

`gauth = GoogleAuth()`

`gauth.credentials = GoogleCredentials.get_application_default()`

`drive = GoogleDrive(gauth)`

`uploaded = drive.CreateFile({'title': 'SS.h5'})`

`uploaded.SetContentFile('SS.h5')`

`uploaded.Upload()`

Find the file in Google Drive


<h2>Setup on server with a GPU</h2>
# (Ubuntu 16.04 GPU Setu)[https://yangcha.github.io/CUDA90/]

`wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb`

`wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb`

`wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb`

`wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb`

`wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb`

`sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb`

`sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb`

`sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb`

`sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb`

`sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb`

`sudo apt-get update`

`sudo apt-get install cuda=9.0.176-1`

`sudo apt-get install libcudnn7-dev`

`sudo apt-get install libnccl-dev`


# Arch Linux:

`sudo pacman -S cuda`

Check to see if Keras on TensorFlow is using the GPU (last line should give the GPU brand)
Write the followinglines into a python script and run it:

`from keras import backend as K`

`K.tensorflow_backend._get_available_gpus()`


<h2>Using TensorBoard</h2>
Run the following command from the terminal and open the given URL

`tensorboard --logdir=./`
