<h1>Use a GPU with Google in Colaboratory (Colab)</h1>

<h2>Setup Gmail Account</h2>

* Go to Google Drive

* Right Click > +More > Search > Colaboratory > Connect

* Right Click > Creat New Colaboratory

* Rename notebook

* Edit > Notebook Settings > python3 and GPU


<h2>Get a file from Google</h2>
* Run the following python script

`!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials

auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)

uploaded = drive.CreateFile({'title': 'SS.h5'})

uploaded.SetContentFile('SS.h5')

uploaded.Upload()`

Find the file in Google Drive
