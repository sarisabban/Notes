<h1>Setup Gmail Account</h1>
* Go to Google Drive
* Make a directory
* Open directory > right click > +more > search > Colaboratory > connect
* On directory again > right click > creat new Colaboratory
* Rename notebook
* Edit > Notebook Settings > python3 and GPU

Execute the following code:

`!apt update
!apt full-upgrade
!apt install -y -qq software-properties-common python-software-properties module-init-tools fuse
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt -y install -qq google-drive-ocamlfuse
from google.colab import auth
from oauth2client.client import GoogleCredentials
import getpass
auth.authenticate_user()
creds = GoogleCredentials.get_application_default()
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
!mkdir -p drive
!google-drive-ocamlfuse drive
!pip install -q keras`
