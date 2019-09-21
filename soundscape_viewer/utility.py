class gdrive_handle:
  def __init__(self, folder_id):
    get_ipython().system('pip install -U -q PyDrive')
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    
    self.folder_id=folder_id
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    #
    gauth.LocalWebserverAuth()
    gauth.LoadCredentialsFile(gauth)
    #
    self.Gdrive = GoogleDrive(gauth)
    print('Now establishing link to Google drive.')
    
  def upload(self, filename):
    upload_ = self.Gdrive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": self.folder_id}], 'title': filename})
    upload_.SetContentFile(filename)
    upload_.Upload()
    print('Successifully upload to Google drive')
    
  def list_query(self, file_extension):
    location_cmd="title contains '"+file_extension+"' and '"+self.folder_id+"' in parents and trashed=false"
    self.file_list = self.Gdrive.ListFile({'q': location_cmd}).GetList()
    
class save_parameters:
  def __init__(self):
    self.platform='python'
    
  def pcnmf(self, f, W, W_cluster, source_num, feature_length, sparseness, basis_num):
    self.f=f
    self.W=W
    self.W_cluster=W_cluster
    self.k=source_num
    self.time_frame=feature_length
    self.sparseness=sparseness
    self.basis_num=basis_num

  def LTS_Result(self, LTS_median, LTS_mean, f, link):
    self.LTS_median = LTS_median
    self.LTS_mean = LTS_mean
    self.f = f
    self.link = link

  def LTS_Parameters(self, FFT_size, overlap, sensitivity, sampling_freq, channel):
    self.FFT_size=FFT_size
    self.overlap=overlap 
    self.sensitivity=sensitivity 
    self.sampling_freq=sampling_freq 
    self.channel=channel
