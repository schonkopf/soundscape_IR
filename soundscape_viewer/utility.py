import numpy as np

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
    self.gauth=gauth
    self.Gdrive = GoogleDrive(gauth)
    print('Now establishing link to Google drive.')
  
  def renew_token(self):
    if self.gauth.access_token_expired:
        # Refresh them if expired
        print('Drive token expired, refreshing')
        self.gauth.Refresh()
        self.Gdrive = GoogleDrive(self.gauth)

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

class matrix_operation:
  def __init__(self, header=[]):
    self.header=header
    
  def gap_fill(self, time_vec, data):
     # fill the gaps in a time series
     temp = np.argsort(time_vec)
     time_vec=time_vec[temp]
     if data.ndim>1:
         output=data[temp,:]
     else:
         output=data[temp]
     resolution=np.round((time_vec[1]-time_vec[0])*24*3600)
     n_time_vec=np.arange(np.floor(np.min(time_vec))*24*3600, 
                             np.ceil(np.max(time_vec))*24*3600,resolution)/24/3600

     if data.ndim>1:
         save_result=np.zeros((n_time_vec.size, data.shape[1]+1))
     else:
         save_result=np.zeros((n_time_vec.size, 2))

     save_result[:,0]=n_time_vec-693960
     segment_list=np.round(np.diff(time_vec*24*3600)/resolution)
     split_point=np.vstack((np.concatenate(([0],np.where(segment_list!=1)[0]+1)),
                               np.concatenate((np.where(segment_list!=1)[0],[time_vec.size-1]))))

     for run in np.arange(split_point.shape[1]):
       i=np.argmin(np.abs(n_time_vec-time_vec[split_point[0,run]]))
       if data.ndim>1:
             save_result[np.arange(i,i+np.diff(split_point[:,run])+1),1:]=output[np.arange(split_point[0,run], split_point[1,run]+1),:]
       else:
             save_result[np.arange(i,i+np.diff(split_point[:,run])+1),1]=output[np.arange(split_point[0,run], split_point[1,run]+1)]
     return save_result
