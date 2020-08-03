import numpy as np

class batch_processing:  
  def __init__(self, folder = [], filename=[], folder_id = [], file_extension = '.wav'):
    if folder:
      self.collect_folder(folder)
    elif filename:
      self.collect_pumilio(filename)
    elif folder_id:
      self.collect_Gdrive(folder_id)

  def collect_folder(self, path):
    import os
    file_list = os.listdir(path)
    self.link = path
    self.cloud = 0   
    self.audioname=np.array([], dtype=np.object)
    for filename in file_list:
        if filename.endswith(".wav"):
            self.audioname = np.append(self.audioname, filename)
    print('Identified ', len(self.audioname), 'files')
    
  def collect_pumilio(self, filename):
    file_list=pd.read_csv(filename, sep=',', header=0)
    self.link=file_list.FilePath
    self.audioname=file_list.SoundName
    self.SoundID=file_list.SoundID
    self.cloud=1
    print('Identified ', len(self.audioname), 'files')
    
  def collect_Gdrive(self, folder_id):
    from .utility import gdrive_handle
    Gdrive=gdrive_handle(folder_id)
    Gdrive.list_query(file_extension='.wav')
    self.cloud=2
    self.link=np.array([], dtype=np.object)
    self.audioname=np.array([], dtype=np.object)
    self.Gdrive=Gdrive
    for file in Gdrive.file_list:
      self.link=np.append(self.link, file['alternateLink'])
      self.audioname=np.append(self.audioname, file['title'])
    print('Identified ', len(self.audioname), 'files')

  def params_spectrogram(self, fft_size = 512, environment='wav', time_resolution=None, window_overlap=0, f_range = [], prewhiten_percent=0, plot_type= None):
    self.fft_size = fft_size 
    self.time_resolution = time_resolution
    self.window_overlap = window_overlap
    self.f_range = f_range
    self.prewhiten_percent = prewhiten_percent
    self.plot_type = plot_type
    self.environment = environment

  def params_separation(self, iter = 50, adaptive_alpha = 0, additional_basis = 0):
    #self.model = model
    self.iter = iter
    self.adaptive_alpha = adaptive_alpha
    self.additional_basis = additional_basis
  
  def params_spetrogram_detection(self, source = 1, threshold = 20, smooth=3, frequency_cut=20, padding=0, folder_id = []):
    if isinstance(source, int):
      self.source = [source]
    else:
      self.source = source

    if isinstance(threshold, int) or isinstance(threshold, float):
        self.threshold = [threshold]*len(self.source)
    else:
      self.threshold = threshold
    
    if isinstance(frequency_cut, int) or isinstance(frequency_cut, float):
      self.frequency_cut = [frequency_cut]*len(self.source)
    else:
      self.frequency_cut = frequency_cut
    
    self.smooth = smooth
    self.padding = padding
    self.folder_id = folder_id
  
  def supervised_separation(self, model):
    from .utility import audio_visualization
    from .utility import spectrogram_detection
    import copy
    import os
    if self.cloud==1:
      import urllib.request

    model_backup = copy.deepcopy(model)

    for file in range(0,len(self.audioname)):
      if self.cloud==1:
        urllib.request.urlretrieve(self.link[file], self.audioname[file])
        path='.'
      elif self.cloud==2:
        #title = self.Gdrive.filelist[i]['title']
        temp = self.Gdrive.file_list[file]
        temp.GetContentFile(temp['title'])
        path='.'
      else:
        path=self.link

      print('processing '+str(file+1)+'/'+str(len(self.audioname))+' file...')
      audio = audio_visualization(self.audioname[file], path, FFT_size = self.fft_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range,
                                  environment=self.environment, plot_type=self.plot_type, prewhiten_percent=self.prewhiten_percent)
      
      model = copy.deepcopy(model_backup)
      model.supervised_separation(audio.data, audio.f, iter = self.iter, adaptive_alpha = self.adaptive_alpha, additional_basis = self.additional_basis)
      for n in range(0, len(self.source)):
        filename=self.audioname[file][:-4]+'_S'+str(self.source[n])+'.txt'
        spectrogram_detection(model.separation[self.source[n]-1], model.f, threshold=self.threshold[n], smooth=self.smooth, frequency_cut=self.frequency_cut[n], pad_size=self.padding, filename=filename, folder_id = self.folder_id)
      
      if self.cloud>=1:
        os.remove(self.audioname[file])
        
  def pulse_interval_analysis(self, energy_percentile, interval_range, plot_type= None):
    from .utility import audio_visualization
    from .utility import pulse_interval
    
    import copy
    import os
    if self.cloud==1:
      import urllib.request

    for file in range(0,len(self.audioname)):
      if self.cloud==1:
        urllib.request.urlretrieve(self.link[file], self.audioname[file])
        path='.'
      elif self.cloud==2:
        #title = self.Gdrive.filelist[i]['title']
        temp = self.Gdrive.file_list[file]
        temp.GetContentFile(temp['title'])
        path='.'
      else:
        path=self.link

      print('processing '+str(file+1)+'/'+str(len(self.audioname))+' file...')
      audio = audio_visualization(self.audioname[file], path, FFT_size = self.fft_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range,
                                  environment=self.environment, plot_type=self.plot_type, prewhiten_percent=self.prewhiten_percent)
      pulse_analysis_result=pulse_interval(audio, energy_percentile, interval_range, plot_type=None)
      if file==0:
        result=pulse_analysis_result.result[None]
      else:
        result=np.vstack((result, pulse_analysis_result.result[None]))

      if self.cloud>=1:
        os.remove(self.audioname[file])
      
      self.energy_percentile=energy_percentile
      self.interval_range=interval_range
      self.result=result
      self.PI=pulse_analysis_result.PI
