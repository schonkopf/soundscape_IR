import numpy as np

class batch_processing:  
  def __init__(self, folder = [], filename=[], folder_id = [], file_extension = '.wav', import_Raven_selections=False, filename_add=[]):
    if folder:
      self.collect_folder(folder, file_extension)
    elif filename:
      self.collect_pumilio(filename)
    elif folder_id:
      self.Gdrive, self.link, self.audioname=self.collect_Gdrive(folder_id, file_extension)
      if import_Raven_selections:
        self.Gdrive_selections, _, _=self.collect_Gdrive(folder_id, '.txt')
    self.Raven_selections=filename_add

  def collect_folder(self, path, file_extension='.wav'):
    import os
    file_list = os.listdir(path)
    self.link = path
    self.cloud = 0   
    self.audioname=np.array([], dtype=np.object)
    for filename in file_list:
        if filename.endswith(file_extension):
            self.audioname = np.append(self.audioname, filename)
    print('Identified ', len(self.audioname), 'files')
    
  def collect_pumilio(self, filename):
    file_list=pd.read_csv(filename, sep=',', header=0)
    self.link=file_list.FilePath
    self.audioname=file_list.SoundName
    self.SoundID=file_list.SoundID
    self.cloud=1
    print('Identified ', len(self.audioname), 'files')
    
  def collect_Gdrive(self, folder_id, file_extension='.wav'):
    from soundscape_IR.soundscape_viewer.utility import gdrive_handle
    Gdrive=gdrive_handle(folder_id)
    Gdrive.list_query(file_extension=file_extension)
    self.cloud=2
    link=np.array([], dtype=np.object)
    audioname=np.array([], dtype=np.object)
    #self.Gdrive=Gdrive
    for file in Gdrive.file_list:
      link=np.append(link, file['alternateLink'])
      audioname=np.append(audioname, file['title'])
    print('Identified ', len(audioname), 'files')
    return Gdrive, link, audioname

  def params_spectrogram(self, fft_size=512, environment='wav', time_resolution=None, window_overlap=0, f_range=[], prewhiten_percent=0, padding=0, folder_combine=False, mel_comp=0):
    self.fft_size = fft_size 
    self.time_resolution = time_resolution
    self.window_overlap = window_overlap
    self.f_range = f_range
    self.prewhiten_percent = prewhiten_percent
    self.environment = environment
    self.run_separation=False
    self.run_detection=False
    self.run_pulse_analysis=False
    self.folder_combine=folder_combine
    self.annotation_padding=padding
    self.mel_comp=mel_comp

  def params_separation(self, model, iter=50, adaptive_alpha=0, additional_basis=0):
    self.model = model
    self.iter = iter
    self.adaptive_alpha = adaptive_alpha
    self.additional_basis = additional_basis
    self.run_separation = True
  
  def params_spectrogram_detection(self, source=1, threshold=20, smooth=3, frequency_cut=20, minimum_interval=0, padding=0, folder_id=[]):
    if isinstance(source, int):
      self.source = [source]
      if source==0:
        self.source = None
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

    if isinstance(minimum_interval, int) or isinstance(minimum_interval, float):
      self.minimum_interval = [minimum_interval]*len(self.source)
    else:
      self.minimum_interval = minimum_interval
    
    self.smooth = smooth
    self.padding = padding
    self.folder_id = folder_id
    self.run_detection=True

  def params_pulse_interval(self, energy_percentile=50, interval_range=[1, 1000]):
    self.energy_percentile=energy_percentile
    self.interval_range=interval_range
    self.run_pulse_analysis=True
  
  def run(self):
    from soundscape_IR.soundscape_viewer.utility import audio_visualization
    from soundscape_IR.soundscape_viewer.utility import spectrogram_detection
    from soundscape_IR.soundscape_viewer.utility import pulse_interval

    import copy
    import os
    if self.cloud==1:
      import urllib.request

    if self.run_separation:
      model_backup = copy.deepcopy(self.model)

    for file in range(0,len(self.audioname)):
      if self.cloud==1:
        urllib.request.urlretrieve(self.link[file], self.audioname[file])
        path='.'
      elif self.cloud==2:
        temp = self.Gdrive.file_list[file]
        temp.GetContentFile(temp['title'])
        if self.Raven_selections:
          self.Gdrive_selections.list_query(file_extension=temp['title'][:-4]+self.Raven_selections)
          temp2 = self.Gdrive_selections.file_list[0]
          temp2.GetContentFile(temp2['title'])
        path='.'
      else:
        path=self.link

      print('Processing file no. '+str(file+1)+' :'+temp['title']+', in total: '+str(len(self.audioname))+' files')
      if self.Raven_selections:
        audio = audio_visualization(self.audioname[file], path, FFT_size = self.fft_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range, 
                                    environment=self.environment, plot_type=None, prewhiten_percent=self.prewhiten_percent, annotation = temp2['title'], padding = self.annotation_padding, mel_comp=self.mel_comp)
        if self.folder_combine:
          if file==0:
            folder_data=audio.data
            time_notation=np.add((file+1)*np.ones((audio.data.shape[0],1),dtype = int)*10000, audio.time_notation)
          else:
            folder_data=np.vstack((folder_data, audio.data))
            time_notation = np.vstack((time_notation, np.add((file+1)*np.ones((audio.data.shape[0],1),dtype = int)*10000, audio.time_notation)))
            
          folder_data[:,0]=np.arange(folder_data.shape[0])*(folder_data[1,0]-folder_data[0,0])
          self.spectrogram=np.array(folder_data)
          self.f=np.array(audio.f)
          self.time_notation=time_notation
      else:
        audio = audio_visualization(self.audioname[file], path, FFT_size = self.fft_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range,
                                  environment=self.environment, plot_type=None, prewhiten_percent=self.prewhiten_percent)
      
      if self.run_separation:
        model = copy.deepcopy(model_backup)
        model.supervised_separation(audio.data, audio.f, iter = self.iter, adaptive_alpha = self.adaptive_alpha, additional_basis = self.additional_basis)
        if self.run_detection:
          for n in range(0, len(self.source)):
            filename=self.audioname[file][:-4]+'_S'+str(self.source[n])+'.txt'
            spectrogram_detection(model.separation[self.source[n]-1], model.f, threshold=self.threshold[n], smooth=self.smooth, frequency_cut=self.frequency_cut[n], minimum_interval=self.minimum_interval[n], pad_size=self.padding, filename=filename, folder_id = self.folder_id)
            
      if self.run_detection:
        if not self.source:
          filename=self.audioname[file][:-4]+'.txt'
          spectrogram_detection(audio.data, audio.f, threshold=self.threshold[n], smooth=self.smooth, frequency_cut=self.frequency_cut[n], minimum_interval=self.minimum_interval[n], pad_size=self.padding, filename=filename, folder_id = self.folder_id)

      if self.run_pulse_analysis:
        if self.run_separation:
          pulse_analysis_result=pulse_interval(model.separation[self.source-1], self.energy_percentile, self.interval_range, plot_type= None)
        else:
          pulse_analysis_result=pulse_interval(audio.data, self.energy_percentile, self.interval_range, plot_type= None)
        if file==0:
          self.result=pulse_analysis_result.result[None]
          self.PI=pulse_analysis_result.PI
        else:
          self.result=np.vstack((self.result, pulse_analysis_result.result[None]))

      if self.cloud>=1:
        os.remove(self.audioname[file])
        if self.Raven_selections:
          os.remove(temp2['title'])
