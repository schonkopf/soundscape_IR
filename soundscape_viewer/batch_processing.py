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

  def params_spectrogram(self, offset_read=0, fft_size=512, environment='wat', time_resolution=None, window_overlap=0, f_range=[], prewhiten_percent=0, padding=0, folder_combine=False, mel_comp=0, sensitivity=0):
    self.offset_read = offset_read
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
    self.sensitivity=sensitivity
    self.run_lts=False
    self.run_adaptive_prewhiten=False

  def params_adaptive_prewhiten(self, eps=0.1, smooth=1, continuous_adaptive=True):
    self.eps=eps
    self.adaptive_smooth=smooth
    self.run_adaptive_prewhiten=True
    self.continuous_adaptive=continuous_adaptive

  def params_separation(self, model, iter=50, adaptive_alpha=0, additional_basis=0):
    self.model = model
    self.iter = iter
    self.adaptive_alpha = adaptive_alpha
    self.additional_basis = additional_basis
    self.run_separation = True
  
  def params_spectrogram_detection(self, source=1, threshold=20, smooth=3, frequency_cut=20, frequency_count=0, minimum_interval=0, padding=0, folder_id=[],path='./'):
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

    if isinstance(frequency_count, int) or isinstance(frequency_count, float):
      self.frequency_count = [frequency_count]*len(self.source)
    else:
      self.frequency_count = frequency_count

    if isinstance(minimum_interval, int) or isinstance(minimum_interval, float):
      self.minimum_interval = [minimum_interval]*len(self.source)
    else:
      self.minimum_interval = minimum_interval
    
    self.smooth = smooth
    self.padding = padding
    self.folder_id = folder_id
    self.path = path
    self.run_detection=True
  
  def params_lts_maker(self, source=1, time_resolution=[], dateformat='yyyymmdd_HHMMSS', initial=[], year_initial=2000, filename='Separation_LTS.mat', folder_id=[]):
    self.run_lts=True
    self.lts_time_resolution=time_resolution
    self.dateformat=dateformat
    self.initial=initial
    self.year_initial=year_initial
    self.lts_source=source
    self.lts_filename=filename
    self.lts_folder_id=folder_id
    self.interval_range=[]
    self.energy_percentile=0

  def params_pulse_interval(self, energy_percentile=50, interval_range=[1, 1000], LTS_combine=False):
    self.energy_percentile=energy_percentile
    self.interval_range=interval_range
    if LTS_combine:
      self.run_pulse_analysis=False
    else:
      self.run_pulse_analysis=True
  
  def run(self, start=0, num_file=None):
    from soundscape_IR.soundscape_viewer.lts_maker import lts_maker
    from soundscape_IR.soundscape_viewer.utility import audio_visualization
    from soundscape_IR.soundscape_viewer.utility import spectrogram_detection
    from soundscape_IR.soundscape_viewer.utility import pulse_interval
    from soundscape_IR.soundscape_viewer.utility import matrix_operation

    import copy
    import os
    if self.cloud==1:
      import urllib.request

    if self.run_separation:
      model_backup = copy.deepcopy(self.model)

    self.start = start
    if num_file:
      run_list = range(self.start, self.start+num_file)
    else:
      run_list = range(self.start, len(self.audioname))

    for file in run_list:
      print('\r', end='')
      if self.cloud==1:
        urllib.request.urlretrieve(self.link[file], self.audioname[file])
        path='.'
        print('Processing file no. '+str(file)+' :'+temp['title']+', in total: '+str(len(self.audioname))+' files', flush=True, end='')
      elif self.cloud==2:
        temp = self.Gdrive.file_list[file]
        temp.GetContentFile(temp['title'])
        if self.Raven_selections:
          self.Gdrive_selections.list_query(file_extension=temp['title'][:-4]+self.Raven_selections)
          temp2 = self.Gdrive_selections.file_list[0]
          temp2.GetContentFile(temp2['title'])
        path='.'
        print('Processing file no. '+str(file)+' :'+temp['title']+', in total: '+str(len(self.audioname))+' files', flush=True, end='')
      else:
        temp = self.audioname[file]
        path = self.link
        print('Processing file no. '+str(file)+' :'+temp+', in total: '+str(len(self.audioname))+' files', flush=True, end='')

      
      if self.Raven_selections:
        audio = audio_visualization(self.audioname[file], path, FFT_size = self.fft_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range, sensitivity=self.sensitivity, 
                                    environment=self.environment, plot_type=None, prewhiten_percent=self.prewhiten_percent, annotation = temp2['title'], padding = self.annotation_padding, mel_comp=self.mel_comp)
        if self.folder_combine:
          if file==0:
            folder_data=audio.data
            time_notation=np.add((file)*np.ones((audio.data.shape[0],1),dtype = int)*10000, audio.time_notation)
          else:
            folder_data=np.vstack((folder_data, audio.data))
            time_notation = np.vstack((time_notation, np.add((file)*np.ones((audio.data.shape[0],1),dtype = int)*10000, audio.time_notation)))
            
          folder_data[:,0]=np.arange(folder_data.shape[0])*(folder_data[1,0]-folder_data[0,0])
          self.spectrogram=np.array(folder_data)
          self.f=np.array(audio.f)
          self.time_notation=time_notation
      else:
        audio = audio_visualization(self.audioname[file], path, offset_read = self.offset_read, FFT_size = self.fft_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range, sensitivity=self.sensitivity,
                                  environment=self.environment, plot_type=None, prewhiten_percent=self.prewhiten_percent, mel_comp=self.mel_comp)
        if self.run_adaptive_prewhiten:
          if file==self.start:
            audio.data[:,1:], ambient=matrix_operation.adaptive_prewhiten(audio.data[:,1:], prewhiten_percent=50, axis=0, eps=self.eps, smooth=self.adaptive_smooth)
          else:
            if not self.continuous_adaptive:
              ambient=None
            audio.data[:,1:], ambient=matrix_operation.adaptive_prewhiten(audio.data[:,1:], axis=0, noise_init=ambient, eps=self.eps, smooth=self.adaptive_smooth)
          audio.data[np.isnan(audio.data)]=0
      
      if self.run_separation:
        model = copy.deepcopy(model_backup)
        model.prediction(audio.data, audio.f, iter = self.iter, adaptive_alpha = self.adaptive_alpha, additional_basis = self.additional_basis)
        if self.run_lts:
          if file==self.start:
            lts = lts_maker(time_resolution=self.lts_time_resolution)
            lts.filename_check(self.dateformat, self.initial, self.year_initial, self.audioname[file])
            lts.f=audio.f
            lts.sf=audio.sf
            lts.link=[]
          lts.get_file_time(self.audioname[file])
          lts.compress_spectrogram(10**(model.separation[self.lts_source-1][:,1:]/10), model.separation[self.lts_source-1][:,0], self.lts_time_resolution, linear_scale=True, interval_range=self.interval_range, energy_percentile=self.energy_percentile)

        if self.run_detection:
          for n in range(0, len(self.source)):
            filename=self.audioname[file][:-4]+'_S'+str(self.source[n])+'.txt'
            spectrogram_detection(model.separation[self.source[n]-1], model.f, threshold=self.threshold[n], smooth=self.smooth, frequency_cut=self.frequency_cut[n], frequency_count=self.frequency_count[n], minimum_interval=self.minimum_interval[n], pad_size=self.padding, filename=filename, folder_id = self.folder_id, status_print=False,path=self.path)
            
      if self.run_detection:
        if not self.source:
          filename=self.audioname[file][:-4]+'.txt'
          spectrogram_detection(audio.data, audio.f, threshold=self.threshold[n], smooth=self.smooth, frequency_cut=self.frequency_cut[n], frequency_count=self.frequency_count[n], minimum_interval=self.minimum_interval[n], pad_size=self.padding, filename=filename, folder_id = self.folder_id, status_print=False,path=self.path)

      if self.run_pulse_analysis:
        if self.run_separation:
          pulse_analysis_result=pulse_interval(model.separation[self.source-1], energy_percentile=self.energy_percentile, interval_range=self.interval_range, plot_type=None)
        else:
          pulse_analysis_result=pulse_interval(audio.data, energy_percentile=self.energy_percentile, interval_range=self.interval_range, plot_type=None)
        if file==0:
          self.result=pulse_analysis_result.result[None]
          self.PI=pulse_analysis_result.PI
        else:
          self.result=np.vstack((self.result, pulse_analysis_result.result[None]))

      if self.cloud>=1:
        os.remove(self.audioname[file])
        if self.Raven_selections:
          os.remove(temp2['title'])

    if self.run_lts:
      lts.save_lts(self.lts_filename, self.lts_folder_id, status_print=False)
