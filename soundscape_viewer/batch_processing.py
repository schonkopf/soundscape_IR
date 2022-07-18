import numpy as np

class batch_processing:
  def __init__(self, folder=[], folder_id=[], file_extension='.wav', annotation_extension=None):
    if folder:
      self.collect_folder(folder, file_extension)
    elif folder_id:
      self.Gdrive, self.link, self.audioname=self.collect_Gdrive(folder_id, file_extension)
      if annotation_extension:
        self.Gdrive_selections, _, _=self.collect_Gdrive(folder_id, '.txt')
    self.Raven_selections=annotation_extension
    self.run_spectrogram=False  
    self.run_separation=False
    self.run_detection=False
    self.run_pulse_analysis=False
    self.run_lts=False
    self.run_adaptive_prewhiten=False
    self.run_load_basis=False
    
  def collect_folder(self, path, file_extension='.wav'):
    import os
    file_list=os.listdir(path)
    self.link=path
    self.cloud=0   
    self.audioname=np.array([], dtype=object)
    for filename in file_list:
        if filename.endswith(file_extension):
            self.audioname = np.append(self.audioname, filename)
    print('Identified ', len(self.audioname), 'files')
    
  def collect_Gdrive(self, folder_id, file_extension='.wav'):
    from soundscape_IR.soundscape_viewer.utility import gdrive_handle
    Gdrive=gdrive_handle(folder_id, status_print=False)
    Gdrive.list_query(file_extension=file_extension)
    self.cloud=2
    link=np.array([], dtype=object)
    audioname=np.array([], dtype=object)
    for file in Gdrive.file_list:
      link=np.append(link, file['alternateLink'])
      audioname=np.append(audioname, file['title'])
    print('Identified ', len(audioname), 'files')
    return Gdrive, link, audioname

  def params_spectrogram(self, offset_read=0, FFT_size=512, environment='wat', time_resolution=None, window_overlap=0, f_range=[], prewhiten_percent=0, padding=0, folder_combine=False, mel_comp=0, sensitivity=0):
    self.run_spectrogram = True
    self.offset_read = offset_read
    self.FFT_size = FFT_size 
    self.time_resolution = time_resolution
    self.window_overlap = window_overlap
    self.f_range = f_range
    self.prewhiten_percent = prewhiten_percent
    self.environment = environment
    self.folder_combine=folder_combine
    self.annotation_padding=padding
    self.mel_comp=mel_comp
    self.sensitivity=sensitivity

  def params_adaptive_prewhiten(self, eps=0.1, smooth=1, continuous_adaptive=True):
    self.run_adaptive_prewhiten=True
    self.eps=eps
    self.adaptive_smooth=smooth
    self.continuous_adaptive=continuous_adaptive

  def params_separation(self, model, iter=50, adaptive_alpha=0, additional_basis=0, save_basis=False, folder_id=[], path='./'):
    self.run_separation = True
    self.model = model
    self.iter = iter
    self.adaptive_alpha = adaptive_alpha
    self.additional_basis = additional_basis
    self.save_basis = save_basis
    self.save_basis_folder_id = folder_id
    self.save_basis_path = path

  def params_spectrogram_detection(self, source=0, threshold=20, smooth=0, minimum_interval=0, minimum_duration = None, maximum_duration=None, pad_size=0, folder_id=[], path='./', show_result=False):
    self.run_detection=True
    if isinstance(source, int):
      self.source = [source]
      if source==0:
        self.source = [None]
    else:
      self.source = source

    if isinstance(threshold, int) or isinstance(threshold, float):
        self.threshold = [threshold]*len(self.source)
    else:
      self.threshold = threshold
    
    if isinstance(minimum_interval, int) or isinstance(minimum_interval, float):
      self.minimum_interval = [minimum_interval]*len(self.source)
    else:
      self.minimum_interval = minimum_interval
    
    if isinstance(minimum_duration, int) or isinstance(minimum_duration, float) or isinstance(minimum_duration, type(None)):
      self.minimum_duration = [minimum_duration]*len(self.source)
    else:
      self.minimum_duration = minimum_duration

    if isinstance(maximum_duration, int) or isinstance(maximum_duration, float) or isinstance(maximum_duration, type(None)):
      self.maximum_duration = [maximum_duration]*len(self.source)
    else:
      self.maximum_duration = maximum_duration
    
    if isinstance(smooth, int) or isinstance(smooth, float):
      self.smooth = [smooth]*len(self.source)
    else:
      self.smooth = smooth

    if isinstance(pad_size, int) or isinstance(pad_size, float):
      self.padding = [pad_size]*len(self.source)
    else:
      self.padding = pad_size

    self.detection_folder_id = folder_id
    self.detection_path = path
    self.show_result = show_result
  
  def params_lts_maker(self, source=1, time_resolution=[], dateformat='yyyymmdd_HHMMSS', initial=[], year_initial=2000, filename='Separation_LTS.mat', folder_id=[], path='./'):
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
    self.save_lts_path=path

  def params_pulse_interval(self, energy_percentile=50, interval_range=[1, 1000], lts_maker=False):
    if lts_maker:
      self.run_pulse_analysis=False
    else:
      self.run_pulse_analysis=True
    self.energy_percentile=energy_percentile
    self.interval_range=interval_range
  
  def params_load_basis(self, initial=[], dateformat='yyyymmdd_HHMMSS', year_initial=0): 
    self.run_load_basis = True
    self.format_initial = initial
    self.dateformat = dateformat
    self.year_initial = year_initial

  def run(self, start=0, num_file=None):
    from soundscape_IR.soundscape_viewer import lts_maker
    from soundscape_IR.soundscape_viewer import audio_visualization
    from soundscape_IR.soundscape_viewer import spectrogram_detection
    from soundscape_IR.soundscape_viewer import pulse_interval
    from soundscape_IR.soundscape_viewer import matrix_operation
    from soundscape_IR.soundscape_viewer import source_separation
    from soundscape_IR.soundscape_viewer import save_parameters

    import datetime
    import copy
    import os
    if self.cloud==1:
      import urllib.request

    self.start = start
    if not num_file:
      num_file=len(self.audioname)
    run_list=range(self.start, self.start+num_file)

    if self.run_separation:
      model_backup = copy.deepcopy(self.model)
    
    for file in run_list:
      print('\r', end='')
      if self.run_spectrogram:
        if self.cloud==2:
          temp = self.Gdrive.file_list[file]
          temp.GetContentFile(temp['title'])
          if self.Raven_selections:
            self.Gdrive_selections.list_query(file_extension=temp['title'][:-4]+self.Raven_selections)
            temp2 = self.Gdrive_selections.file_list[0]
            temp2.GetContentFile(temp2['title'])
            selections_filename = temp2['title']
          path='.'
          print('Processing file no. '+str(file+1)+' :'+temp['title']+', in total: '+str(num_file)+' files', flush=True, end='')
        else:
          temp = self.audioname[file]
          path = self.link
          if self.Raven_selections:
            selections_filename = self.link + '/' + self.audioname[file][:-4]+self.Raven_selections
          print('Processing file no. '+str(file+1)+' :'+temp+', in total: '+str(num_file)+' files', flush=True, end='')
        
        if self.Raven_selections:
          audio = audio_visualization(self.audioname[file], path, FFT_size = self.FFT_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range, sensitivity=self.sensitivity, 
                                      environment=self.environment, plot_type=None, prewhiten_percent=self.prewhiten_percent, annotation = selections_filename, padding = self.annotation_padding, mel_comp=self.mel_comp)
        else:
          audio = audio_visualization(self.audioname[file], path, offset_read = self.offset_read, FFT_size = self.FFT_size, time_resolution=self.time_resolution, window_overlap=self.window_overlap, f_range = self.f_range, sensitivity=self.sensitivity,
                                    environment=self.environment, plot_type=None, prewhiten_percent=self.prewhiten_percent, mel_comp=self.mel_comp)
        if self.folder_combine:
          if file==self.start:
            folder_data=audio.data
          else:
            folder_data=np.vstack((folder_data, audio.data))
              
          folder_data[:,0]=np.arange(folder_data.shape[0])*(folder_data[1,0]-folder_data[0,0])
          self.spectrogram=np.array(folder_data)
          self.f=np.array(audio.f)
        
        else:
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
        if(self.save_basis):
          filename=self.audioname[file][:-4]+'.mat'
          if self.save_basis_folder_id:
            model.save_model(filename=filename, folder_id=self.save_basis_folder_id)
          else:
            model.save_model(filename=self.save_basis_path+'/'+filename)
        
        if self.run_lts:
          if file==self.start:
            lts = lts_maker(time_resolution=self.lts_time_resolution)
            lts.filename_check(self.dateformat, self.initial, self.year_initial, self.audioname[file])
            lts.f=audio.f
            lts.sf=audio.sf
            lts.link=[]
          lts.get_file_time(self.audioname[file])
          lts.compress_spectrogram(10**(model.separation[self.lts_source[0]-1][:,1:]/10), model.separation[self.lts_source-1][:,0], self.lts_time_resolution, linear_scale=True, interval_range=self.interval_range, energy_percentile=self.energy_percentile)

        if self.run_detection:
          for n in range(0, len(self.source)):
            filename=self.audioname[file][:-4]+'_S'+str(self.source[n])+'.txt'
            spectrogram_detection(model.separation[self.source[n]-1], model.f, threshold=self.threshold[n], smooth=self.smooth[n], minimum_interval=self.minimum_interval[n], minimum_duration=self.minimum_duration[n], maximum_duration=self.maximum_duration[n], pad_size=self.padding[n], filename=filename, folder_id=self.detection_folder_id, path=self.detection_path, status_print=False, show_result=self.show_result)
      if self.run_detection:
        if not self.source[0]:
          filename=self.audioname[file][:-4]+'.txt'
          spectrogram_detection(audio.data, audio.f, threshold=self.threshold[0], smooth=self.smooth[0], minimum_interval=self.minimum_interval[0], minimum_duration=self.minimum_duration[0], maximum_duration=self.maximum_duration[0], pad_size=self.padding[0], filename=filename, folder_id=self.detection_folder_id, path=self.detection_path, status_print=False, show_result=self.show_result)
      if self.run_pulse_analysis:
        if self.run_separation:
          pulse_analysis_result=pulse_interval(model.separation[self.source[0]-1], energy_percentile=self.energy_percentile, interval_range=self.interval_range, plot_type=None)
        else:
          pulse_analysis_result=pulse_interval(audio.data, energy_percentile=self.energy_percentile, interval_range=self.interval_range, plot_type=None)
        if file==self.start:
          self.PI_result=pulse_analysis_result.result[None]
          self.PI=pulse_analysis_result.PI
        else:
          self.PI_result=np.vstack((self.PI_result, pulse_analysis_result.result[None]))
      
      if self.run_load_basis:
        if self.cloud==2:
          temp = self.Gdrive.file_list[file]
          temp.GetContentFile(temp['title'])
          temp = temp['title']
          path='.'
        else:
          temp = self.audioname[file][:-4]+'.mat'
          if hasattr(self,'save_basis_path'):
            path = self.save_basis_path
          else:
            path = self.link
        print('Processing file no. '+str(file+1)+' :'+temp+', in total: '+str(num_file)+' files', flush=True, end='')

        if self.dateformat:
          if file==self.start:
            format_idx = lts_maker()
            format_idx.filename_check(dateformat=self.dateformat, initial=self.format_initial, year_initial=self.year_initial, filename=temp)
          format_idx.get_file_time(temp)
          str2ord=-366+format_idx.time_vec/24/3600
        else:
          str2ord=int(temp[len(self.format_initial):-4])

        temp_model = source_separation()
        temp_model.load_model(path+'/'+temp, model_check = False)
        if file==self.start:
          self.model = copy.deepcopy(temp_model)
          self.model.time_vec = str2ord*np.ones((temp_model.W.shape[1],))
        else:
          self.model.W = np.concatenate((self.model.W, temp_model.W), axis=1) 
          self.model.W_cluster = np.concatenate((self.model.W_cluster, temp_model.W_cluster), axis=None) 
          self.model.time_vec = np.concatenate((self.model.time_vec,str2ord*np.ones((temp_model.W.shape[1],))), axis=None) 

      if self.cloud>=1:
        os.remove(self.audioname[file])
        if self.Raven_selections:
          os.remove(temp2['title'])

    if self.run_load_basis:
      temp = np.argsort(self.model.time_vec)
      self.model.W=self.model.W[:,temp]
      self.model.W_cluster=self.model.W_cluster[temp]
      self.model.time_vec=self.model.time_vec[temp]

    if self.run_lts:
      if self.lts_folder_id:
        lts.save_lts(self.lts_filename, self.lts_folder_id, status_print=False)
      else:
        lts.save_lts(self.save_lts_path+'/'+self.lts_filename, status_print=False)
