"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""
import numpy as np
import pandas as pd
import os
import datetime
from scipy.io import savemat
from .utility import save_parameters
from .utility import gdrive_handle
from .utility import pulse_interval

class lts_maker:
  """
  Long-term spectrogram maker
  1. Initiate LTS_maker and assign parameters
  2. Load all the wav files at a specific folder (local, access from Pumilio, or by Google drive)
  3. Check whether the retrieval of recording date/time is correct
  4. Run the LTS-maker
  
  Examples
  --------
  Load local data
  >>> LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=512, 
                   window_overlap=0, initial_skip=0)
  >>> LTS_run.collect_folder(path=r"D:\Data")
  >>> LTS_run.filename_check(dateformat='yymmddHHMMSS',initial='1207984160.',year_initial=2000)
  >>> LTS_run.run(save_filename='LTS.mat')
  
  Load data on a folder in Google drive
  >>> LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=512, 
                   window_overlap=0, initial_skip=0)
  >>> LTS_run.collect_Gdrive(folder_id='1-JXEVJobPlfEd1CIET8kL_ziYsWB5pgP')
  >>> LTS_run.filename_check(dateformat='yymmddHHMMSS',initial='1207984160.',year_initial=2000)
  >>> LTS_run.run(save_filename='LTS.mat')
  
  Load data from Pumilio (need to access csv by R at first)
  >>> LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=512, 
                        window_overlap=0, initial_skip=0)
  >>> LTS_run.collect_pumilio('list.csv')
  >>> LTS_run.filename_check(dateformat='yyyymmddHHMMSS',initial='TW_LHC01_',year_initial=0)
  >>> LTS_run.run(save_filename='LTS.mat')
  
  Parameters
  ----------
  sensitivity : float
      Sensitivity of the recording system (dB re 1 V/Pa).
      Default: 0 (return relative level)
      
  channel : int (>0)
      Recording channel. 
      In stereo recordings, 1 refer to the left channel, and 2 refer to the right channel.
      Default: 1
      
  environment : 'air' | 'wat'
      Recording environmental, either in the air or underwater. 
      This will affect the reference pressure.
      Default: 'wat'
  
  FFT_size : int (>0)
      Sample size for Fast Fourier transform. 
      Frequency resolution = sampling rate/FFT_size.
      Default: 512
      
  window_overlap : float (0-<1)
      Overlapping ratio among windows.
      Default: 0
  
  initial_skip : float (>0)
      Initial duration (seconds) to skip the audio analysis. 
      This option is for audio data contain calibration tone in the beginning.
      Default: 0
  
  """
  def __init__(self, sensitivity=0, channel=1, environment='wat', FFT_size=512, window_overlap=0, initial_skip=0, time_resolution=None):
    if environment=='wat':
        P_ref=1
    elif environment=='air':
        P_ref=20
        
    self.sen = sensitivity
    self.channel = channel
    self.pref = P_ref 
    self.FFT_size = FFT_size
    self.overlap=int(window_overlap*FFT_size)
    self.skip_duration=initial_skip
    self.time_resolution=time_resolution
    self.Result_median=np.array([])
    self.Result_mean=np.array([])
    self.IPI_result=np.array([])
 
  def filename_check(self, dateformat='yyyymmdd_HHMMSS', initial=[], year_initial=2000, filename=[]):
    """
    Time stamps on the file name
    For example: TW_LHC01_150102-001530.wav
    >>> initial='TW_LHC01_'
    >>> dateformat='yymmdd-HHMMSS'
    >>> year_initial=2000

    # TW_LHC_20150102-001530.wav
    >>> initial='TW_LHC_'
    >>> dateformat='yyyymmdd-HHMMSS'
    >>> year_initial=0
    """
    if not filename:
      filename=self.audioname[0]
    idx=len(initial)
    
    if dateformat.find('yyyy')==-1:
        self.yy_pos = np.array([dateformat.find('yy')+idx, dateformat.find('yy')+2+idx, year_initial])
    else:
        self.yy_pos = np.array([dateformat.find('yyyy')+idx, dateformat.find('yyyy')+4+idx, 0])
    self.year_initial = year_initial

    self.mm_pos = np.array([dateformat.find('mm')+idx, dateformat.find('mm')+2+idx])
    self.dd_pos = np.array([dateformat.find('dd')+idx, dateformat.find('dd')+2+idx])
    self.HH_pos = np.array([dateformat.find('HH')+idx, dateformat.find('HH')+2+idx])
    self.MM_pos = np.array([dateformat.find('MM')+idx, dateformat.find('MM')+2+idx])
    self.SS_pos = np.array([dateformat.find('SS')+idx, dateformat.find('SS')+2+idx])
      
    print('Example: ', filename)
    print('Please review whether the date and time are retrieved correctly.')
    print('Year:', self.yy_pos[2]+int(filename[self.yy_pos[0]:self.yy_pos[1]]))
    print('Month:', filename[self.mm_pos[0]:self.mm_pos[1]])
    print('Day:', filename[self.dd_pos[0]:self.dd_pos[1]])
    print('Hour:', filename[self.HH_pos[0]:self.HH_pos[1]])
    print('Minute:', filename[self.MM_pos[0]:self.MM_pos[1]])
    print('Second:', filename[self.SS_pos[0]:self.SS_pos[1]]) 
  
  def collect_folder(self, path):
    file_list = os.listdir(path)
    self.link = path
    self.cloud = 0   
    n = 0
    self.audioname=np.array([], dtype=np.object)
    for filename in file_list:
        if filename.endswith(".wav"):
            self.audioname = np.append(self.audioname, filename)
            n = n+1
    print('Identified ', len(self.audioname), 'files')
    
  def collect_pumilio(self, filename):
    """
    Please execute the following codes in R and then upload the csv
    >>> library(pumilioR)
    >>> list<-getSounds(pumilio_URL="http://soundscape.twgrid.org/", SiteID=site_id, type="site")
    >>> write.csv(apply(subset(a, select=c("SoundID","SoundName", "FilePath")),2,as.character),"list.csv")
    """
    file_list=pd.read_csv(filename, sep=',', header=0)
    self.link=file_list.FilePath
    self.audioname=file_list.SoundName
    self.SoundID=file_list.SoundID
    self.cloud=1
    print('Identified ', len(self.audioname), 'files')
    
  def collect_Gdrive(self, folder_id, file_extension='.wav', subfolder=False):
    Gdrive=gdrive_handle(folder_id)
    Gdrive.list_query(file_extension=file_extension, subfolder=subfolder)
    self.cloud=2
    self.link=np.array([], dtype=np.object)
    self.audioname=np.array([], dtype=np.object)
    self.Gdrive=Gdrive
    n=0
    for file in Gdrive.file_list:
      self.link=np.append(self.link, file['alternateLink'])
      self.audioname=np.append(self.audioname, file['title'])
      n=n+1
    print('Identified ', len(self.audioname), 'files')
    
  def get_file_time(self, infilename):
    yy = int(infilename[self.yy_pos[0]:self.yy_pos[1]])
    if self.year_initial>0:
      yy = yy+self.yy_pos[2]
    mm = int(infilename[self.mm_pos[0]:self.mm_pos[1]])
    dd = int(infilename[self.dd_pos[0]:self.dd_pos[1]])
    HH = int(infilename[self.HH_pos[0]:self.HH_pos[1]])
    MM = int(infilename[self.MM_pos[0]:self.MM_pos[1]])
    SS = int(infilename[self.SS_pos[0]:self.SS_pos[1]])
    date=datetime.datetime(yy,mm,dd)
    self.time_vec=date.toordinal()*24*3600+HH*3600+MM*60+SS+366*24*3600 

  def compress_spectrogram(self, spec_data, spec_time, time_resolution=[], linear_scale=True, interval_range=0, energy_percentile=0):
    if time_resolution:
      read_interval=np.array([0, time_resolution])
    else:
      read_interval=np.array([0, spec_time[-1]])
    
    run=0
    while read_interval[0]<spec_time[-1]-0.5*time_resolution:
      read_list=np.where((spec_time>read_interval[0])*(spec_time<=read_interval[1])==True)[0]
      if len(interval_range)>0:
        temp=np.hstack((spec_time[read_list,None], spec_data[read_list,:]))
        pulse_analysis_result=pulse_interval(temp, energy_percentile=energy_percentile, interval_range=interval_range, plot_type=None, standardization=False)
        temp_PI=pulse_analysis_result.result

      if linear_scale:
        temp_median=10*np.log10(np.median(spec_data[read_list,:],axis=0))-self.sen
        temp_mean=10*np.log10(np.mean(spec_data[read_list,:],axis=0))-self.sen
      else:
        temp_median=np.median(spec_data[read_list,:],axis=0)
        temp_mean=np.mean(spec_data[read_list,:],axis=0)
      
      if self.Result_median.size == 0:
        self.Result_median=np.hstack((np.array(self.time_vec+read_interval[0])/24/3600,temp_median))
        self.Result_mean=np.hstack((np.array(self.time_vec+read_interval[0])/24/3600,temp_mean))
        if len(interval_range)>0:
          self.Result_PI=np.hstack((np.array(self.time_vec+read_interval[0])/24/3600,temp_PI))
      else:
        self.Result_median=np.vstack((np.hstack((np.array(self.time_vec+read_interval[0])/24/3600,temp_median)), self.Result_median))
        self.Result_mean=np.vstack((np.hstack((np.array(self.time_vec+read_interval[0])/24/3600,temp_mean)), self.Result_mean))
        if len(interval_range)>0:
          self.Result_PI=np.vstack((np.hstack((np.array(self.time_vec+read_interval[0])/24/3600,temp_PI)), self.Result_PI))
      run=+1
      read_interval=read_interval+time_resolution
      self.PI=pulse_analysis_result.PI
    
  def save_lts(self, save_filename, folder_id=[]):
    Result=save_parameters()
    Parameters=save_parameters()
    Result.LTS_Result(self.Result_median, self.Result_mean, self.f, self.link, self.PI, self.Result_PI)
    Parameters.LTS_Parameters(self.FFT_size, self.overlap, self.sen, self.sf, self.channel)
    savemat(save_filename, {'Result':Result,'Parameters':Parameters})
    print('Successifully save to '+save_filename)
    
    if folder_id:
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(save_filename)
    
  def run(self, save_filename='LTS.mat', folder_id=[], file_begin=0, num_file=[], duration_read=[]):
    import audioread
    import librosa
    import scipy.signal
    import sys
    import urllib.request
    
    if not num_file:
      num_file=len(self.audioname)
    file_end=file_begin+num_file
    if file_end>len(self.audioname):
      file_end=len(self.audioname)
      
    for file in range(file_begin, file_end):
      print('\r', end='')
      print('Total ', len(self.audioname), ' files, now analyzing file #', file, ': ', self.audioname[file], flush=True, end='')
      
      # Generate MATLAB time format
      self.get_file_time(self.audioname[file])

      # Download audio file
      if self.cloud==1:
        urllib.request.urlretrieve(self.link[file], self.audioname[file])
        path='.'
      elif self.cloud==2:
        temp=self.Gdrive.file_list[file]
        temp.GetContentFile(temp['title'])
        path='.'
      else:
        path=self.link

      # Load audio file
      if file==file_begin:
        with audioread.audio_open(path+'/'+self.audioname[file]) as temp:
          sf=temp.samplerate
      x, self.sf = librosa.load(path+'/'+self.audioname[file], sr=sf)

      if duration_read:
        total_segment=int(np.ceil(len(x)/sf/duration_read))
      else:
        total_segment=1
        duration_read=len(x)/sf
        
      if not self.time_resolution:
        self.time_resolution=duration_read
      
      for segment_run in range(total_segment):
        read_interval=[np.floor(duration_read*segment_run*sf), np.ceil(duration_read*(segment_run+1)*sf)]
        if segment_run==0:
          read_interval[0]=self.skip_duration*sf
        if read_interval[1]>len(x):
          read_interval[1]=len(x)
        if (read_interval[1]-read_interval[0])>(0.5*self.time_resolution*sf):
          self.f,t,P = scipy.signal.spectrogram(x[int(read_interval[0]):int(read_interval[1])], fs=sf, window=('hann'), nperseg=self.FFT_size, 
                                         noverlap=self.overlap, nfft=self.FFT_size, return_onesided=True, mode='psd')
          P = P/np.power(self.pref,2)
          self.time_vec=self.time_vec+duration_read*segment_run
          self.compress_spectrogram(P.T, t, self.time_resolution, linear_scale=True)

      if self.cloud>=1:
        os.remove(self.audioname[file])
    
    temp = np.argsort(self.Result_median[:,0])
    self.Result_median=self.Result_median[temp,:]
    self.Result_mean=self.Result_mean[temp,:]
    self.Result_PI=self.Result_PI[temp,:]
    self.save_lts(save_filename, folder_id)
