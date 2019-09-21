"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""
import numpy as np
import pandas as pd
import os
from scipy.io import savemat
from .utility import save_parameters
from .utility import gdrive_handle

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
  def __init__(self, sensitivity=0, channel=1, environment='wat', FFT_size=512, window_overlap=0, initial_skip=0):
    if environment=='wat':
        P_ref=1
    elif environment=='air':
        P_ref=20
      
    print('Please review the following parameters:')
    print('Sensitivity (dB re 1 V/Pa): %s' % (sensitivity))
    print('Recording channel: %s' % (channel))
    print('Recording environment: %s' % (environment))
    print('Reference: %s' % (P_ref))
    print('FFT size: %s' % (FFT_size))
    print('Window overlap (%%): %s' % (window_overlap))
    print('Initial seconds to skip: %s' % (initial_skip))
        
    self.sen = sensitivity
    self.channel = channel
    self.pref = P_ref 
    self.FFT_size = FFT_size
    self.overlap=window_overlap
    self.avoid_duration=initial_skip
 
  def filename_check(self, dateformat='yyyymmdd_HHMMSS', initial=[], year_initial=2000):
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
    filename=self.audioname[0]
    idx=len(initial)
    
    if dateformat.find('yyyy')==-1:
        self.yy_pos = np.array([dateformat.find('yy')+idx, dateformat.find('yy')+2+idx, year_initial])
    else:
        self.yy_pos = np.array([dateformat.find('yyyy')+idx, dateformat.find('yyyy')+4+idx, 0])

    self.mm_pos = np.array([dateformat.find('mm')+idx, dateformat.find('mm')+2+idx])
    self.dd_pos = np.array([dateformat.find('dd')+idx, dateformat.find('dd')+2+idx])
    self.HH_pos = np.array([dateformat.find('HH')+idx, dateformat.find('HH')+2+idx])
    self.MM_pos = np.array([dateformat.find('MM')+idx, dateformat.find('MM')+2+idx])
    self.SS_pos = np.array([dateformat.find('SS')+idx, dateformat.find('SS')+2+idx])
      
    print(filename)
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
    
  def collect_Gdrive(self, folder_id):
    Gdrive=gdrive_handle(folder_id)
    Gdrive.list_query(file_extension='.wav')
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
    
  def run(self, save_filename='LTS.mat', folder_id=[], file_begin=0, num_file=[]):
    import audioread
    import librosa
    import scipy.signal
    import datetime
    import sys
    import urllib.request
    
    Result_median=np.array([]); 
    Result_mean=np.array([]); 
    
    if not num_file:
      num_file=len(self.audioname)
    file_end=file_begin+num_file
    if file_end>len(self.audioname):
      file_end=len(self.audioname)
      
    for file in range(file_begin, file_end):
      print('\r', end='')
      print('Total ', len(self.audioname), 'files, now retrieving file #', file, ':', self.audioname[file], flush=True, end='')
      if self.cloud==1:
        urllib.request.urlretrieve(self.link[file], self.audioname[file])
        path='.'
      elif self.cloud==2:
        temp=self.Gdrive.file_list[file]
        temp.GetContentFile(temp['title'])
        path='.'
      else:
        path=self.link
        
      if file==file_begin:
        with audioread.audio_open(path+'/'+self.audioname[file]) as temp:
          sf=temp.samplerate
      x, sf = librosa.load(path+'/'+self.audioname[file], sr=sf)
      x=x[self.avoid_duration*sf:]
      f,t,P = scipy.signal.spectrogram(x, fs=sf, window=('hamming'), nperseg=None, 
                                       noverlap=self.overlap, nfft=self.FFT_size, 
                                       return_onesided=True, mode='psd')
      P = 10*np.log10(P/np.power(self.pref,2))-self.sen
      if self.cloud>=1:
        os.remove(self.audioname[file])

      infilename=self.audioname[file]
      yy = int(infilename[self.yy_pos[0]:self.yy_pos[1]])
      if self.yy_pos.size == 3:
          yy = yy+self.yy_pos[2]
      mm = int(infilename[self.mm_pos[0]:self.mm_pos[1]])
      dd = int(infilename[self.dd_pos[0]:self.dd_pos[1]])
      HH = int(infilename[self.HH_pos[0]:self.HH_pos[1]])
      MM = int(infilename[self.MM_pos[0]:self.MM_pos[1]])
      SS = int(infilename[self.SS_pos[0]:self.SS_pos[1]])

      # Generate MATLAB time format
      date=datetime.datetime(yy,mm,dd)
      time_vec=date.toordinal()+HH/24+MM/(24*60)+SS/(24*3600)+366 

      if Result_median.size == 0:
          Result_median=np.hstack((np.array(time_vec),10*np.log10(np.median(np.power(10,np.divide(P,10)),axis=1))))
          Result_mean=np.hstack((np.array(time_vec),10*np.log10(np.mean(np.power(10,np.divide(P,10)),axis=1))))
      else:
          Result_median=np.vstack((np.hstack((np.array(time_vec),
                                              10*np.log10(np.median(np.power(10,np.divide(P,10)),axis=1)))), Result_median))
          Result_mean=np.vstack((np.hstack((np.array(time_vec),10*np.log10(np.mean(np.power(10,np.divide(P,10)),axis=1)))), Result_mean))
    
    Result=save_parameters()
    Parameters=save_parameters()
    Result.LTS_Result(Result_median, Result_mean, f, self.link)
    Parameters.LTS_Parameters(self.FFT_size, self.overlap, self.sen, sf, self.channel)
    scipy.io.savemat(save_filename, {'Result':Result,'Parameters':Parameters})
    print('Successifully save to '+save_filename)
    
    if folder_id:
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(save_filename)
