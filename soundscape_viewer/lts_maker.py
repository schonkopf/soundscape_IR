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
    Making a long-term spectrogram to investigate the temporal and spectral variations of long-duration recordings.
    
    This class generates a visualization of long-duration recordings called long-term spectrogram. Before running this program, copy a set of recordings obtained from the same recorder (presumably at the same location) to a folder. There are three procedures for making a long-term spectrogram. First, recordings are divided into short segments. Next, each segment is transformed into a spectrogram using discrete Fourier transform (DFT), and each spectrogram is compressed into a vector of power spectrum according to two statistical measurements (median and mean) to retain spectral variation. Finally, all power spectra are concatenated and saved as a mat file.
      
    Parameters
    ----------
    sensitivity : float, default = 0
        Sensitivity of recording system (in dB re 1 V /μPa). 
        
        Use 0 when sensitivity is not available.

    channel : int ≥ 1, default = 1
        Recording channel for analysis. 
        
        In stereo recordings, set to 1 for the left channel and set to 2 for the right channel.

    environment : {'wat', 'air'}, default = 'wat'
        Recording environment (underwater or in air) of the audio file.

    FFT_size : int > 0, default = 512
        Window size to perform discrete Fourier transform (in samples).

    initial_skip : float ≥ 0, default = 0
        Initial duration (in seconds) of each audio recording to skip the audio analysis.
        
        Some autonomous recorders, e.g., SoundTrap, support the function to investigate changes in microphone/hydrophone sensitivity by generating calibration tones in the beginning of each recording session. Use this parameter for removing the fragment containing calibration tones.
          
    time_resolution : None or float > 0, default = None
        Time resolution of a long-term spectrogram (in seconds). 
        
        For each audio recording, ``lts_maker`` generates one power spectrum for each fragment according to ``time_resolution``. However, if ``time_resolution`` is not set, only one power spectrum is generated for the entire audio recording. In this case, the ``time_resolution`` equals the duty cycle of recordings.
        
    Attributes
    ----------
    f : ndarray of shape (frequency,)
        Frequency of spectrogram data.
        
    Result_median : ndarray of shape (time, frequency+1)
        Median-based long-term spectrogram.

        The first column is time, and the subsequent columns are power spectral densities associated with ``f``.
        
    Result_mean : ndarray of shape (time, frequency+1)
        Mean-based long-term spectrogram.

        The first column is time, and the subsequent columns are power spectral densities associated with ``f``.
          
    Examples
    --------
    Generate a long-term spectrogram of locally saved audio recordings.
    
    >>> from soundscape_IR.soundscape_viewer import lts_maker
    >>> LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=512, initial_skip=0)
    >>> LTS_run.collect_folder(path='./wav/')
    >>> LTS_run.filename_check(dateformat='yymmddHHMMSS',initial='1207984160.',year_initial=2000)
    >>> LTS_run.run(save_filename='LTS.mat')

    Generate an LTS for audio recordings hosted on a Google Drive folder.
    
    >>> from soundscape_IR.soundscape_viewer import lts_maker
    >>> LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=512, initial_skip=0)
    >>> LTS_run.collect_Gdrive(folder_id='XXXXXXXXXXXXXXXXXXXXXXXXXXX')
    >>> LTS_run.filename_check(dateformat='yyyymm_ddHHMMSS',initial='TW_SiteA_',year_initial=0)
    >>> LTS_run.run(save_filename='LTS.mat')

    """
    def __init__(self, sensitivity=0, channel=1, environment='wat', FFT_size=512, initial_skip=0, time_resolution=None):
        if environment=='wat':
            P_ref=1
        elif environment=='air':
            P_ref=20

        self.sen = sensitivity
        self.channel = channel
        self.pref = P_ref 
        self.FFT_size = FFT_size
        self.overlap=int(0.5*FFT_size)
        self.skip_duration=initial_skip
        self.time_resolution=time_resolution
        self.Result_median=np.array([])
        self.Result_mean=np.array([])
        self.Result_PI=np.array([])
        self.PI=np.array([])

    def filename_check(self, dateformat='yyyymmdd_HHMMSS', initial=[], year_initial=2000, filename=[]):
        """
        Check recording time information retrieved from the file name.
        
        Enter the filename information to extract the recording date and time from each audio file. For example, if the file name is 'KT08_20171118_123000.wav', please enter ``initial='KT08_'`` and ``dateformat='yyyymmdd_HHMMSS'``. 
        
        If the file name does not contain information of century (when ``dateformat='yymmdd_HHMMSS'``), specify century in ``year_initial``. For example, the recording 'KT08_171118_123000.wav' was made in 2017, then set ``year_initial=2000``.
        
        Example 1: TW-LHC01-150102-001530.wav
        
        * initial='TW-LHC01-'
        * dateformat='yymmdd-HHMMSS'
        * year_initial=2000
        
        Example 2: TW-LHC-20150102 001530.wav
        
        * initial='TW-LHC-'
        * dateformat='yyyymmdd HHMMSS'
        * year_initial=0
        
        Example 3: 190730053000.wav
        
        * initial=[]
        * dateformat='yymmddHHMMSS'
        * year_initial=2000

        Parameters
        ----------
        dateformat : str, default = 'yyyymmdd_HHMMSS'
            Recording date and time format on the file name, such as 'yymmddHHMMSS',  'yyyymmddHHMMSS',  'yymmdd_HHMMSS',  'yyyymmdd_HHMMSS'.
        
        initial= [] or str, default = []
            File name initials before the recording date and time information.
        
        year_initial= {0, 1900, 2000}, default = 2000
            Century of recording date. 
            
            For dateformat = 'yymmddHHMMSS' or 'yymmdd_HHMMSS', use this parameter to declare the century in which the recording is made. 
            
            Set to 0 if dateformat = 'yyyymmddHHMMSS' or 'yyyymmdd_HHMMSS'.
            
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

    def collect_folder(self, path, file_extension='.wav'):
        """
        Specify the folder where the recording files are located.
        
        Parameters
        ----------
        folder : str
            Folder path for analysis.
            
        file_extension : {'.wav', '.WAV'}, default = '.wav'
            Data format for analysis.
        
        """
        file_list=os.listdir(path)
        file_list.sort(key = str.lower)
        self.link = path
        self.cloud = 0   
        n = 0
        self.audioname=np.array([], dtype=object)
        for filename in file_list:
            if filename.endswith(file_extension):
                self.audioname = np.append(self.audioname, filename)
                n = n+1
        print('Identified ', len(self.audioname), 'files')

    def collect_Gdrive(self, folder_id, file_extension='.wav', subfolder=False):
        """
        Specify a Drive folder where the recording files are located.
        
        Parameters
        ----------
        folder_id : str
            The folder ID of Google Drive folder for analysis.
            
            See https://ploi.io/documentation/database/where-do-i-get-google-drive-folder-id for the detial of folder ID. 
            
        file_extension : {'.wav', '.WAV'}, default = '.wav'
            Data format for analysis.
            
        subfolder : boolean, default = False
            Set to True if subfolders also contain recordings for analysis.
        
        """
        from natsort import index_natsorted
        Gdrive=gdrive_handle(folder_id)
        Gdrive.list_query(file_extension=file_extension, subfolder=subfolder)
        self.cloud=2
        self.link=np.array([], dtype=object)
        self.audioname=np.array([], dtype=object)
        self.Gdrive=Gdrive
        for file in Gdrive.file_list:
            self.link=np.append(self.link, file['alternateLink'])
            self.audioname=np.append(self.audioname, file['title'])
        idx = index_natsorted(self.audioname)
        self.audioname = self.audioname[idx]
        Gdrive.file_list = np.array(Gdrive.file_list)[idx]
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

    def compress_spectrogram(self, spec_data, spec_time, time_resolution=[], linear_scale=True, interval_range=[], energy_percentile=0):
        if time_resolution:
            read_interval=np.array([0, time_resolution])
        else:
            read_interval=np.array([0, spec_time[-1]])
            time_resolution=spec_time[-1]

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
        if len(interval_range)>0:
            self.PI=pulse_analysis_result.PI

    def save_lts(self, save_filename, folder_id=[], status_print=True):
        Result=save_parameters()
        Parameters=save_parameters()
        Result.LTS_Result(self.Result_median, self.Result_mean, self.f, self.link, self.PI, self.Result_PI)
        Parameters.LTS_Parameters(self.FFT_size, self.overlap, self.sen, self.sf, self.channel)
        savemat(save_filename, {'Result':Result,'Parameters':Parameters})
        print('Successifully save to '+save_filename)

        if folder_id:
            Gdrive=gdrive_handle(folder_id, status_print=True)
            Gdrive.upload(save_filename, status_print=True)

    def run(self, save_filename='LTS.mat', folder_id=[], start=1, num_file=None, duration_read=None):
        """
        Initiate analysis procedures. 
        
        Parameters
        ----------
        save_filename : str, default = 'LTS.mat'
            Name of the mat file.
            
        folder_id : [] or str, default = []
            The folder ID of Google Drive folder for saving analysis result.
            
            See https://ploi.io/documentation/database/where-do-i-get-google-drive-folder-id for the detial of folder ID. 
        
        start : int ≥ 1, default = 1
            The file number to begin LTS analysis.
        
        num_file : None or int ≥ 1, default = None
            The number of files after ``start`` for LTS analysis.
            
        duration_read : None or float > 0, default = None
            When an audio file is too long (e.g., last several hours), reading the entire audio recording at once may occupy too much memory. Specifying the duration (in seconds) for each read can reduce the memory space used and speed up the analysis.
            
        """
        import audioread
        import librosa
        import scipy.signal
        import sys
        import urllib.request
        
        start=start-1
        if not num_file:
            num_file=len(self.audioname)
        file_end=start+num_file
        if file_end>len(self.audioname):
            file_end=len(self.audioname)

        for file in range(start, file_end):
            print('\r', end='')
            print('Total ', len(self.audioname), ' files, now analyzing file #', file+1, ': ', self.audioname[file], flush=True, end='')

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
            if file==start:
                with audioread.audio_open(path+'/'+self.audioname[file]) as temp:
                    sf=temp.samplerate
            x, self.sf = librosa.load(path+'/'+self.audioname[file], sr=sf, mono=False)
            if len(x.shape)==2:
                x=x[channel-1,:]
            x=x-np.mean(x)

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
                    self.f,t,P = scipy.signal.spectrogram(x[int(read_interval[0]):int(read_interval[1])], fs=sf, window=('hann'), nperseg=self.FFT_size, noverlap=self.overlap, nfft=self.FFT_size, return_onesided=True, mode='psd')
                    P = P/np.power(self.pref,2)
                    self.time_vec=self.time_vec+duration_read*segment_run
                    self.compress_spectrogram(P.T, t, self.time_resolution, linear_scale=True)

            if self.cloud>=1:
                os.remove(self.audioname[file])
        temp = np.argsort(self.Result_median[:,0])
        self.Result_median=self.Result_median[temp,:]
        self.Result_mean=self.Result_mean[temp,:]
        if len(self.Result_PI)>0:
            self.Result_PI=self.Result_PI[temp,:]
        self.save_lts(save_filename, folder_id)
