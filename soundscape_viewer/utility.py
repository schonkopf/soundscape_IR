"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""

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
    
    def supervised_nmf(self, f, W, feature_length, basis_num):
        self.f=f
        self.W=W
        self.time_frame=feature_length
        self.basis_num=basis_num
    
    def pcnmf(self, f, W, W_cluster, source_num, feature_length, basis_num):
        self.f=f
        self.W=W
        self.W_cluster=W_cluster
        self.k=source_num
        self.time_frame=feature_length
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

class audio_visualization:
    def __init__(self, filename=None,  path=None, offset_read=0, duration_read=None, FFT_size=512, time_resolution=None, window_overlap=0.5, f_range=[], sensitivity=0, environment='wat', plot_type='Both', vmin=None, vmax=None, prewhiten_percent=0, mel_comp=0):
        import audioread
        import librosa
        import os
        
        if not path:
          path=os.getcwd()
        
        if filename:
          # Get the sampling frequency
          with audioread.audio_open(path+'/'+filename) as temp:
              sf=temp.samplerate
              
          # load audio data
          x, _ = librosa.load(filename, sr=sf, offset=offset_read, duration=duration_read)
          self.x=x
          self.sf=sf
          self.run(x, sf, offset_read, duration_read, FFT_size, time_resolution, window_overlap, f_range, sensitivity, environment, plot_type, vmin, vmax, prewhiten_percent, mel_comp)
        
    def run(self, x, sf, offset_read=0, duration_read=None, FFT_size=512, time_resolution=None, window_overlap=0.5, f_range=[], sensitivity=0, environment='wat', plot_type='Both', vmin=None, vmax=None, prewhiten_percent=0, mel_comp=0):
        import scipy.signal
        import librosa
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if environment=='wat':
          P_ref=1
        elif environment=='air':
          P_ref=20
        
        # plot the waveform
        if plot_type=='Both':
          fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(14, 12))
        elif plot_type=='Waveform':
          fig, ax1 = plt.subplots(figsize=(14, 6))
        elif plot_type=='Spectrogram':
          fig, ax2 = plt.subplots(figsize=(14, 6))
        
        if plot_type=='Both' or plot_type=='Waveform':
          ax1.plot(offset_read+np.arange(1,len(x)+1)/sf, x)
          ax1.set_ylabel('Amplitude')
          ax1.set_xlabel('Time')
          ax1.set_xlim(offset_read, offset_read+len(x)/sf)
        
        # run FFT and make a log-magnitude spectrogram
        if time_resolution:
          if FFT_size>time_resolution*sf:
            samples=int(time_resolution*sf/2)
            print('FFT_size has been changed to '+str(samples))
          else:
            samples=FFT_size
          for segment_run in range(int(np.ceil(len(x)/sf/time_resolution))):
            read_interval=[np.floor(time_resolution*segment_run*sf), np.ceil(time_resolution*(segment_run+1)*sf)]
            if read_interval[1]>len(x):
              read_interval[1]=len(x)
            if read_interval[1]-read_interval[0]>=samples:
              f,t,P = scipy.signal.spectrogram(x[int(read_interval[0]):int(read_interval[1])], fs=sf, window=('hann'), nperseg=samples, 
                                               noverlap=int(window_overlap*samples), nfft=FFT_size, return_onesided=True, mode='psd')
              P = P/np.power(P_ref,2)
              if segment_run==0:
                data=10*np.log10(np.mean(P,axis=1))-sensitivity
              else:
                data=np.vstack((data, 10*np.log10(np.mean(P,axis=1))-sensitivity))
          data=data.T
          t=np.arange(time_resolution-time_resolution/2, time_resolution*(data.shape[1]), time_resolution)
        else:
          f,t,P = scipy.signal.spectrogram(x, fs=sf, window=('hann'), nperseg=FFT_size, 
                                           noverlap=int(window_overlap*FFT_size), nfft=FFT_size, return_onesided=True, mode='psd')
          data = 10*np.log10(P/np.power(P_ref,2))-sensitivity
        t=t+offset_read
        
        if mel_comp>0:
          mel_basis = librosa.filters.mel(sr = sf, n_fft=FFT_size, n_mels=mel_comp)
          data = 10*np.log10(np.dot(mel_basis, np.power(10, data/10)))
          f = librosa.core.mel_frequencies(n_mels=mel_comp)

        if prewhiten_percent>0:
          data=matrix_operation.prewhiten(data, prewhiten_percent, 1)
          data[data<0]=0
            
        # f_range: Hz
        if f_range:
            f_list=(f>=min(f_range))*(f<=max(f_range))
            f_list=np.where(f_list==True)[0]
        else:
            f_list=np.arange(len(f))
            
        f=f[f_list]
        data=data[f_list,:]
        
        # plot the spectrogram
        if plot_type=='Both' or plot_type=='Spectrogram':
          im = ax2.imshow(data, vmin=vmin, vmax=vmax,
                       origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[t[0], t[-1], f[0], f[-1]], interpolation='none')
          ax2.set_ylabel('Frequency')
          ax2.set_xlabel('Time')
          cbar = fig.colorbar(im, ax=ax2)
          cbar.set_label('PSD')

        self.data=np.hstack((t[:,None],data.T))
        self.f=f
        
class matrix_operation:
    def __init__(self, header=[]):
        self.header=header
    
    def gap_fill(self, time_vec, data, tail=[]):
        # fill the gaps in a time series
        temp = np.argsort(time_vec)
        time_vec=time_vec[temp]
        if data.ndim>1:
            output=data[temp,:]
        else:
            output=data[temp]
        
        resolution=np.round((time_vec[1]-time_vec[0])*24*3600)
        if not tail:
            n_time_vec=np.arange(np.floor(np.min(time_vec)*24*3600), 
                                 np.ceil(np.max(time_vec)*24*3600)+resolution, resolution)/24/3600
        else:
            n_time_vec=np.arange(np.floor(np.min(time_vec))*24*3600, 
                                 np.ceil(np.max(time_vec))*24*3600+resolution,resolution)/24/3600

        if data.ndim>1:
            save_result=np.zeros((n_time_vec.size, data.shape[1]+1))
        else:
            save_result=np.zeros((n_time_vec.size, 2))
        #save_result[:] = np.nan

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

    def spectral_variation(self, input_data, f, percentile=[], hour_selection=[], month_selection=[]):
        import pandas as pd
        
        if not percentile:
            percentile=np.arange(1,100)
        
        time_vec=input_data[:,0]
        input_data=input_data[:,1:]

        if len(hour_selection)==1:
            hour_selection=np.concatenate((np.array(hour_selection), np.array(hour_selection)+1))
        
        hour=24*(time_vec-np.floor(time_vec))
        if len(hour_selection)>1:
            if hour_selection[1]>hour_selection[0]:
                list_hour=(hour>=np.min(hour_selection))*(hour<np.max(hour_selection))
            if hour_selection[1]<hour_selection[0]:
                hour=(hour<hour_selection[1])*24+hour
                list_hour=(hour>=hour_selection[0])
        else:
            list_hour=hour>0
            
        if len(month_selection)==1:
            month_selection=np.concatenate((np.array(month_selection), np.array(month_selection)+1))
        
        month=np.array(time_vec-693960-2)
        month=pd.to_datetime(month, unit='D',origin=pd.Timestamp('1900-01-01')).month
        if len(month_selection) > 1:
            if(month_selection[0] < month_selection[1]):
                list_month=(month >= np.min(month_selection))*(month < np.max(month_selection))
            if(month_selection[0] > month_selection[1]):
                list_month = (month >= month_selection[0]) + (month < month_selection[1])
        else:
            list_month=month>0
            
        list=list_hour*list_month
        self.PSD_dist = np.percentile(input_data[list,:], percentile, axis=0)
        self.f=f
        self.percentile=percentile
        self.hour_selection=hour_selection
        self.month_selection=month_selection

    def plot_psd(self, freq_scale='linear', amplitude_range=[], f_range=[], fig_width=6, fig_height=6):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        cmap=cm.get_cmap('jet', len(self.percentile))
        cmap_table=cmap(range(len(self.percentile)))
        
        c = np.arange(1, len(self.percentile) + 1)
        cbar = ax.scatter(c, c, c=c, cmap=cmap)
        ax.cla()
        
        for n in np.arange(len(self.percentile)):
            plt.plot(self.f, self.PSD_dist[n,:], color=cmap_table[n,:], linewidth=2)
        
        plt.xscale(freq_scale)
        if not f_range:
            plt.xlim(np.min(self.f), np.max(self.f))
        else:
            plt.xlim(np.min(f_range), np.max(f_range))
        
        if len(amplitude_range)>0:
            plt.ylim(np.min(amplitude_range), np.max(amplitude_range))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        
        if len(self.percentile)>5:
            cbar=fig.colorbar(cbar, ticks=self.percentile[::int(np.ceil(len(self.percentile)/5))])
        else:
            cbar=fig.colorbar(cbar, ticks=self.percentile)
        cbar.set_label('Percentile')
    
    def plot_lts(self, input_data, f, vmin=None, vmax=None, fig_width=18, fig_height=6):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as co
        
        temp=matrix_operation().gap_fill(time_vec=input_data[:,0], data=input_data[:,1:], tail=[])
        temp[:,0]=temp[:,0]+693960-366
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(temp[:,1:].T, vmin=vmin, vmax=vmax,
                       origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[np.min(temp[:,0]), np.max(temp[:,0]), f[0], f[-1]], interpolation='none')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Date')
        ax.xaxis_date()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Amplitude')
        
        return ax, cbar;
    
    def prewhiten(input_data, prewhiten_percent, axis):
        import numpy.matlib
        ambient = np.percentile(input_data, prewhiten_percent, axis=axis)
        if axis==0:
            input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[axis], 1))
        elif axis==1:
            input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[axis], 1).T)
        return input_data

class spectrogram_detection:
  def __init__(self, input, f, threshold, smooth=3, frequency_cut=25, minimum_interval=0, frequency_count=0, pad_size=0, filename='Detection.txt',folder_id=[]):
      from scipy.ndimage import gaussian_filter
      
      time_vec=input[:,0]
      data=input[:,1:]

      if smooth>0:
        level = gaussian_filter(data, smooth)>threshold
      level=level.astype(int).sum(axis = 1)>frequency_count
      begin=time_vec[np.where(np.diff(level.astype(int),1)==1)[0]]
      ending=time_vec[np.where(np.diff(level.astype(int),1)==-1)[0]+1]

      if level[0]:
        begin=np.append(time_vec[0], begin)
      if level[-1]:
        ending=np.append(ending, time_vec[-1])
      
      if minimum_interval>0:
        remove_list=np.where((begin[1:]-ending[0:-1])>minimum_interval)[0]
        begin=begin[np.append(0,remove_list+1)]
        ending=ending[np.append(remove_list, len(ending)-1)]

      begin=begin-pad_size
      ending=ending+pad_size

      min_F=np.array([])
      max_F=np.array([])
      if frequency_cut:
        for n in range(len(begin)):
          psd=np.mean(data[(time_vec>=begin[n])*(time_vec<=ending[n]),:], axis=0)
          f_temp=f[psd>(np.max(psd)-frequency_cut)]
          min_F=np.append(min_F, np.min(f_temp))
          max_F=np.append(max_F, np.max(f_temp))
      self.output=np.vstack([np.arange(len(begin))+1, np.repeat('Spectrogram',len(begin)), np.repeat(1,len(begin)), begin, ending, min_F, max_F]).T
      self.header=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Frequency (Hz)', 'High Frequency (Hz)']
      self.save_txt(filename=filename, folder_id=folder_id)

  def save_txt(self, filename='Separation.txt',folder_id=[]):
      import pandas as pd
      df = pd.DataFrame(self.output, columns = self.header) 
      df.to_csv(filename, sep='\t', index=False)
      print('Successifully save to '+filename)
        
      if folder_id:
        #import Gdrive_upload
        Gdrive=gdrive_handle(folder_id)
        Gdrive.upload(filename)
    
class performance_evaluation:
  def __init__(self, label_filename):
    import pandas as pd
    self.annotations = pd.read_table(label_filename,index_col=0)

  def spectrogram(self, ori_spec, test_spec, fpr_control=0.05, plot=True):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    time_vec=ori_spec[:,0]
    label=0*time_vec
    for n in range(len(self.annotations)):
      label[(time_vec>=self.annotations.iloc[n,2])*(time_vec<=self.annotations.iloc[n,3])]=1
    
    level=test_spec[:,1:].max(axis=1)
    self.fpr, self.tpr, thresholds = roc_curve(label, level)
    self.auc = auc(self.fpr, self.tpr)

    self.threshold = thresholds[(np.abs(self.fpr - fpr_control)).argmin()]
    if plot:
      plt.figure()
      plt.plot(self.fpr, self.tpr)
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.show()

class pulse_interval:
  def __init__(self, sound, energy_percentile=50, interval_range=None, plot_type='Both'):
    duration=len(sound.x)/sound.sf
    data=np.percentile(sound.data[:,1:], energy_percentile, axis=1)
    time_vec=sound.data[:,0]
    self.autocorrelation(data, time_vec, duration, interval_range, plot_type)

  def autocorrelation(self, data, time_vec, duration, interval_range=None, plot_type='Both', millisec=True):
    import matplotlib.pyplot as plt
    self.data=data
    PI=np.arange(-1*data.shape[0], data.shape[0])
    time_resolution=time_vec[1]-time_vec[0]
    if millisec:
      PI=1000*PI*time_resolution

    PI_list=(PI>=min(interval_range))*(PI<=max(interval_range))
    PI_list=np.where(PI_list)[0]
    self.PI=PI[PI_list]
    self.result=np.correlate(data, data, mode='full')[PI_list]
    
    # plot the waveform
    if plot_type=='Both':
        fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(14, 12))
    elif plot_type=='Time':
        fig, ax1 = plt.subplots(figsize=(14, 6))
    elif plot_type=='PI':
        fig, ax2 = plt.subplots(figsize=(14, 6))

    if plot_type=='Both' or plot_type=='Time':
      ax1.plot(time_vec, data)
      ax1.set_xlabel('Time')
      ax1.set_ylabel('Amplitude')
      
    if plot_type=='Both' or plot_type=='PI':
      ax2.plot(self.PI, self.result)
      ax2.set_xlabel('Pulse interval')
      ax2.set_ylabel('Correlation coefficient')
