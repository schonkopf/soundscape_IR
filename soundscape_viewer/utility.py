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
    def __init__(self, filename, offset_read=0, duration_read=None, FFT_size=512, time_resolution=None, window_overlap=0.5, sensitivity=0, environment='wat', plot_type='Both', vmin=None, vmax=None, prewhiten_percent=0):
        import audioread
        import librosa
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import os
        import scipy.signal

        if environment=='wat':
          P_ref=1
        elif environment=='air':
          P_ref=20
        
        # Get the sampling frequency
        with audioread.audio_open(os.getcwd()+'/'+filename) as temp:
            sf=temp.samplerate
            
        # load audio data
        x, _ = librosa.load(filename, sr=sf, offset=offset_read, duration=duration_read)
        
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
          for segment_run in range(int(np.ceil(len(x)/sf/time_resolution))):
            read_interval=[np.floor(time_resolution*segment_run*sf), np.ceil(time_resolution*(segment_run+1)*sf)]
            if read_interval[1]>len(x):
              read_interval[1]=len(x)
            f,t,P = scipy.signal.spectrogram(x[int(read_interval[0]):int(read_interval[1])], fs=sf, window=('hann'), nperseg=None, 
                                        noverlap=window_overlap, nfft=FFT_size, return_onesided=True, mode='psd')
            P = P/np.power(P_ref,2)
            if segment_run==0:
              data=10*np.log10(np.mean(P,axis=1))-sensitivity
            else:
              data=np.vstack((data, 10*np.log10(np.mean(P,axis=1))-sensitivity))
          data=data.T
          t=np.arange(time_resolution-time_resolution/2, time_resolution*(segment_run+1), time_resolution)
        else:
          f,t,P = scipy.signal.spectrogram(x, fs=sf, window=('hann'), nperseg=None, 
                                           noverlap=window_overlap, nfft=FFT_size, return_onesided=True, mode='psd')
          data = 10*np.log10(P/np.power(P_ref,2))-sensitivity
        t=t+offset_read
        
        if prewhiten_percent>0:
          data=matrix_operation.prewhiten(data, prewhiten_percent, 1)
        
        # plot the spectrogram
        if plot_type=='Both' or plot_type=='Spectrogram':
          im = ax2.imshow(data, vmin=vmin, vmax=vmax,
                       origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[t[0], t[-1], f[0], f[-1]], interpolation='none')
          ax2.set_ylabel('Frequency')
          ax2.set_xlabel('Time')
          cbar = fig.colorbar(im, ax=ax2)
          cbar.set_label('PSD')

        self.x=x
        self.sf=sf
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
  def __init__(self, input, f, threshold, smooth=3, frequency_cut=25, pad_size=0, filename='Detection.txt',folder_id=[]):
      from scipy.ndimage import gaussian_filter1d
      
      time_vec=input[:,0]
      data=input[:,1:]
      level = 10*np.log10((10**(data/10)).sum(axis=1))
      if smooth>0:
        level = gaussian_filter1d(level, smooth)
      level=level>threshold
      begin=time_vec[np.where(np.diff(level.astype(int),1)==1)[0]]
      ending=time_vec[np.where(np.diff(level.astype(int),1)==-1)[0]+1]

      if level[0]>threshold:
        begin=np.append(time_vec[0], begin)
      if level[-1]>threshold:
        ending=np.append(ending, time_vec[-1])

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
      print(self.output)

  def save_txt(self, filename='Separation.txt',folder_id=[]):
      import pandas as pd
      df = pd.DataFrame(self.output, columns = self.header) 
      df.to_csv(filename, sep='\t', index=False)
      print('Successifully save to '+filename)
        
      if folder_id:
        #import Gdrive_upload
        Gdrive=gdrive_handle(folder_id)
        Gdrive.upload(filename)
    
