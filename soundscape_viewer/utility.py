"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""

import numpy as np
import pandas as pd
import audioread
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.patches as patches

class gdrive_handle:
    def __init__(self, folder_id, status_print=True):
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
        if status_print:
            print('Now establishing link to Google drive.')
    
    def upload(self, filename, status_print=True):
        upload_ = self.Gdrive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": self.folder_id}], 'title': filename})
        upload_.SetContentFile(filename)
        upload_.Upload()
        if status_print:
            print('Successifully upload to Google drive')
    
    def list_query(self, file_extension, subfolder=False):
        location_cmd="title contains '"+file_extension+"' and '"+self.folder_id+"' in parents and trashed=false"
        self.file_list = self.Gdrive.ListFile({'q': location_cmd}).GetList()
        self.subfolder_list=[]
        if self.file_list!=[]:
          self.subfolder_list=['Top']*len(self.file_list)
        if subfolder:
          self.list_subfolder(file_extension)

    def list_subfolder(self, file_extension):
      self.folder_list = self.Gdrive.ListFile({"q": "mimeType='application/vnd.google-apps.folder' and '"+self.folder_id+"' in parents and trashed=false"}).GetList()
      if self.folder_list!=[]:
        for folder in self.folder_list:
          location_cmd="title contains '"+file_extension+"' and '"+folder['id']+"' in parents and trashed=false"
          subfolder_list=self.Gdrive.ListFile({'q': location_cmd}).GetList()
          if subfolder_list!=[]:
            self.file_list.extend(subfolder_list)
            self.subfolder_list=np.append(self.subfolder_list, [folder['title']]*len(subfolder_list))
        
    def list_display(self):
        n=0
        for file in self.file_list:
            print('File No.'+str(n)+': '+file['title'])
            n+=1
    
class save_parameters:
    def __init__(self):
        self.platform='python'
    
    def supervised_nmf(self, f, W, W_cluster, source_num, feature_length, basis_num):
        self.f=f
        self.W=W
        self.W_cluster=W_cluster
        self.k=source_num
        self.time_frame=feature_length
        self.basis_num=basis_num
    
    def pcnmf(self, f, W, W_cluster, source_num, feature_length, basis_num, sparseness):
        self.f=f
        self.W=W
        self.W_cluster=W_cluster
        self.k=source_num
        self.time_frame=feature_length
        self.basis_num=basis_num
        self.sparseness=sparseness

    def LTS_Result(self, LTS_median, LTS_mean, f, link=[], PI=[], Result_PI=[]):
        self.LTS_median = LTS_median
        self.LTS_mean = LTS_mean
        self.f = f
        self.link = link
        if len(PI)>0:
          self.PI = PI
          self.Result_PI = Result_PI

    def LTS_Parameters(self, FFT_size, overlap, sensitivity, sampling_freq, channel):
        self.FFT_size=FFT_size
        self.overlap=overlap 
        self.sensitivity=sensitivity 
        self.sampling_freq=sampling_freq 
        self.channel=channel

class audio_visualization:
    def __init__(self, filename=None,  path=None, offset_read=0, duration_read=None, FFT_size=512, time_resolution=None, window_overlap=0.5, f_range=[], sensitivity=0, environment='wat', plot_type='Spectrogram', vmin=None, vmax=None, prewhiten_percent=0, mel_comp=0, annotation=None, padding=0):
        if not path:
          path=os.getcwd()
        
        if filename:
          # Get the sampling frequency
          with audioread.audio_open(path+'/'+filename) as temp:
              sf=temp.samplerate
          self.sf=sf
          self.FFT_size=FFT_size
          self.overlap=window_overlap
          self.filename = filename

          # load audio data  
          if annotation:
            df = pd.read_table(annotation,index_col=0) 
            idx_st = np.where(df.columns.values == 'Begin Time (s)')[0][0]
            idx_et = np.where(df.columns.values == 'End Time (s)')[0][0]
            for i in range(len(df)):
              x, _ = librosa.load(path+'/'+filename, sr=sf, offset=df.iloc[i,idx_st]-padding, duration=df.iloc[i,idx_et]-df.iloc[i,idx_st]+padding*2)
              self.run(x, sf, df.iloc[i,idx_st]-padding, FFT_size, time_resolution, window_overlap, f_range, sensitivity, environment, None, vmin, vmax, prewhiten_percent, mel_comp)
              if i==0:
                spec = np.array(self.data)
                time_notation = (i+1)*np.ones((spec.shape[0],1),dtype = int)
              else:
                spec = np.vstack((spec, self.data))
                time_notation = np.vstack((time_notation, (i+1)*np.ones((self.data.shape[0],1),dtype = int)))              

            spec[:,0]=np.arange(spec.shape[0])*(spec[1,0]-spec[0,0])
            self.data=np.array(spec)
            self.time_notation=time_notation

            if plot_type=='Spectrogram':
              fig, ax2 = plt.subplots(figsize=(14, 6))
              im = ax2.imshow(self.data[:,1:].T, vmin=vmin, vmax=vmax, origin='lower',  aspect='auto', cmap=cm.jet,
                          extent=[self.data[0,0], self.data[-1,0], self.f[0], self.f[-1]], interpolation='none')
              ax2.set_ylabel('Frequency')
              ax2.set_xlabel('Time')
              ax2.set_title('Concatenated spectrogram of %s' % self.filename)
                
              if(mel_comp > 0):
                ymin, ymax = ax2.get_ylim()
                N=6
                ax2.set_yticks(np.round(np.linspace(ymin, ymax, N), 2)) 
                idx = np.linspace(0, len(self.f)-1, N, dtype = 'int')
                yticks = self.f[idx]+0.5
                ax2.set_yticklabels(yticks.astype(int))
              cbar = fig.colorbar(im, ax=ax2)
              #cbar.set_label('PSD')
            
          else:
            x, _ = librosa.load(path+'/'+filename, sr=sf, offset=offset_read, duration=duration_read)
            self.x=x-np.mean(x)
            self.run(self.x, sf, offset_read, FFT_size, time_resolution, window_overlap, f_range, sensitivity, environment, plot_type, vmin, vmax, prewhiten_percent, mel_comp)

            
    def run(self, x, sf, offset_read=0, FFT_size=512, time_resolution=None, window_overlap=0.5, f_range=[], sensitivity=0, environment='wat', plot_type='Both', vmin=None, vmax=None, prewhiten_percent=0, mel_comp=0):
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
          ax1.set_title('Waveform of %s' % self.filename)
        
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
              f,t,P = scipy.signal.stft(x[int(read_interval[0]):int(read_interval[1])], fs=sf, window='hann', nperseg=samples, noverlap=int(window_overlap*FFT_size), nfft=FFT_size, detrend='constant', boundary=None, padded=False)
              P = np.abs(P)/np.power(P_ref,2)
              if segment_run==0:
                data=10*np.log10(np.mean(P,axis=1))-sensitivity
              else:
                data=np.vstack((data, 10*np.log10(np.mean(P,axis=1))-sensitivity))
          data=data.T
          t=np.arange(time_resolution-time_resolution/2, time_resolution*(data.shape[1]), time_resolution)
        else:
          f,t,P = scipy.signal.stft(x, fs=sf, window='hann', nperseg=FFT_size, noverlap=int(window_overlap*FFT_size), nfft=FFT_size, detrend='constant', boundary=None, padded=False)
          data = 10*np.log10(np.abs(P)/np.power(P_ref,2))-sensitivity
        t=t+offset_read
        
        if mel_comp>0:
          mel_basis = librosa.filters.mel(sr = sf, n_fft=FFT_size, n_mels=mel_comp)
          data = 10*np.log10(np.dot(mel_basis, np.power(10, data/10)))
          f = librosa.core.mel_frequencies(n_mels=mel_comp)

        if prewhiten_percent>0:
          data, ambient=matrix_operation.prewhiten(data, prewhiten_percent, 1)
          data[data<0]=0
        else:
          ambient=data[:,0:1]*0

        # f_range: Hz
        if f_range:
            f_list=(f>=min(f_range))*(f<=max(f_range))
            f_list=np.where(f_list==True)[0]
        else:
            f_list=np.arange(len(f))
            
        f=f[f_list]
        data=data[f_list,:]
        ambient=ambient[f_list]
        P=P[f_list,:]
        
        # plot the spectrogram
        if plot_type=='Both' or plot_type=='Spectrogram':
          im = ax2.imshow(data, vmin=vmin, vmax=vmax,
                       origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[t[0], t[-1], f[0], f[-1]], interpolation='none')
          ax2.set_ylabel('Frequency')
          ax2.set_xlabel('Time')
          ax2.set_title('Spectrogram of %s' % self.filename)
          if(mel_comp > 0):
              ymin, ymax = ax2.get_ylim()
              N=6
              ax2.set_yticks(np.round(np.linspace(ymin, ymax, N), 2)) 
              idx = np.linspace(0, len(f)-1, N, dtype = 'int')
              yticks = f[idx]+0.5
              ax2.set_yticklabels(yticks.astype(int))
          cbar = fig.colorbar(im, ax=ax2)
          #cbar.set_label('PSD')

        self.data=np.hstack((t[:,None],data.T))
        self.ambient=ambient
        self.f=f
        if not time_resolution:
          self.phase=np.angle(P)

    def convert_audio(self, magnitude_spec, snr_factor=1):
        temp=np.multiply(10**(magnitude_spec[:,1:].T*snr_factor/10), np.exp(1j*self.phase))
        _, self.xrec = scipy.signal.istft(temp, fs=self.sf, nperseg=self.FFT_size, noverlap=int(self.overlap*self.FFT_size))
        
class matrix_operation:
    def __init__(self, header=[]):
        self.header=header
    
    def gap_fill(self, time_vec, data, tail=[], value_input=0):
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
            save_result=np.zeros((n_time_vec.size, data.shape[1]+1))+value_input
        else:
            save_result=np.zeros((n_time_vec.size, 2))+value_input
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
        
        if np.mean(save_result[-1,1:])==value_input:
            save_result=np.delete(save_result, -1, 0)
        return save_result

    def spectral_variation(self, input_data, f, percentile=[], hour_selection=[], month_selection=[]):
        if len(percentile)==0:
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

    def plot_psd(self, freq_scale='linear', amplitude_range=[], f_range=[], fig_width=6, fig_height=6, title=[]):
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
        if title:
            plt.title(title)
        
        if len(self.percentile)>5:
            cbar=fig.colorbar(cbar, ticks=self.percentile[::int(np.ceil(len(self.percentile)/5))])
        else:
            cbar=fig.colorbar(cbar, ticks=self.percentile)
        cbar.set_label('Percentile')
    
    def plot_lts(self, input_data, f, vmin=None, vmax=None, fig_width=18, fig_height=6, lts=True, mel=False, title=[]):
        if lts:
            temp=matrix_operation().gap_fill(time_vec=input_data[:,0], data=input_data[:,1:], tail=[])
            temp[:,0]=temp[:,0]+693960-366
        else:
            temp=input_data
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(temp[:,1:].T, vmin=vmin, vmax=vmax,
                       origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[np.min(temp[:,0]), np.max(temp[:,0]), f[0], f[-1]], interpolation='none')
        ax.set_ylabel('Frequency')
        if lts:
            ax.set_xlabel('Date')
            ax.xaxis_date()
        if mel:
            ymin, ymax = ax.get_ylim()
            N=6
            ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 2)) 
            idx = np.linspace(0, len(f)-1, N, dtype = 'int')
            yticks = f[idx]+0.5
            ax.set_yticklabels(yticks.astype(int))
        if title:
            plt.title(title)
        cbar = fig.colorbar(im, ax=ax)
        #cbar.set_label('Amplitude')
        
        
    def prewhiten(input_data, prewhiten_percent, axis):
        import numpy.matlib
        list=np.where(np.abs(input_data)==float("inf"))[0]
        input_data[list]=float("nan")
        input_data[list]=np.nanmin(input_data)
        
        ambient = np.percentile(input_data, prewhiten_percent, axis=axis)
        if axis==0:
            input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[axis], 1))
        elif axis==1:
            input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[axis], 1).T)
        return input_data, ambient;

    def adaptive_prewhiten(input_data, axis, prewhiten_percent=50, noise_init=None, eps=0.1, smooth=1):
        from scipy.ndimage import gaussian_filter
        list=np.where(np.abs(input_data)==float("inf"))[0]
        input_data[list]=float("nan")
        input_data[list]=np.nanmin(input_data)
        if smooth>0:
          input_data = gaussian_filter(input_data, smooth)
        if axis==1:
          input_data = input_data.T  

        ambient = np.zeros((input_data.shape))
        if noise_init is None:
          noise_init = np.percentile(input_data, prewhiten_percent, axis=0)
        for i in range(input_data.shape[0]):
          if i==0:
            ambient[i] = noise_init
            noise = (1-eps)*noise_init + eps*input_data[i]
          else:
            ambient[i] = noise
            noise = (1-eps)*noise + eps*input_data[i]

        input_data = np.subtract(input_data, ambient)
        input_data[input_data<0]=0
        ambient=noise
        if axis==1:
          input_data=input_data.T
          ambient=ambient.T

        return input_data, ambient;
    
    def frame_normalization(input, axis=0, type='min-max'):
        if axis==0:
          if type=='min-max':
            input=input-np.matlib.repmat(np.min(input, axis=axis),input.shape[0],1)
            input=input/np.matlib.repmat(np.max(input, axis=axis),input.shape[0],1)
          elif type=='sum':
            input=input/np.matlib.repmat(np.sum(input, axis=axis),input.shape[0],1)
        elif axis==1:
          if type=='min-max':
            input=input-np.matlib.repmat(np.min(input, axis=axis).T,input.shape[1],1).T
            input=input/np.matlib.repmat(np.max(input, axis=axis).T,input.shape[1],1).T
          elif type=='sum':
            input=input/np.matlib.repmat(np.sum(input, axis=axis).T,input.shape[1],1).T
        input[np.isnan(input)]=0
        return input
    
    def max_pooling(input, sample=3):
        output=np.zeros([input.shape[0],int(sample*np.ceil(input.shape[1]/sample))])
        output[:,0:input.shape[1]]=input
        output=np.reshape(output, (-1,int(output.shape[1]/sample)))
        output=np.max(output, axis=1)
        output=np.reshape(output,(-1,sample))
        return output

class spectrogram_detection:
  def __init__(self, input, f, threshold, smooth=0, minimum_interval=0, minimum_duration = None, maximum_duration=None, pad_size=0, filename='Detection.txt', folder_id=[], path='./', status_print=True, show_result = True):
      from scipy.ndimage import gaussian_filter
      
      time_vec=input[:,0]
      data=input[:,1:]
      begin=np.array([])

      if smooth>0:
        level_2d = gaussian_filter(data, smooth)>threshold
      elif smooth==0:
        level_2d = data>threshold

      level=level_2d.astype(int).sum(axis = 1)>0
      begin=time_vec[np.where(np.diff(level.astype(int),1)==1)[0]]
      ending=time_vec[np.where(np.diff(level.astype(int),1)==-1)[0]+1]

      if level[0]:
        begin=np.append(time_vec[0], begin)
      if level[-1]:
        ending=np.append(ending, time_vec[-1])
      
      if minimum_interval>0:
        remove_list=np.where((begin[1:]-ending[0:-1])>minimum_interval)[0]
        if len(remove_list)>0:
          begin=begin[np.append(0,remove_list+1)]
          ending=ending[np.append(remove_list, len(ending)-1)]
            
      if maximum_duration:
        keep_list=np.where((ending-begin)<=maximum_duration)[0]
        if len(remove_list)>0:
          begin=begin[keep_list]
          ending=ending[keep_list]

      if minimum_duration:
        keep_list=np.where((ending-begin)>=minimum_duration)[0]
        if len(remove_list)>0:
          begin=begin[keep_list]
          ending=ending[keep_list]
      
      if len(begin)>0:
        begin=begin-pad_size
        ending=ending+pad_size

      min_F=np.array([])
      max_F=np.array([])
      for n in range(len(begin)):
        idx = (time_vec >= begin[n]) & (time_vec < ending[n])
        f_idx = np.where(np.sum(level_2d[idx,:],axis = 0) > 0)[0]
        min_F = np.append(min_F, f[f_idx[0]])
        max_F = np.append(max_F, f[f_idx[-1]])

      self.output=np.vstack([np.arange(len(begin))+1, np.repeat('Spectrogram',len(begin)), np.repeat(1,len(begin)), begin, ending, min_F, max_F]).T
      self.detection=np.vstack((begin, ending)).T
      self.header=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Frequency (Hz)', 'High Frequency (Hz)']
      if filename:
        self.save_txt(filename=path+filename, folder_id=folder_id, status_print=status_print)
      if show_result:
        x_lim=[time_vec[0],time_vec[-1]]
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(data.T, origin='lower',  aspect='auto', cmap=cm.jet, extent=[x_lim[0], x_lim[1], f[0], f[-1]], interpolation='none')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')
        cbar = fig.colorbar(im, ax=ax)
        #cbar.set_label('Amplitude')

        for n in range(len(begin)):
          rect = patches.Rectangle((begin[n], min_F[n]), ending[n]-begin[n], max_F[n]-min_F[n], linewidth=1.5, edgecolor='r', facecolor='none')
          ax.add_patch(rect)

  def save_txt(self, filename='Separation.txt',folder_id=[], status_print=True):
      df = pd.DataFrame(self.output, columns = self.header) 
      df.to_csv(filename, sep='\t', index=False)
      if status_print:
            print('Successifully save to '+filename)
        
      if folder_id:
        #import Gdrive_upload
        Gdrive=gdrive_handle(folder_id, status_print=False)
        Gdrive.upload(filename, status_print=False)
    
class performance_evaluation:
  def __init__(self, label_filename):
    if label_filename:
        self.annotations = pd.read_table(label_filename,index_col=0)

  def spectrogram(self, ori_spec, test_spec, fpr_control=0.05, plot=True):
    from sklearn.metrics import roc_curve, auc
    time_vec=ori_spec[:,0]
    label=0*time_vec
    for n in range(len(self.annotations)):
      label[(time_vec>=self.annotations.iloc[n,2])*(time_vec<=self.annotations.iloc[n,3])]=1
    
    level=test_spec[:,1:].max(axis=1)
    self.fpr, self.tpr, thresholds = roc_curve(label, level)
    self.auc = auc(self.fpr, self.tpr)
    self.all_threshold = thresholds

    self.threshold = thresholds[(np.abs(self.fpr - fpr_control)).argmin()]
    if plot:
      plt.figure()
      plt.plot(self.fpr, self.tpr)
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.show()

class pulse_interval:
  def __init__(self, data, sf=None, energy_percentile=50, interval_range=None, plot_type='Both', standardization=True):
    from scipy.signal import hilbert
    if len(data.shape)==2:
      time_vec=data[:,0]
      data=np.percentile(data[:,1:], energy_percentile, axis=1)
    elif len(data.shape)==1:
      data=data/np.max(np.abs(data))
      #input=np.power(data,2)
      input=np.abs(hilbert(data))
      time_vec=np.arange(len(data))/sf
    self.data=data
    self.autocorrelation(data, time_vec, interval_range, plot_type, millisec=True, standardization=standardization)

  def autocorrelation(self, data, time_vec, interval_range=None, plot_type='Both', millisec=True, standardization=True):
    if plot_type=='Both':
        fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(14, 12))
    elif plot_type=='Time':
        fig, ax1 = plt.subplots(figsize=(14, 6))
    elif plot_type=='PI':
        fig, ax2 = plt.subplots(figsize=(14, 6))

    # plot the waveform
    if plot_type=='Both' or plot_type=='Time':
      ax1.plot(time_vec, self.data)
      ax1.set_xlabel('Time')
      ax1.set_ylabel('Amplitude')

    data=data-np.median(data)
    if standardization==True:
      data=data/np.max(np.abs(data))
    PI=np.arange(-1*data.shape[0], data.shape[0])
    if millisec:
      time_vec=np.round(time_vec*1000000)/1000
      time_resolution=time_vec[1]-time_vec[0]
      PI=np.round(100*PI*time_resolution)/100

    PI_list=(PI>=min(interval_range))*(PI<=max(interval_range))
    PI_list=np.where(PI_list)[0]
    self.PI=PI[PI_list]
    self.result=np.correlate(data, data, mode='full')[PI_list]
      
    if plot_type=='Both' or plot_type=='PI':
      ax2.plot(self.PI, self.result)
      ax2.set_xlabel('Lagged time (ms)')
      ax2.set_ylabel('Correlation score')
    
class tonal_detection:
  def __init__(self, tonal_threshold=6, temporal_prewhiten=50, spectral_prewhiten=50):
    self.tonal_threshold=tonal_threshold
    self.temporal_prewhiten=temporal_prewhiten
    self.spectral_prewhiten=spectral_prewhiten
  
  def local_max(self, input, f, threshold=None, smooth=2):
    # Do vertical and horizontal prewhitening
    temp0=input[:,1:]
    if self.spectral_prewhiten:
      temp0, _=matrix_operation.prewhiten(temp0, prewhiten_percent=self.spectral_prewhiten, axis=1)
    if self.temporal_prewhiten:
      temp0, _=matrix_operation.prewhiten(temp0, prewhiten_percent=self.temporal_prewhiten, axis=0)
    temp0[temp0<0]=0

    # Smooth the spectrogram
    from scipy.ndimage import gaussian_filter
    temp0=gaussian_filter(temp0, sigma=smooth)
    
    # Applying local-max detector to extract whistle contours
    temp=(-1*np.diff(temp0,n=2,axis=1))>self.tonal_threshold
    temp=np.hstack((np.zeros([temp.shape[0],1]),temp))
    temp=np.hstack((temp,np.zeros([temp.shape[0],1])))
    temp2=temp*temp0
    temp2[temp2<0]=0

    # Smooth the contour fragments
    temp2=gaussian_filter(temp2, sigma=smooth)

    # Produce detection result
    if threshold:
      temp3=temp*temp0
      rc=np.nonzero((temp3)>threshold)
      amp=temp3.flatten()
      amp=amp[np.where((amp>threshold))[0]]
      detection=pd.DataFrame(np.hstack((input[rc[0],0:1], f[rc[1]][:,None], amp[:,None])), columns = ['Time','Frequency','Strength']) 
      temp2[temp2<threshold]=threshold
    else:
      detection=np.array([])
      
    # Normalize energy variations 
    temp2=matrix_operation.frame_normalization(temp2, axis=1, type='min-max')
    temp2[np.isnan(temp)]=0
    output=np.hstack((input[:,0:1], temp2))

    return output, detection
