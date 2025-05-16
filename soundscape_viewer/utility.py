import numpy as np
import pandas as pd
import audioread
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.patches as patches
from scipy.io import savemat
import copy
import soundfile

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
        self.platform='soundscape_IR V1.1'

    def feature_extraction(self, detection, f, spectral_result, PI, PI_result):
        self.detection=detection
        self.f=f
        self.spectral_result=spectral_result
        self.PI=PI
        self.PI_result=PI_result
    
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
    """
    This class loads the waveform of an audio recording (only WAVE files) and applies discrete Fourier transform to generate a spectrogram on the Hertz or Mel scale. 
    
    Two noise reduction methods are provided. Welch’s method reduces random noise by measuring the average power spectrum over a short period of time (Welch 1967). 
    
    Spectrogram prewhitening method finds the spectral pattern of background noise by calculating a specific percentile of power spectral densities at each frequency bin and subsequently subtracting the entire spectrogram from the background noise (Lin et al. 2021). After the prewhitening procedure, sound intensities are converted into signal-to-noise ratios.

    This class can also generate a concatenated spectrogram of annotated fragments by importing a text file containing annotations. The text file can be prepared using the Raven software (https://ravensoundsoftware.com).

    Parameters
    ----------
    filename : str 
        Name of the audio file.

    path : None or str, default = None
        Path of the input audio file.

        If ``path`` is not set, current folder is used.
        
    channel : int ≥ 1, default = 1
        Recording channel for analysis. 
        
        In stereo recordings, set to 1 for the left channel and set to 2 for the right channel.

    offset_read : float ≥ 0, default = 0
        Start reading time of the input audio file (in seconds).

    duration_read : None or float > 0, default = None
        Duration load after ``offset_read`` (in seconds).

        If ``duration_read`` is not set, the entire audio file after ``offset_read`` is processed.

    FFT_size : int > 0, default = 512
        Window size to perform discrete Fourier transform (in samples).

    window_overlap : float [0, 1), default = 0.5
        Ratio of overlap between consecutive windows.

    time_resolution : None or float > 0, default = None
        Applying Welch's method to calculate averaging power spectra. 
        
        After generating a regular spectrogram, a mean power spectrum is calculated within the range of ``time_resolution`` (in seconds).

        ``time_resolution`` should not be smaller than (1-window_overlap)*FFT_size/sf, which is the original time resolution of a regular spectrogram. ``sf`` represents the sampling frequency of an audio file. 

    f_range : None or a list of 2 scalars [min, max], default = None
        Minimum and maximum frequencies of the spectrogram.

    prewhiten_percent : None or float [0, 100), default = None
        Applying prewhitening method to suppress background noise and convert power spectral densities into signal-to-noise ratios.
        
        After generating a regular spectrogram and applying Welch's averaging method, the spectral pattern of background noise is estimated by calculating the percentile of power spectral densities in each frequency bin. Subtracting background noise from the whole spectrogram, signal-to-noise ratios below 0 are converted to 0.

    mel_comp : None or int ≥ 0, default = None
        Number of Mel bands to generate. 

        If ``mel_comp`` is not set, a Hertz scaled spectrogram is generated.

    sensitivity : float, default = 0
        Recording sensitivity of the input audio file (in dB re 1 V/μPa). 
        
        Set to 0 when sensitivity information is not available.

    environment : {'wat', 'air'}, default = 'wat'
        Recording environment (underwater or in air) of the input audio file.

    plot_type : None or {'Spectrogram', 'Waveform', 'Both'}, default = 'Spectrogram'
        Choose to only generate a spectrogram or a waveform, or do both plots.

    vmin, vmax : None or float, default = None
        The data range that the colormap covers. 

        By default (None), the colormap covers the complete value range of the spectrogram.

    annotation : None or str, default = None
        Path and name of the text file containing annotations. 

        The text file should be saved using the format supported by the Raven software (https://ravensoundsoftware.com).

    padding : float ≥ 0, default = 0
        Duration that increase the length before and after each annotation (in seconds).

    Attributes
    ----------
    sf : int
        Sampling frequency of the input audio file.

    x : ndarray of shape (time,)
        Waveform data, with subtraction of the DC value.

    f : ndarray of shape (frequency,)
        Frequency of spectrogram data (in Hertz).

    data : ndarray of shape (time, frequency+1)
        Log-scaled power spectral densities (in dB). 

        The first column is time, and the subsequent columns are power spectral densities associated with ``f``.

    phase : ndarray of shape (frequency,time)
        Phase of the spectrogram data.

        Not available when ``time_resolution`` is set.

    ambient : ndarray of shape (frequency,)
        Background noise estimated using the spectrogram prewhitening method.

    Examples
    --------
    Load an audio recording and generate the associated waveform and spectrogram.
    
    >>> from soundscape_IR.soundscape_viewer import audio_visualization
    >>> sound = audio_visualization(filename='audio.wav', path='./wav/', f_range=[0, 8000], plot_type='Both')
                          
    Use Welch's method to suppress random noise and reduce time resolution.
    
    >>> from soundscape_IR.soundscape_viewer import audio_visualization
    >>> sound = audio_visualization(filename='audio.wav', path='./wav/', time_resolution=0.1, f_range=[0, 8000], plot_type='Spectrogram')
    
    Generate a prewhitened spectrogram in Mel scale.
    
    >>> from soundscape_IR.soundscape_viewer import audio_visualization
    >>> sound = audio_visualization(filename='audio.wav', path='./wav/', FFT_size=2048, prewhiten_percent=10, mel_comp=128, plot_type='Spectrogram')

    Generate a concatenated spectrogram by importing annotations, with 0.5 s padding before and after each annotation.
    
    >>> from soundscape_IR.soundscape_viewer import audio_visualization
    >>> sound = audio_visualization(filename='audio.wav', path='./wav/', annotation='./txt/annotations.txt', padding=0.5)
    
    References
    ----------
    .. [1] Welch, P. D. (1967). The use of Fast Fourier Transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. IEEE Transactions on Audio and Electroacoustics, 15 (2): 70–73. https://doi.org/10.1109/TAU.1967.1161901
    
    .. [2] Lin, T.-H., Akamatsu, T., & Tsao, Y. (2021). Sensing ecosystem dynamics via audio source separation: A case study of marine soundscapes off northeastern Taiwan. PLoS Computational Biology, 17(2), e1008698. https://doi.org/10.1371/journ al.pcbi.1008698
    """
    def __init__(self, filename, path=None,  channel=1, offset_read=0, duration_read=None, FFT_size=512, time_resolution=None, window_overlap=0.5, f_range=None, sensitivity=0, environment='wat', plot_type='Spectrogram', vmin=None, vmax=None, prewhiten_percent=None, mel_comp=None, annotation=None, padding=0, save_clip_path=None, resolution_method='mean'):
        self.filename = filename
        if not path:
            path=os.getcwd()
        
        if filename:
            # Get the sampling frequency
            with audioread.audio_open(path+'/'+filename) as temp:
                sf=temp.samplerate
            self.sf=sf
            self.FFT_size=FFT_size
            self.overlap=window_overlap
            self.data=np.array([])

            # load audio data  
            if annotation:
                df = pd.read_table(annotation,index_col=0) 
                idx_st = np.where(df.columns.values == 'Begin Time (s)')[0][0]
                idx_et = np.where(df.columns.values == 'End Time (s)')[0][0]
                if len(df)>0:
                    for i in range(len(df)):
                        x, _ = librosa.load(path+'/'+filename, sr=sf, offset=df.iloc[i,idx_st]-padding, duration=df.iloc[i,idx_et]-df.iloc[i,idx_st]+padding*2, mono=False)
                        if len(x.shape)==2:
                            x=x[channel-1,:]
                        x=x-np.mean(x)
                        self.run(x, sf, df.iloc[i,idx_st]-padding, FFT_size, time_resolution, window_overlap, f_range, sensitivity, environment, None, vmin, vmax, prewhiten_percent, mel_comp, resolution_method)
                        if i==0:
                            spec = np.array(self.data)
                            time_notation = (i+1)*np.ones((spec.shape[0],1),dtype = int)
                        else:
                            spec = np.vstack((spec, self.data))
                            time_notation = np.vstack((time_notation, (i+1)*np.ones((self.data.shape[0],1),dtype = int)))
                        if save_clip_path:
                            soundfile.write(save_clip_path+'/'+filename[0:-3]+str(i+1)+'.wav', x, sf)

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
    
                        if mel_comp:
                            ymin, ymax = ax2.get_ylim()
                            N=6
                            ax2.set_yticks(np.round(np.linspace(ymin, ymax, N), 2)) 
                            idx = np.linspace(0, len(self.f)-1, N, dtype = 'int')
                            yticks = self.f[idx]+0.5
                            ax2.set_yticklabels(yticks.astype(int))
                        cbar = fig.colorbar(im, ax=ax2)
                        #cbar.set_label('PSD')
            
            else:
                x, _ = librosa.load(path+'/'+filename, sr=sf, offset=offset_read, duration=duration_read, mono=False)
                if len(x.shape)==2:
                    x=x[channel-1,:]
                self.x=x-np.mean(x)
                if FFT_size:
                    self.run(self.x, sf, offset_read, FFT_size, time_resolution, window_overlap, f_range, sensitivity, environment, plot_type, vmin, vmax, prewhiten_percent, mel_comp, resolution_method)

            
    def run(self, x, sf, offset_read=0, FFT_size=512, time_resolution=None, window_overlap=0.5, f_range=[], sensitivity=0, environment='wat', plot_type='Both', vmin=None, vmax=None, prewhiten_percent=None, mel_comp=None, resolution_method='mean'):
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
        f,t,P = scipy.signal.stft(x, fs=sf, window='hann', nperseg=FFT_size, noverlap=int(window_overlap*FFT_size), nfft=FFT_size, detrend='constant', boundary=None, padded=False)
        if time_resolution:
            P = np.abs(P)/np.power(P_ref,2)
            P_scaled = matrix_operation().rescale(np.hstack((t[:,None], P.T)), time_resolution, resolution_method)
            data=10*np.log10(P_scaled[:,1:].T)-sensitivity
            t=P_scaled[:,0]
        else:
            data = 10*np.log10(np.abs(P)/np.power(P_ref,2))-sensitivity
        t=t+offset_read
        
        if mel_comp:
            mel_basis = librosa.filters.mel(sr = sf, n_fft=FFT_size, n_mels=mel_comp)
            data = 10*np.log10(np.dot(mel_basis, np.power(10, data/10)))
            f = librosa.core.mel_frequencies(n_mels=mel_comp)

        if prewhiten_percent:
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
        
        # plot the spectrogram
        if plot_type=='Both' or plot_type=='Spectrogram':
            im = ax2.imshow(data, vmin=vmin, vmax=vmax,
                       origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[t[0], t[-1], f[0], f[-1]], interpolation='none')
            ax2.set_ylabel('Frequency')
            ax2.set_xlabel('Time')
            if self.filename:
                ax2.set_title('Spectrogram of %s' % self.filename)
            if mel_comp:
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
            self.phase=np.angle(P[f_list,:])

    def convert_audio(self, magnitude_spec, snr_factor=1):
        """
        This method recovers a time-domain waveform from a magnitude spectrogram by using inverse discrete Fourier transform. 
        
        The input spectrogram should be prepared in Hertz scale. Phase data is necessary during the inverse procedure,  so ``time_resolution``, ``f_range``, and ``mel_comp`` should not be used when loading an audio file.

        Parameters
        ----------
        magnitude_spec : ndarray of shape (time, frequency+1)
            Log-scaled power spectral densities, presumably to be noise filtered. 
            
            The first column is time, and the subsequent columns are power spectral densities associated with ``f``.
        
        snr_factor : float > 0, default=1
            A ratio for amplifying the input signal.

        Attributes
        ----------
        xrec : ndarray of shape (time,)
            Reconstructed waveform data.

        Examples
        --------
        Load an audio recording and apply spectrogram prewhitening to suppress background noise. Then, use the prewhitened spectrogram to generate a waveform.
        
        >>> from soundscape_IR.soundscape_viewer import audio_visualization
        >>> sound = audio_visualization(filename='audio.wav', path='./wav/', FFT_size=512, window_overlap=0.5, prewhiten_percent=10, plot_type=None)
        >>> sound.convert_audio(sound.data, snr_factor=1.5)
        >>>
        >>> from IPython.display import Audio
        >>> Audio(sound.xrec, rate=sound.sf)
        
        Use a source separation model to separate non-target signals and reconstruct the waveform of target source.
        
        >>> from soundscape_IR.soundscape_viewer import source_separation
        >>> model=source_separation(filename='model.mat')
        >>> model.prediction(input_data=sound.data, f=sound.f)
        >>> sound.convert_audio(model.separation[0], snr_factor=1.5)
        """

        temp=np.multiply(10**(magnitude_spec[:,1:].T*snr_factor/10), np.exp(1j*self.phase))
        _, self.xrec = scipy.signal.istft(temp, fs=self.sf, nperseg=self.FFT_size, noverlap=int(self.overlap*self.FFT_size))

class matrix_operation:
    def __init__(self, header=[]):
        self.header=header

    def rescale(self, spec, time_reso, method='mean'):
        rescale_spec=spec
        time_ori=spec[0,0]
        if time_reso>(spec[1,0]-spec[0,0]):
            spec[:,0]=np.floor((spec[:,0]-spec[0,0])/time_reso)
            if method=='mean':
                rescale_spec=np.array(pd.DataFrame(spec).groupby(by=[0]).mean().reset_index())
            elif method=='median':
                rescale_spec=np.array(pd.DataFrame(spec).groupby(by=[0]).median().reset_index())
            elif method=='max':
                rescale_spec=np.array(pd.DataFrame(spec).groupby(by=[0]).max().reset_index())
            elif method=='min':
                rescale_spec=np.array(pd.DataFrame(spec).groupby(by=[0]).min().reset_index())
            rescale_spec[:,0]=time_reso*rescale_spec[:,0]+time_ori
        return rescale_spec
    
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

    def plot_psd(self, freq_scale='linear', x_label='Frequency', y_label='PSD', amplitude_range=[], f_range=[], fig_width=6, fig_height=6, title=[]):
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
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        
        if len(self.percentile)>5:
            cbar=fig.colorbar(cbar, ax=ax, ticks=self.percentile[::int(np.ceil(len(self.percentile)/5))])
        else:
            cbar=fig.colorbar(cbar, ax=ax, ticks=self.percentile)
        cbar.set_label('Percentile')
        return fig, ax
    
    def plot_lts(self, input_data, f, day_correct=0, vmin=None, vmax=None, fig_width=18, fig_height=6, lts=True, mel=False):
        if day_correct=='windows':
            day_correct=-719163
        if lts:
            temp=matrix_operation().gap_fill(time_vec=input_data[:,0], data=input_data[:,1:], tail=[])
            temp[:,0]=temp[:,0]+693960-366
        else:
            temp=input_data
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(temp[:,1:].T, vmin=vmin, vmax=vmax,
                       origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[np.min(temp[:,0]+day_correct), np.max(temp[:,0]+day_correct), f[0], f[-1]], interpolation='none')
        ax.set_ylabel('Frequency')
        if lts:
            ax.xaxis_date()
        if mel:
            ymin, ymax = ax.get_ylim()
            N=6
            ax.set_yticks(np.round(np.linspace(ymin, ymax, N), 2)) 
            idx = np.linspace(0, len(f)-1, N, dtype = 'int')
            yticks = f[idx]+0.5
            ax.set_yticklabels(yticks.astype(int))
        cbar = fig.colorbar(im, ax=ax)
        #cbar.set_label('Amplitude')
        return fig, ax
        
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
    """
    This class applies an energy thresholding method to find regions of interest displayed on a spectrogram. 
    
    It uses a known estimate of minimum signal interval to separate regions of interest and subsequently remove false alarms according to the minimum and maximum signal duration. 
    
    The output is a table containing Begin Time (s), End Time (s), Low Frequency (Hz), High Frequency (Hz), and Maximum SNR (dB). The table is saved in a text file, which can be imported to the Raven software (https://ravensoundsoftware.com).

    Parameters
    ----------
    input : ndarray of shape (time, frequency+1)
        Spectrogram data for analysis. 
        
        The first column is time, and the subsequent columns are power spectral densities (or signal-to-noise ratios) associated with ``f``. Use the same spectrogram format generated from ``audio_visualization``.
    
    f : ndarray of shape (frequency,)
        Frequency of the input spectrogram data.
    
    threshold : float
        Energy threshold for binarizing the spectrogram data. 
        
        Only time and frequency bins with intensities higher than ``threshold`` are considered as detections.
    
    smooth : float ≥ 0, default = 0
        Standard deviation of Gaussian kernel for smoothing the spectrogram data. 
        
        See ``sigma`` in ``scipy.ndimage.gaussian_filter`` for details.
    
    minimum_interval : float ≥ 0, default = 0
        Minimum time interval (in seconds) for the algorithm to separate two regions of interest. 
        
        If the interval between two signals is shorter than ``minimum_interval``, the two signals are considered to be the same detection.
    
    minimum_duration, maximum_duration : None or float > 0, default = None
        Minimum and maximum signal durations of each detection (in seconds).
    
    pad_size : float ≥ 0, default = 0
        Duration that increases the length before and after each detection (in seconds).
    
    filename : str, default = 'Detection.txt'
        Name of the txt file contains detections.

    path : str
        Path to save detection result.
    
    folder_id : [] or str, default = []
        The folder ID of Google Drive folder for saving detection result.
        
        See https://ploi.io/documentation/database/where-do-i-get-google-drive-folder-id for the detial of folder ID. 
    
    status_print : boolean, default = True
        Print file saving process if set to True.
    
    show_result : boolean, default = True
        Plot detection results on the spectrogram if set to True.
        
    run_detection : boolean, default = True
        Run detection procedures if set to True. 
        
        Set to False will generate one detection covering the entire duration of spectrogram. Only set to False for the purpose of extracting acoustic features.
    
    Attributes
    ----------
    detection : ndarray of shape (detection,2)
        Begin time (the first column) and end time (the second column) of each detection (row).

    result : pandas DataFrame
        A table contains time and frequency boundaries of regions of interest.

    Examples
    --------
    Generate a prewhitened spectrogram and detect high-intensity signals with signal-to-noise ratios higher than 6 dB. Combine consecutive signals with intervals shorter than 0.1 sec for one detection. Only signals with durations ranging between 0.1 and 1 sec are saved.  
    
    >>> from soundscape_IR.soundscape_viewer import audio_visualization
    >>> sound = audio_visualization(filename='audio.wav', path='./wav/', prewhiten_percent=50, plot_type=None)
    >>>
    >>> from soundscape_IR.soundscape_viewer import spectrogram_detection
    >>> sp=spectrogram_detection(sound.data, sound.f, threshold=6, smooth=1, minimum_interval=0.5, minimum_duration=0.1, maximum_duration=1, filename='Detection.txt', path='./save/')
                                
    Detect regions of interest from a spectrogram that is filtered by using a source separation model. Add 0.1-sec padding before and after each detection.
    
    >>> from soundscape_IR.soundscape_viewer import source_separation
    >>> model=source_separation()
    >>> model.load_model(filename='./model/model.mat')
    >>> model.prediction(input_data=sound.data, f=sound.f)
    >>>
    >>> from soundscape_IR.soundscape_viewer import spectrogram_detection
    >>> source_num = 1 # Choose the source for signal detection
    >>> sp=spectrogram_detection(model.separation[source_num-1], model.f, threshold=3, smooth=1, minimum_interval=0.5, pad_size=0.1, filename='Detection.txt', path='./save/')

    """
    def __init__(self, input, f, threshold, smooth=0, minimum_interval=0, minimum_duration=None, maximum_duration=None, pad_size=0, filename='Detection.txt', folder_id=[], path='./', status_print=True, show_result=True, run_detection=True):
        from scipy.ndimage import gaussian_filter
        self.input_type='Spectrogram'
        self.input=input
        self.f=f
        if not run_detection:
            self.detection=np.array([input[0,0],input[-1,0]])[None,:]
        else:
            time_vec=input[:,0]
            data=input[:,1:]
            begin=np.array([])
            remove_list=[]

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
                begin[begin<0]=0
                ending[ending>input[-1,0]]=input[-1,0]

            min_F=np.array([])
            max_F=np.array([])
            snr=np.array([])
            for n in range(len(begin)):
                idx = (time_vec >= begin[n]) & (time_vec <= ending[n])
                f_idx = np.where(np.sum(level_2d[idx,:],axis = 0) > 0)[0]
                min_F = np.append(min_F, f[f_idx[0]])
                max_F = np.append(max_F, f[f_idx[-1]])
                psd=np.mean(data[idx,:], axis=0)
                snr=np.append(snr, np.max(psd))
            self.output=np.vstack([np.arange(len(begin))+1, np.repeat('Spectrogram',len(begin)), np.repeat(1,len(begin)), begin, ending, min_F, max_F, snr]).T
            self.detection=np.vstack((begin, ending)).T
            self.header=['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Maximum SNR (dB)']
            if filename:
                self.save_txt(filename=filename, path=path, folder_id=folder_id, status_print=status_print)
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

    def save_txt(self, filename='Separation.txt', path='./', folder_id=[], status_print=True):
        df = pd.DataFrame(self.output, columns = self.header) 
        self.result = df
        df.to_csv(path+'/'+filename, sep='\t', index=False)
        if status_print:
            print('Successifully save to '+filename)

        if folder_id:
            #import Gdrive_upload
            Gdrive=gdrive_handle(folder_id, status_print=False)
            Gdrive.upload(filename, status_print=False)

    def feature_extraction(self, interval_range=[1, 500], energy_percentile=None, filename='Features.mat', folder_id=[], path='./'):
        """
        This method extracts spectral and temporal features from regions of interest. 
        
        Spectral features are extracted by averaging the power spectral densities across time bins. Temporal features are extracted by performing autocorrelation on the time-domain energy envelope of the input spectrogram data. For sounds with repetitive pulse structure, this method generates a time-lagged autocorrelation function that represents the variation of inter-pulse intervals. 
        
        According to the table of detection results, this method will extract spectral and temporal features for each region of interest.

        Parameters
        ----------
        interval_range : a list of 2 scalars [min, max], default = [1, 500]
            Minimum and maximum time intervals (in milliseconds) for measuring autocorrelation function. 
            
            The maximum time interval should not be greater than ``minimum_duration`` in the procedure of ``spectrogram_detection``.
        
        energy_percentile : None or float > 0, default = None
            Choose a percentile to represent the energy envelope of spectrogram data. 
            
            Use this parameter when a spectrogram contains high-intensity noise. If ``energy_percentile`` is not set, the energy envelope is extracted by averaging power spectral densities across frequencies.
            
        filename : str, default = 'Features.mat'
            Name of the mat file contains feature extraction results.
            
        path : str
            Path to save feature extraction results.

        folder_id : [] or str, default = []
            The folder ID of Google Drive folder for saving feature extraction result.
            
            See https://ploi.io/documentation/database/where-do-i-get-google-drive-folder-id for the detial of folder ID. 

        Attributes
        ----------
        PI : ndarray of shape (autocorrelation function,)
            Array of segment time intervals (in milliseconds).
        
        PI_result : ndarray of shape (detection, autocorrelation function)
            Autocorrelation function(s) of region(s) of interest.
        
        f : ndarray of shape (frequency,)
            Array of sample frequencies (in Hertz).
        
        spectral_result : ndarray of shape (detection, frequency)
            Mean spectrum(s) of region(s) of interest (in dB).

        Examples
        --------
        Detect a list of signals and extract their spectral and temporal features. Restrict the range of autocorrelation functions between 1 and 200 ms. Features are saved in a mat file. 
    
        >>> from soundscape_IR.soundscape_viewer import spectrogram_detection
        >>> sp=spectrogram_detection(sound.data, sound.f, threshold=6, smooth=1, minimum_interval=0.5, minimum_duration=0.2, maximum_duration=1)
        >>> sp.feature_extraction(interval_range=[1, 200], filename='Features.mat')
        
        """
        self.PI=np.array([])
        self.PI_result=np.array([])
        self.spectral_result=np.array([])
        for n in range(0, self.detection.shape[0]):
            detection_list=np.where(((self.input[:,0]>=self.detection[n,0])*(self.input[:,0]<=self.detection[n,1]))==1)[0]
            # Analyze pulse structure
            if self.input_type=='Spectrogram':
                pulse_analysis_result=pulse_interval(self.input[detection_list,:], energy_percentile=energy_percentile, interval_range=interval_range, plot_type=None, standardization=True)
            elif self.input_type=='Waveform':
                event_list=[np.floor(self.detection[n,0]*self.sf), np.ceil(self.detection[n,1]*self.sf)]
                if event_list[0]<0:
                    event_list[0]=0
                if event_list[1]>len(self.x):
                    event_list[1]=len(self.x)
                pulse_analysis_result=pulse_interval(self.x[int(event_list[0]):int(event_list[1])], sf=self.sf, interval_range=interval_range, plot_type=None, standardization=True)
            # Analyze spectral features
            spectral_result=np.mean(self.input[detection_list,1:], axis=0)
            if n==0:
                self.PI_result=pulse_analysis_result.result[None,:]
                self.PI=pulse_analysis_result.PI
                self.spectral_result=spectral_result
            else:
                self.PI_result=np.vstack((self.PI_result, pulse_analysis_result.result[None,:]))
                self.spectral_result=np.vstack((self.spectral_result, spectral_result))
        features=save_parameters()
        features.feature_extraction(self.detection, self.f, self.spectral_result, self.PI, self.PI_result)
        savemat(path+'/'+filename, {'save_features':features})

        # save the result in Gdrive as a mat file
        if folder_id:
            Gdrive=gdrive_handle(folder_id, status_print=False)
            Gdrive.upload(filename, status_print=False)
    
class performance_evaluation:
    def __init__(self, test_spec, label_filename, fpr_control=0.05, plot=True): 
        annotations = pd.read_table(label_filename,index_col=0)
        time_vec=test_spec[:,0]
        self.label=0*time_vec
        for n in range(len(annotations)):
            self.label[((time_vec+np.diff(time_vec[0:2]))>annotations.iloc[n,2])*(time_vec<annotations.iloc[n,3])]=1
        self.level=test_spec[:,1:].max(axis=1)
        if fpr_control>0:
            threshold, fpr, tpr=self.auc(self.label, self.level, fpr_control=fpr_control, plot=plot)

    def f1_confidence(self, label, level, plot=False):
        from sklearn.metrics import f1_score
        f1_threshold=np.linspace(0, np.ceil(np.max(self.level)), num=100)
        f1_result=0*np.linspace(0, np.ceil(np.max(self.level)), num=100)
        for a in np.arange(len(f1_threshold)):
            f1_result[a]=f1_score(label, level>f1_threshold[a], average='binary')
        if plot:
            plt.figure()
            plt.plot(f1_threshold, f1_result)
            plt.xlabel('Threshold')
            plt.ylabel('F1 score')
        return f1_threshold, f1_result

    def precision_recall(self, label, level, plot=False):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(label, level)
        if plot:
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        return precision, recall, thresholds

    def auc(self, label, level, fpr_control=0.05, plot=True):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(label, level)
        auc_score = auc(fpr, tpr)
        threshold = thresholds[(np.abs(fpr - fpr_control)).argmin()]
        if plot:
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('AUC: '+str(np.round(auc_score*100)/100))
            plt.show()
            print('Threshold = '+str(np.round(threshold*100)/100)+' @ '+str(fpr_control)+' FPR')
        return threshold, fpr, tpr

class pulse_interval:
    def __init__(self, data, sf=None, energy_percentile=None, interval_range=[1, 500], plot_type='Both', standardization=True, envelope=True):
        from scipy.signal import hilbert
        if len(data.shape)==2:
            time_vec=data[:,0]
            if energy_percentile:
                data=np.percentile(data[:,1:], energy_percentile, axis=1)
            else:
                data=np.mean(data[:,1:], axis=1)
        elif len(data.shape)==1:
            data=data/np.max(np.abs(data))
            if envelope:
                data=np.abs(hilbert(data))
            time_vec=np.arange(len(data))/sf
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
        if (max(time_vec)-min(time_vec))<1.5*max(interval_range)/1000:
            time_vec=np.arange(time_vec[0],time_vec[0]+1.5*max(interval_range)/1000,time_vec[1]-time_vec[0])
            data_new=np.zeros(time_vec.shape)
            data_new[0:len(data)]=data
            data=data_new
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
    """
    This class applies a local-max detector (Lin et al. 2013) to extract representative frequencies of tonal sounds. 
    
    The local-max detector consists of four steps. First, it uses spectrogram prewhitening to remove long-duration noise and applies a Gaussian filter to smooth the spectrogram. Second, the second derivate of power spectrum is calculated to search spectral peaks. Third, spectral peaks with low signal-to-noise ratios are removed. Finally, a noise filter is applied to remove isolated tonal fragments.
    
    It can be used as a detector for animal vocalizations with evident tonal characteristics, such as dolphin whistles or bird chirps. It can also work as a tonal sound filter to improve source separation performance.
    
    The output is a table containing Time (s), Frequency (Hz), and SNR (dB). The table is saved in a text file.

    Parameters
    ----------
    tonal_threshold : float > 0, default = 0.1
        The threshold of second derivative for binarizing a power spectrum. 
        
        Only frequency bins with second derivative higher than ``tonal_threshold`` are considered as spectral peaks.
    
    temporal_prewhiten : None or float [0, 100), default = 50
        Applying prewhitening method to suppress background noise and convert power spectral densities into signal-to-noise ratios.
        
        In ``tonal_detection``, this parameter is for the spectrogram prewhitening along time axis.
    
    spectral_prewhiten : None or float [0, 100), default = 50
        Applying prewhitening method to highlight spectral peaks with high signal-to-noise ratios.
        
        In ``tonal_detection``, this parameter is for the spectrogram prewhitening along frequency axis.
        
    smooth : float ≥ 0, default = 1
        Standard deviation of Gaussian kernel for smoothing the spectrogram data. 
        
        See ``sigma`` in ``scipy.ndimage.gaussian_filter`` for details.
    
    threshold : float ≥ 0, default = 0
        Energy threshold for binarizing the spectrogram data. 
        
        Only time and frequency bins with intensities higher than ``threshold`` are considered as detections.
    
    noise_filter_width : float ≥ 0 or a list of 2 scalars, default = 3
        Size of the median filter window. 
        
        Elements of kernel_size should be odd. If kernel_size is a scalar, then this scalar is used as the size in each dimension.
        
        See ``kernel_size`` in ``scipy.signal.medfilt2d`` for details.
    
    Returns
    -------
    detection : pandas DataFrame
        A table contains time, frequency, and amplitude of tonal sounds.

    output : ndarray of shape (time, frequency+1)
        Spectrogram of tonal spectral peaks.

        The first column is time, and the subsequent columns are signal-to-noise ratios associated with ``f``.

    Examples
    --------
    Generate a spectrogram and use the local-max detector to search spectral peaks.
    
    >>> from soundscape_IR.soundscape_viewer import audio_visualization
    >>> sound = audio_visualization(filename='audio.wav', path='./wav/', plot_type='Spectrogram')
    >>>
    >>> from soundscape_IR.soundscape_viewer import tonal_detection
    >>> tonal=tonal_detection(tonal_threshold=0.1, temporal_prewhiten=50, spectral_prewhiten=50, smooth=1, threshold=0, noise_filter_width=3)
    >>> output, detection = tonal.local_max(sound.data, sound.f, filename='Tonal.txt', path='./', folder_id=[])
    
    References
    ----------
    .. [1] Lin, T.-H., Chou, L.-S., Akamatsu, T., Chan, H.-C., & Chen, C.-F. (2013). An automatic detection algorithm for extracting the representative frequency of cetacean tonal sounds. Journal of the Acoustical Society of America, 134: 2477-2485. https://doi.org/10.1121/1.4816572

    """
    def __init__(self, tonal_threshold=0.1, temporal_prewhiten=50, spectral_prewhiten=50, smooth=1, threshold=0, noise_filter_width=3):
        self.tonal_threshold=tonal_threshold
        self.temporal_prewhiten=temporal_prewhiten
        self.spectral_prewhiten=spectral_prewhiten
        self.threshold=threshold
        self.smooth=smooth
        self.noise_filter_width=noise_filter_width
    
    def local_max(self, input, f, axis=1, filename='Tonal.txt', path='./', folder_id=[]):
        """
        Run local-max detector and save tonal detection results.
        
        Parameters
        ----------
        input : ndarray of shape (time, frequency+1)
            Spectrogram data for analysis. 
            
            The first column is time, and the subsequent columns are power spectral densities associated with ``f``. Use the same spectrogram format generated from ``audio_visualization``.
    
        f : ndarray of shape (frequency,)
            Frequency of the input spectrogram data.
    
        filename : str, default = 'Tonal.txt'
            Name of the text file contains tonal sound detections.
            
        path : str
            Path to save tonal detection results.

        folder_id : [] or str, default = []
            The folder ID of Google Drive folder for saving tonal detection result.
            
            See https://ploi.io/documentation/database/where-do-i-get-google-drive-folder-id for the detial of folder ID. 
        """
        
        # Do vertical and horizontal prewhitening
        temp0=copy.deepcopy(input[:,1:])
        if self.temporal_prewhiten:
            temp0, _=matrix_operation.prewhiten(temp0, prewhiten_percent=self.temporal_prewhiten, axis=0)
        if self.spectral_prewhiten:
            temp0, _=matrix_operation.prewhiten(temp0, prewhiten_percent=self.spectral_prewhiten, axis=1)

        temp0[temp0<0]=0

        # Smooth the spectrogram
        from scipy.ndimage import gaussian_filter
        from scipy.signal import medfilt2d
        temp0=gaussian_filter(temp0, sigma=self.smooth)

        # Applying local-max detector to extract whistle contours
        temp=(-1*np.diff(temp0,n=2,axis=axis))>self.tonal_threshold
        if axis==1:
            temp=np.hstack((np.zeros([temp.shape[0],1]),temp))
            temp=np.hstack((temp,np.zeros([temp.shape[0],1])))
        elif axis==0:
            temp=np.vstack((np.zeros([1,temp.shape[1]]),temp))
            temp=np.vstack((temp,np.zeros([1,temp.shape[1]])))
        temp2=temp*temp0
        temp2[temp2<0]=0
        output=np.hstack((input[:,0:1], temp2))

        # Produce detection result
        if self.threshold>=0:
            temp2=medfilt2d(temp2, kernel_size=self.noise_filter_width)
            rc=np.nonzero(temp2>self.threshold)
            amp=temp2.flatten()
            amp=amp[np.where((amp>self.threshold))[0]]
            detection=pd.DataFrame(np.hstack((input[rc[0],0:1], input[rc[0],0:1], f[rc[1]][:,None], f[rc[1]][:,None], amp[:,None])), columns = ['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)','SNR (dB)'])
            if filename:
                detection.to_csv(path+'/'+filename, sep='\t', index=False)
                if folder_id:
                    #import Gdrive_upload
                    Gdrive=gdrive_handle(folder_id, status_print=False)
                    Gdrive.upload(filename, status_print=False)
        else:
            detection=np.array([])

        return output, detection
