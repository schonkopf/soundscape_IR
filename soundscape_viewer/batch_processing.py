import numpy as np
import pandas as pd

class batch_processing:
    """
    Batch processing a large amount of acoustic data.
    
    This class integrates a set of methods supported by soundscape_IR to automatically process a large amount of soundscape recordings. At first, copy all the WAVE files collected from one recorder (presumably made at the same location) to a folder. Then, define analysis procedures and associated parameters. Finally, run the program to initiate analysis procedures. 
    
    There are three types of outputs:
    
    - Spectrogram detection results are saved in text files (using the format of Raven software). 
    
    - Feature extraction results are saved in mat files.
    
    - Basis functions learned in source separation procedures are saved in mat files.

    Parameters
    ----------
    folder : str
        Folder path for analysis.
    
    folder_id : str
        The folder ID of Google Drive folder for analysis.
        
        See https://ploi.io/documentation/database/where-do-i-get-google-drive-folder-id for the detial of folder ID. 
    
    file_extension : {'.wav', '.WAV', '.mat', '.txt'}, default = '.wav'
        Data format for analysis.
    
    annotation_extension : str, default = None
        Extended file name of the associated txt files contain annotations (generated by using Raven software). 
        
        For example, if our recording files are named using the structure of Location_Date_Time.wav and txt files are Location_Date_Time.Table.1.selections.txt, please set ``annotation_extension='.Table.1.selections.txt'``
    
        
    Examples
    --------
    Combine manual annotations of multiple recordings to generate a concatenated spectrogram for model training.
    
    >>> from soundscape_IR.soundscape_viewer import batch_processing
    >>> batch = batch_processing(folder='./data/', annotation_extension='.Table.1.selections.txt')
    >>> batch.params_spectrogram(FFT_size=1024, prewhiten_percent=50, time_resolution=0.1, padding=0.5, folder_combine=True)
    >>> batch.run()
    >>> 
    >>> from soundscape_IR.soundscape_viewer import matrix_operation
    >>> matrix_operation().plot_lts(batch.spectrogram, batch.f, lts=False)
    
    Apply an energy detector to identify regions of interest.
    
    >>> from soundscape_IR.soundscape_viewer import batch_processing
    >>> batch = batch_processing(folder='./data/')
    >>> batch.params_spectrogram(FFT_size=1024, environment='wat', prewhiten_percent=50, time_resolution=0.1)
    >>> batch.params_spectrogram_detection(threshold=6, smooth=1, minimum_interval=0.5, path='./data/txt')
    >>> batch.run()
    
    Run adaptive and semi-supervised source separation.
    
    >>> from soundscape_IR.soundscape_viewer import source_separation
    >>> model=source_separation(filename='model.mat')
    >>> 
    >>> from soundscape_IR.soundscape_viewer import batch_processing
    >>> batch = batch_processing(folder='./data/)
    >>> batch.params_spectrogram(FFT_size=512, time_resolution=0.1, window_overlap=0.5, prewhiten_percent=25, f_range=[0,8000])
    >>> batch.params_separation(model, adaptive_alpha=[0,0.2], additional_basis = 2, save_basis=True, path='./data/mat/')
    >>> batch.run()

    """
    def __init__(self, folder=[], folder_id=[], file_extension='.wav', annotation_extension=None, annotation_folder=None, query_list=None):
        if folder:
            if not annotation_folder:
                annotation_folder=folder
            self.collect_folder(folder, file_extension)
        elif folder_id:
            self.Gdrive, self.link, self.audioname=self.collect_Gdrive(folder_id, file_extension, query_list)
            if annotation_extension:
                if not annotation_folder:
                    annotation_folder=folder_id
                self.Gdrive_selections, _, _=self.collect_Gdrive(annotation_folder, '.txt')
        self.Raven_selections=annotation_extension
        self.Raven_folder=annotation_folder
        self.run_spectrogram=False  
        self.run_separation=False
        self.run_detection=False
        self.run_feature_extraction=False
        self.run_tonal=False
        self.run_lts=False
        self.run_adaptive_prewhiten=False
        self.run_load_result=False

    def collect_folder(self, path, file_extension='.wav'):
        import os
        file_list=os.listdir(path)
        file_list.sort(key = str.lower)
        self.link=path
        self.cloud=0   
        self.audioname=np.array([], dtype=object)
        for filename in file_list:
            if filename.endswith(file_extension):
                self.audioname = np.append(self.audioname, filename)
        print('Identified ', len(self.audioname), 'files')
    
    def collect_Gdrive(self, folder_id, file_extension='.wav', query_list=None):
        from soundscape_IR.soundscape_viewer.utility import gdrive_handle
        from natsort import index_natsorted
        if not query_list:
            Gdrive=gdrive_handle(folder_id)
            Gdrive.list_query(file_extension=file_extension)
        else:
            Gdrive=query_list
        self.cloud=2
        link=np.array([], dtype=object)
        audioname=np.array([], dtype=object)
        for file in Gdrive.file_list:
            link=np.append(link, file['alternateLink'])
            audioname=np.append(audioname, file['title'])
        idx = index_natsorted(audioname)
        audioname = audioname[idx]
        Gdrive.file_list = np.array(Gdrive.file_list)[idx]
        print('Identified ', len(audioname), 'files')
        return Gdrive, link, audioname

    def params_spectrogram(self, channel=1, offset_read=0, FFT_size=512, environment='wat', time_resolution=None, window_overlap=0, f_range=[], prewhiten_percent=0, padding=0, folder_combine=False, mel_comp=0, sensitivity=0):
        """
        Define spectrogram parameters.

        See ``soundscape_IR.soundscape_viewer.utility.audio_visualization`` for details.
        
        """
        self.run_spectrogram = True
        self.channel = channel
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
        """
        Define parameters for adaptive prewhitening.

        Apply this method to prewhien spectrograms via time weighted running average of noise spectrum. See Lin et al. (2013) for details.
        
        Parameters
        ----------
        eps : float, default = 0.1
            Ratio to update the weighted running average of power spectral density in each time bin (i). 
            
            Running average of noise is estimated independently in each frequency bin (f).
            
            .. math:: N =  (1-eps)*P_{f,i} + eps*P_{f,i+1}
        
        smooth : float ≥ 0, default = 1
            Standard deviation of Gaussian kernel for smoothing the spectrogram data. 
            
            See ``sigma`` in ``scipy.ndimage.gaussian_filter`` for details.
            
        continuous_adaptive : boolean, default = True
            Set to True when audio recordings are collected in time series. 
            
            Noise estimated from the i_th file will be passed to i+1_th file.
            
        References
        ----------
        .. [1] Lin, T.-H., Chou, L.-S., Akamatsu, T., Chan, H.-C., & Chen, C.-F. (2013) An automatic detection algorithm for extracting the representative frequency of cetacean tonal sounds. Journal of the Acoustical Society of America, 134: 2477-2485. https://doi.org/10.1121/1.4816572
        
        """
        self.run_adaptive_prewhiten=True
        self.eps=eps
        self.adaptive_smooth=smooth
        self.continuous_adaptive=continuous_adaptive

    def params_separation(self, model, iter=50, adaptive_alpha=0, additional_basis=0, save_basis=False, folder_id=[], path='./'):
        """
        Define parameters for the prediction phase of source separation. 
        
        See ``soundscape_IR.soundscape_viewer.source_separation.source_separation`` for details.
        
        Set ``save_basis`` to True for saving basis functions learned in the prediction phase (when adaptive or semi-supervised source separation are enabled). Define the ``folder_id`` or ``path`` to save basis functions as mat files.
        
        """
        self.run_separation = True
        self.model = model
        self.iter = iter
        self.adaptive_alpha = adaptive_alpha
        self.additional_basis = additional_basis
        self.save_basis = save_basis
        self.save_basis_folder_id = folder_id
        self.save_basis_path = path

    def params_spectrogram_detection(self, source=0, threshold=6, smooth=0, minimum_interval=0, minimum_duration = None, maximum_duration=None, pad_size=0, folder_id=[], path='./', show_result=False, save_clip_path=None):
        """
        Define parameters for spectrogram-based sound detection. 
        
        See ``soundscape_IR.soundscape_viewer.utility.spectrogram_detection`` for details.
        
        Define the ``folder_id`` or ``path`` to save detection results as txt files. Set ``show_result`` to True for directly viewing the detection results (WARNING: this may generate a large number of figures). 
        
        """
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
        self.save_clip_path = save_clip_path
  
    def params_lts_maker(self, source=1, time_resolution=[], dateformat='yyyymmdd_HHMMSS', initial=[], year_initial=0, filename='Separation_LTS.mat', folder_id=[], path='./'):
        """
        Define parameters for making a long-term spectrogram. 
        
        See ``soundscape_IR.soundscape_viewer.soundscape_viewer.lts_maker`` for details.
        
        Enter the filename information to extract the recording date and time from each audio file. For example, if the file name is 'KT08_20171118_123000.wav', please enter ``initial='KT08_'`` and ``dateformat='yyyymmdd_HHMMSS'``. 
        
        If the file name does not contain information of century (when ``dateformat='yymmdd_HHMMSS'``), specify century in ``year_initial``. For example, the recording 'KT08_171118_123000.wav' was made in 2017, then set ``year_initial=2000``.
        
        Define the ``folder_id`` or ``path`` to save the long-term spectrogram as a mat file. 
        
        """
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

    def params_feature_extraction(self, source=0, energy_percentile=None, interval_range=[1, 500], waveform_extraction=False, lts_maker=False, folder_id=[], path='./'):
        """
        Define feature extraction parameters. 
        
        See the method ``feature_extraction`` of ``soundscape_IR.soundscape_viewer.utility.spectrogram_detection`` for details.
        
        Define the ``folder_id`` or ``path`` to save feature extraction results as mat files. 
        
        """
        if lts_maker:
            self.run_feature_extraction=False
        else:
            self.run_feature_extraction=True
        self.feature_source=source
        self.energy_percentile=energy_percentile
        self.interval_range=interval_range
        self.save_feature_folder_id=folder_id
        self.save_feature_path=path
        self.waveform_extraction=waveform_extraction

    def params_tonal_detection(self, source=0, tonal_threshold=0.5, smooth=1.5, threshold=3, folder_id=[], path='./'):
        """
        Define parameters for tonal sound extraction.

        See ``soundscape_IR.soundscape_viewer.utility.tonal_detection`` for details.
        
        Define the ``folder_id`` or ``path`` to save detection results as txt files. 
        
        """
        from soundscape_IR.soundscape_viewer import tonal_detection
        self.run_tonal=True
        self.tonal_source=source
        self.local_max=tonal_detection(tonal_threshold=tonal_threshold, smooth=smooth, threshold=threshold)
        self.tonal_folder_id=folder_id
        self.tonal_path=path
  
    def params_load_result(self, data_type='basis', initial=[], dateformat='yyyymmdd_HHMMSS', year_initial=0): 
        """
        Define parameters for loading analysis results.
        
        Currently, only support mat files containing basis funcitons and feature extraction results, and text files containing detection results. Please choose ``data_type`` from {'basis', 'feature', 'detection'}.
        
        See ``soundscape_IR.soundscape_viewer.soundscape_viewer.lts_maker`` for details. 
        
        Parameters
        ----------
        data_type : {'basis', 'feature', 'detection'}, default = 'basis'
        
        Attributes
        ----------
        model
            See ``soundscape_IR.soundscape_viewer.source_separation`` for details.
            
        detection_result
            See ``soundscape_IR.soundscape_viewer.spectrogram_detection`` for details.
            
        PI, PI_result, f, spectral_result
            See ``soundscape_IR.soundscape_viewer.spectrogram_detection.feature_extraction`` for details.
            
        Examples
        --------
        Use batch_processing to combine the entire set of newly learned basis functions.

        >>> from soundscape_IR.soundscape_viewer import batch_processing
        >>> batch = batch_processing(folder='./data/mat/', file_extension='.mat')
        >>> batch.params_load_result(data_type='basis', initial='Location_', dateformat='yyyymmdd_HHMMSS')
        >>> batch.run()
        >>>
        >>> # Plot the basis functions learned by using semi-supervised source separation
        >>> batch.model.plot_nmf(plot_type='W', source=3)
        
        """
        if data_type=='basis':
            self.run_load_result = 1
        elif data_type=='feature':
            self.run_load_result = 2
        elif data_type=='detection':
            self.run_load_result = 3
        self.format_initial = initial
        self.dateformat = dateformat
        self.year_initial = year_initial
        self.time_vec=np.array([])

    def run(self, start=1, num_file=None):
        """
        Run batch processing procedures.
        
        Parameters
        ----------
        start : int ≥ 1, default = 1
            The file number to begin batch analysis.
        
        num_file : None or int ≥ 1, default = None
            The number of files after ``start`` for batch analysis.
        
        """
        from soundscape_IR.soundscape_viewer import lts_maker
        from soundscape_IR.soundscape_viewer import audio_visualization
        from soundscape_IR.soundscape_viewer import spectrogram_detection
        from soundscape_IR.soundscape_viewer import pulse_interval
        from soundscape_IR.soundscape_viewer import matrix_operation
        from soundscape_IR.soundscape_viewer import source_separation

        import datetime
        import copy
        import os
        if self.cloud==1:
            import urllib.request

        self.start = start-1
        if not num_file:
            num_file=len(self.audioname)-self.start
        run_list=range(self.start, self.start+num_file)
        folder_data=np.array([])

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
                        selections_filename = self.Raven_folder + '/' + self.audioname[file][:-4]+self.Raven_selections
                    print('Processing file no. '+str(file+1)+' :'+temp+', in total: '+str(num_file)+' files', flush=True, end='')

                if self.Raven_selections:
                    audio = audio_visualization(self.audioname[file], path=path, channel=self.channel, FFT_size=self.FFT_size, 
                                                time_resolution=self.time_resolution, window_overlap=self.window_overlap, 
                                                f_range = self.f_range, sensitivity=self.sensitivity,
                                                environment=self.environment, plot_type=None, prewhiten_percent=self.prewhiten_percent,
                                                annotation = selections_filename, padding = self.annotation_padding, mel_comp=self.mel_comp)
                else:
                    audio = audio_visualization(self.audioname[file], path=path, channel=self.channel, FFT_size=self.FFT_size,
                                                time_resolution=self.time_resolution, window_overlap=self.window_overlap, 
                                                offset_read=self.offset_read, f_range = self.f_range, sensitivity=self.sensitivity,
                                                environment=self.environment, plot_type=None, 
                                                prewhiten_percent=self.prewhiten_percent, mel_comp=self.mel_comp)
                if self.folder_combine:
                    if len(audio.data)>0:
                        if len(folder_data)==0:
                            folder_data=audio.data
                        else:
                            folder_data=np.vstack((folder_data, audio.data))
                            folder_data[:,0]=np.arange(folder_data.shape[0])*(folder_data[1,0]-folder_data[0,0])  
                            self.spectrogram=folder_data
                            self.f=audio.f
                else:
                    if self.run_adaptive_prewhiten:
                        if file==self.start:
                            audio.data[:,1:], ambient=matrix_operation.adaptive_prewhiten(audio.data[:,1:], prewhiten_percent=50, 
                                                                                          axis=0, eps=self.eps, smooth=self.adaptive_smooth)
                        else:
                            if not self.continuous_adaptive:
                                ambient=None
                            audio.data[:,1:], ambient=matrix_operation.adaptive_prewhiten(audio.data[:,1:], axis=0, noise_init=ambient,
                                                                                          eps=self.eps, smooth=self.adaptive_smooth)
                audio.data[np.isnan(audio.data)]=0

            if self.run_separation:
                model = copy.deepcopy(model_backup)
                model.prediction(audio.data, audio.f, iter = self.iter, 
                                 adaptive_alpha = self.adaptive_alpha, additional_basis = self.additional_basis)
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
                    lts.compress_spectrogram(10**(model.separation[self.lts_source-1][:,1:]/10), 
                                             model.separation[self.lts_source-1][:,0], self.lts_time_resolution, 
                                             linear_scale=True, interval_range=self.interval_range,
                                             energy_percentile=self.energy_percentile)

                if self.run_detection:
                    for n in range(0, len(self.source)):
                        filename=self.audioname[file][:-4]+'_S'+str(self.source[n])+'.txt'
                        sp=spectrogram_detection(model.separation[self.source[n]-1], model.f, threshold=self.threshold[n],
                                                 smooth=self.smooth[n], minimum_interval=self.minimum_interval[n],
                                                 minimum_duration=self.minimum_duration[n], maximum_duration=self.maximum_duration[n],
                                                 pad_size=self.padding[n], filename=filename, folder_id=self.detection_folder_id,
                                                 path=self.detection_path, status_print=False, show_result=self.show_result)
                        if self.save_clip_path:
                            if sp.detection.shape[0]>0:
                                audio = audio_visualization(self.audioname[file], path=path, channel=self.channel, sensitivity=self.sensitivity,
                                                environment=self.environment, plot_type=None,
                                                annotation = self.detection_path+'/'+filename, save_clip_path=self.save_clip_path)

            if self.run_tonal:
                if self.run_separation:
                    filename=self.audioname[file][:-4]+'_S'+str(self.tonal_source[n])+'.txt'
                    _,_=self.local_max.local_max(model.separation[self.tonal_source-1], model.f, 
                                                 filename=filename, folder_id=self.tonal_folder_id, path=self.tonal_path)
                else:
                    filename=self.audioname[file][:-4]+'.txt'
                    _,_=self.local_max.local_max(audio.data, audio.f, 
                                                 filename=filename, folder_id=self.tonal_folder_id, path=self.tonal_path)

            if self.run_detection:
                if not self.source[0]:
                    filename=self.audioname[file][:-4]+'.txt'
                    sp=spectrogram_detection(audio.data, audio.f, threshold=self.threshold[0], smooth=self.smooth[0], 
                                             minimum_interval=self.minimum_interval[0], minimum_duration=self.minimum_duration[0], 
                                             maximum_duration=self.maximum_duration[0], pad_size=self.padding[0], 
                                             filename=filename, folder_id=self.detection_folder_id, path=self.detection_path, 
                                             status_print=False, show_result=self.show_result)
            else:
                if self.run_feature_extraction:
                    sp=spectrogram_detection(audio.data, audio.f, threshold=0, show_result=False, status_print=False, run_detection=False)

            if self.run_feature_extraction:
                if self.run_separation:
                    filename=self.audioname[file][:-4]+'_S'+str(self.feature_source)+'.mat'
                    if self.waveform_extraction:
                        audio.FFT_size=np.round(audio.FFT_size/(audio.sf/np.round(audio.f[-1]*2)))
                        audio.sf=np.round(audio.f[-1]*2)
                        audio.convert_audio(model.separation[self.feature_source-1])
                        sp.x=np.append(np.zeros((int(np.round(self.offset_read*audio.sf)),1)), audio.xrec)
                        sp.sf=audio.sf
                        sp.input_type='Waveform'
                else:
                    filename=self.audioname[file][:-4]+'.mat'
                sp.feature_extraction(interval_range=self.interval_range, energy_percentile=self.energy_percentile,
                                      filename=filename, folder_id=self.save_feature_folder_id, path=self.save_feature_path)

            if self.run_load_result>0:
                from scipy.io import loadmat
                if self.cloud==2:
                    temp = self.Gdrive.file_list[file]
                    temp.GetContentFile(temp['title'])
                    temp = temp['title']
                    path='.'
                else:
                    temp = self.audioname[file]
                    if hasattr(self,'save_basis_path'):
                        path = self.save_basis_path
                    else:
                        path = self.link
                print('Processing file no. '+str(file+1)+' :'+temp+', in total: '+str(num_file)+' files', flush=True, end='')

                if self.dateformat:
                    if file==self.start:
                        format_idx = lts_maker()
                        format_idx.filename_check(dateformat=self.dateformat, initial=self.format_initial, 
                                                year_initial=self.year_initial, filename=temp)
                    format_idx.get_file_time(temp)
                    str2ord=format_idx.time_vec/24/3600
                else:
                    if len(temp[len(self.format_initial):-4])==0:
                        str2ord=0
                    else:
                        str2ord=int(temp[len(self.format_initial):-4])
                    
                if self.run_load_result==1:
                    temp_model = source_separation()
                    temp_model.load_model(path+'/'+temp, model_check = False)
                    if file==self.start:
                        self.model = copy.deepcopy(temp_model)
                        self.model.time_vec = str2ord*np.ones((temp_model.W.shape[1],))
                    else:
                        self.model.W = np.concatenate((self.model.W, temp_model.W), axis=1)
                        self.model.W_cluster = np.concatenate((self.model.W_cluster, temp_model.W_cluster), axis=None)
                        self.model.time_vec = np.concatenate((self.model.time_vec,str2ord*np.ones((temp_model.W.shape[1],))), axis=None)
                elif self.run_load_result==2:
                    data = loadmat(path+'/'+temp)
                    if len(data['save_features']['PI'].item())>0:
                        if len(self.time_vec)==0:
                            self.PI = np.array(data['save_features']['PI'].item()[0])
                            self.PI_result = data['save_features']['PI_result'].item()
                            self.f = np.array(data['save_features']['f'].item()[0])
                            self.spectral_result = data['save_features']['spectral_result'].item()
                            self.time_vec = data['save_features']['detection'].item()[:,0]+str2ord*24*3600
                            if 'embedding_result' in data['save_features'].dtype.names:
                                self.embedding_result = data['save_features']['embedding_result'].item()
                        else:
                            self.time_vec = np.append(self.time_vec, data['save_features']['detection'].item()[:,0]+str2ord*24*3600)
                            self.PI_result = np.vstack((self.PI_result, data['save_features']['PI_result'].item()))
                            self.spectral_result = np.vstack((self.spectral_result, data['save_features']['spectral_result'].item()))
                            if 'embedding_result' in data['save_features'].dtype.names:
                                self.embedding_result = np.vstack((self.embedding_result, data['save_features']['embedding_result'].item()))
                elif self.run_load_result==3:
                    df = pd.read_table(path+'/'+temp,index_col=0) 
                    df[['Begin Time (s)', 'End Time (s)']]=df[['Begin Time (s)', 'End Time (s)']]+str2ord*24*3600
                    if file==self.start:
                        self.detection_result = copy.deepcopy(df)
                    else:
                        self.detection_result = pd.concat([self.detection_result, df])

            if self.cloud>=1:
                os.remove(self.audioname[file])
                if self.Raven_selections:
                    os.remove(temp2['title'])

        if self.run_load_result==1:
            if self.dateformat:
                temp = np.argsort(self.model.time_vec)
                self.model.W=self.model.W[:,temp]
                self.model.W_cluster=self.model.W_cluster[temp]
                self.model.time_vec=self.model.time_vec[temp]
        elif self.run_load_result==2:
            temp = np.argsort(self.time_vec)
            self.PI_result=self.PI_result[temp,:]
            self.spectral_result=self.spectral_result[temp,:]
            self.time_vec=self.time_vec[temp]/24/3600
            if 'embedding_result' in data['save_features'].dtype.names:
                self.embedding_result = self.embedding_result[temp,:]
        elif self.run_load_result==3:
            self.detection_result[['Begin Time (s)', 'End Time (s)']]=self.detection_result[['Begin Time (s)', 'End Time (s)']]/24/3600
            self.detection_result.sort_values(by=['Begin Time (s)'],inplace=True)
            self.detection_result.reset_index(inplace=True, drop=True)

        if self.run_lts:
            if self.lts_folder_id:
                lts.save_lts(self.lts_filename, self.lts_folder_id, status_print=False)
            else:
                lts.save_lts(self.save_lts_path+'/'+self.lts_filename, status_print=False)
                
