import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from scipy.io import savemat
from scipy.fftpack import fft
from sklearn.decomposition import non_negative_factorization as NMF
from sklearn.decomposition._nmf import _initialize_nmf as NMF_init
from sklearn.decomposition._nmf import _update_coordinate_descent as basis_update
from sklearn.utils import check_array
from .utility import save_parameters
from .utility import gdrive_handle
import datetime

class pcnmf:
    """ Periodicity-coded non-negative matrix factorization
    
    Periodicity coded non-negative matrix factorization (PC-NMF) for separating sound sources displaying unique periodicity patterns. PC-NMF first decomposes a spectrogram into two matrices: spectral basis matrix and encoding matrix. Next, using the encoding periodicity information, the spectral bases belonging to the same source are grouped together. Finally, distinct sources are reconstructed on the basis of the cluster of the basis matrix and the corresponding encoding information, and the noise components are then removed.
    
    Parameters
    ----------
    sparseness : float ≥ 0

    feature_length : int ≥ 1

    basis_num : int ≥ 1

    Please check scikit-learn/sklearn/decomposition/nmf.py for the explaination of 
    the other parameters (initial, alpha, beta_loss, solver)

    Examples
    --------
    Train a PC-NMF model and use the model to analyze a long-term spectrogram. 
    
    >>> model=pcnmf(sparseness=0, feature_length=6)
    >>> input_data, f=LTS.input_selection('median')
    >>> model.unsupervised_separation(input_data, f, source_num=2)       
    >>> model.save_model(filename='LTS_median_pcnmf.mat')
    >>> 
    >>> separation_result=pcnmf()
    >>> separation_result.load_model(filename='LTS_median_pcnmf.mat')
    >>> separation_result.supervised_separation(input_data=LTS.input_selection('median'), f=LTS.f, iter=50)
    
    References
    ----------
    .. [1] Tzu-Hao Lin, Shih-Hua Fang, Yu Tsao. (2017) Improving biodiversity assessment via unsupervised separation of biological sounds from long-duration recordings. Scientific Reports, 7: 4547.
    

    """
    def __init__(self, sparseness=0, feature_length=1, basis_num=60, 
               initial='random', alpha=0, beta_loss='frobenius', solver='cd'):
        self.sparseness=sparseness
        self.feature_length=feature_length
        self.basis_num=basis_num
        self.initial=initial
        self.alpha=alpha
        self.beta_loss=beta_loss
        self.solver=solver
    
    def unsupervised_separation(self, input_data, f, source_num=2, iter=200):    
        self.source_num=source_num
        self.f=f
        self.time_vec=input_data[:,0:1]
        input_data=input_data[:,1:].T
        baseline=input_data.min()
        input_data=input_data-baseline

        # Modify the input data based the feature width
        data=self.matrix_conv(input_data)

        # 1st NMF for feature learning
        self.W, self.H, n_iter = NMF(data, n_components=self.basis_num, init=self.initial, 
                                update_H=True, solver=self.solver, beta_loss=self.beta_loss,
                                alpha_W=self.alpha, l1_ratio=self.sparseness, max_iter=iter) 

        # Transform H into periodicity
        H_std = np.subtract(self.H, np.transpose(np.matlib.repmat(np.mean(self.H, axis=1), self.H.shape[1],1)))
        P = abs(fft(H_std))
        P2=P#[:,1:100]
        P2=P2-P2.min()

        # 2nd NMF for feature clustering
        W_p, H_p, n_iter = NMF(np.transpose(P2), n_components=source_num, init=self.initial, 
                               update_H=True, solver=self.solver, beta_loss=self.beta_loss, 
                               alpha_H=self.alpha, l1_ratio=0.5, max_iter=iter) 

        # Get cluster labels for individual sources
        self.W_cluster = np.argmax(H_p, axis=0)

        # Reconstruct individual sources
        self.pcnmf_output(input_data, self.time_vec, baseline)
        self.time_vec=self.time_vec[:,0]
        return self.W, self.H, self.W_cluster
    
    def matrix_conv(self, input_data):
        matrix_shape=input_data.shape
        data=np.zeros((matrix_shape[0]*self.feature_length, matrix_shape[1]-1+self.feature_length))
        for x in range(self.feature_length):
            data[(self.feature_length-(x+1))*matrix_shape[0]:(self.feature_length-x)*matrix_shape[0],x:matrix_shape[1]+x]=input_data
        return data

    def pcnmf_output(self, data, time_vec, baseline=0):
        self.original_level = 10*np.log10((10**((data+baseline).T[:,1:]/10)).sum(axis=1))
        separation=np.zeros(self.source_num, dtype=object)
        relative_level=np.zeros(self.source_num, dtype=object)
        matrix_shape=data.shape
        source0 = np.dot(self.W, self.H)
        for run in range(self.source_num):
            source = np.dot(self.W[:,self.W_cluster==run],self.H[self.W_cluster==run,:])
            mask=np.divide(source,source0)
            temp=np.zeros((matrix_shape))
            for x in range(self.feature_length):
                temp=temp+mask[(self.feature_length-(x+1))*matrix_shape[0]:(self.feature_length-x)*matrix_shape[0],x:matrix_shape[1]+x]
                #temp=temp+mask[x*matrix_shape[0]:(x+1)*matrix_shape[0],x:matrix_shape[1]+x]
            mask=np.divide(temp,self.feature_length)
            mask[np.isnan(mask)]=0
            separation[run] = np.hstack((time_vec, np.multiply(data,mask).T+baseline))
            relative_level[run] = 10*np.log10((10**(separation[run][:,1:]/10)).sum(axis=1))

        self.separation=separation
        self.relative_level=relative_level
  
    def output_selection(self, source=1, begin_date=[], end_date=[], f_range=[]):
        source=source-1
        output_data=self.separation[source]

        # f_range: Hz
        if f_range:
            f_list=(self.f>=min(f_range))*(self.f<=max(f_range))
            f_list=np.where(f_list==True)[0]
        else:
            f_list=np.arange(len(self.f))

        f=self.f[f_list]
        f_list=np.concatenate([np.array([0]), f_list+1])

        # format of begin_data: yyyymmdd
        if begin_date:
            yy=int(begin_date[0:4])
            mm=int(begin_date[4:6])
            dd=int(begin_date[6:8])
            date=datetime.datetime(yy,mm,dd)
            begin_time=date.toordinal()+366
            list=output_data[:,0:1]>=begin_time
            if end_date:
                yy=int(end_date[0:4])
                mm=int(end_date[4:6])
                dd=int(end_date[6:8])
                date=datetime.datetime(yy,mm,dd)
                end_time=date.toordinal()+366+1
            else:
                end_time=begin_time+1
            list=list*(output_data[:,0:1]<end_time)
            list=np.where(list==True)[0]
        else:
            list=np.arange(output_data.shape[0])
        output_data=output_data[:,f_list]
        output_data=output_data[list,:]

        return output_data, f
                          
    def plot_pcnmf(self, source=1):
        source=source-1
        data=self.separation[source]
        if source<self.source_num:
            W=self.W[:,self.W_cluster==source]
            if W.shape[0]>len(self.f):
                W=np.vstack((np.zeros((len(self.f),sum(self.W_cluster==source))), W)).T.reshape((1,-1))
                W=W.reshape((-1,len(self.f))).T
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 10))
            im = ax1.imshow(W, origin='lower',  aspect='auto', cmap=cm.jet,
                            extent=[0, sum(self.W_cluster==source),
                                    np.min(self.f), np.max(self.f)], interpolation='none')

            ax1.set_title('Features')
            ax1.set_ylabel('Frequency')
            ax1.set_xlabel('Basis')
            cbar1 = fig.colorbar(im, ax=ax1)
            #cbar1.set_label('Relative amplitude')

            im2 = ax2.imshow(data[:,1:].T,
                             origin='lower',  aspect='auto', cmap=cm.jet,
                             extent=[0, data.shape[0], np.min(self.f), np.max(self.f)], interpolation='none')

            ax2.set_title('Separation result')
            ax2.set_ylabel('Frequency')
            ax2.set_xlabel('Samples')
            cbar2 = fig.colorbar(im2, ax=ax2)
            #cbar2.set_label('Relative amplitude')
        else:
            print('Higher than the number of separated sources, choose a smaller number.')
      
    def save_model(self, filename='NMF_model.mat', folder_id=[]):
        #import save_parameters
        nmf_model=save_parameters()
        nmf_model.pcnmf(self.f, self.W, self.W_cluster, self.source_num, self.feature_length, self.basis_num, self.sparseness)
        savemat(filename, {'save_nmf':nmf_model})

        # save the result in Gdrive as a mat file
        if folder_id:
            Gdrive=gdrive_handle(folder_id, status_print=False)
            Gdrive.upload(filename, status_print=False)
      
    def model_check(self, model):
        print('Model parameters:')
        intf=model['save_nmf']['f'].item()[0][1]-model['save_nmf']['f'].item()[0][0]
        print('Minimum and maximum frequency:', min(model['save_nmf']['f'].item()[0]), 'Hz and', max(model['save_nmf']['f'].item()[0]), 'Hz')
        print('Frequency resolution:' ,intf, 'Hz')
        print('Feature length:' ,self.feature_length)
        print('Number of basis functions:' ,self.basis_num)
        print('Number of sources:' ,self.source_num)
        if np.any(np.array(model['save_nmf'][0].dtype.names)=='sparseness'):
            print('Sparseness:', self.sparseness)

    def load_model(self, filename, model_check=True):
        model = loadmat(filename)
        self.W=model['save_nmf']['W'].item()
        self.W_cluster=model['save_nmf']['W_cluster'].item()[0]
        self.source_num=model['save_nmf']['k'].item()[0][0]
        self.feature_length=model['save_nmf']['time_frame'].item()[0][0]
        self.basis_num=model['save_nmf']['basis_num'].item()[0][0]
        if np.any(np.array(model['save_nmf'][0].dtype.names)=='sparseness'):
            self.sparseness=model['save_nmf']['sparseness'].item()[0][0]
        if model_check:
            self.model_check(model)
  
    def supervised_separation(self, input_data, f, iter=50):
        self.f=f    
        self.time_vec=input_data[:,0:1]
        input_data=input_data[:,1:].T
        baseline=input_data.min()
        input_data=input_data-baseline

        # Modify the input data based on the feature length
        data=self.matrix_conv(input_data)

        # supervised NMF
        W, H = NMF_init(data, self.basis_num, init='random')

        violation=0
        Ht=H.T
        Ht = check_array(H.T, order='C')
        X = check_array(data, accept_sparse='csr')
        for run in range(iter):
            W=self.W
            violation += basis_update(X.T, W=Ht, Ht=W, l1_reg=0, l2_reg=0, shuffle=False, random_state=None)
        self.H=Ht.T
        self.pcnmf_output(input_data, self.time_vec, baseline)
        self.time_vec=self.time_vec[:,0]
    
class source_separation:
    """
    This class provides a set of source separation methods based on non-negative matrix factorization (NMF). 
    
    NMF is a machine learning algorithm that iteratively learns to reconstruct a non-negative input matrix V by finding a set of basis functions W and encoding vectors H. The NMF algorithm is based on ``sklearn.decomposition.NMF``.
    
    NMF-based source separation consists of a model training phase and a prediction phase. 
    
    In the training phase, a source separation model can be trained using supervised NMF or unsupervised PC-NMF. If training data is clean, we suggest using supervised NMF for learning source-specific features (Lin & Tsao 2020). If training data is noisy, PC-NMF can learn two sets of basis functions by assuming the target source and noise possess different periodicities (Lin et al. 2017). 
    
    In the prediction phase, adaptive source separation is applied if target sources alter their acoustic characteristics (Kwon et al. 2015), and semi-supervised SS is used when unseen sources are encountered (Smaragdis et al. 2007).

    Parameters
    ----------
    feature_length : int ≥ 1, default = 1
        Number of time bins used in the learning procedure of basis functions. 
        
        The duration of each basis function is determined by multiplying ``feature_length`` and the time resolution of the input spectrogram. We suggest choosing a minimum length that can cover the basic unit of animal vocalizations (such as a note or syllable of bird songs). Choosing a shorter duration may result in learning fragmented signals, but choosing a longer duration will slow down the computation speed.
    
    basis_num : int ≥ 1, default = 60
        Number of basis functions used in the training phase of source separation. 
        
        Using a larger number of basis functions is expected to learn more diverse features but may generate a set of time-shifting functions sharing the same spectral structure and reduce the abstraction of invariant features.
        
    filename : str 
        Path and name of the mat file containing a trained source separation model.
        
    Examples
    --------
    Learn two sets of basis functions and combine them within a model.
    
    >>> from soundscape_IR.soundscape_viewer import source_separation
    >>> # Train 1st model
    >>> model=source_separation(feature_length=5, basis_num=10)
    >>> model.learn_feature(sound1.data, sound1.f, method='NMF')
    >>> 
    >>> # Train 2nd model
    >>> model2=source_separation(feature_length=5, basis_num=10)
    >>> model2.learn_feature(sound2.data, sound2.f, method='NMF')
    >>> 
    >>> # Merge the two models
    >>> model.merge([model2])
    
    Train a source separation model using PC-NMF and save the model as a mat file.
    
    >>> from soundscape_IR.soundscape_viewer import source_separation
    >>> model=source_separation(feature_length=5, basis_num=20)
    >>> model.learn_feature(input_data=sound_train.data, f=sound_train.f, method='PCNMF')
    >>> model.save_model(filename='model.mat')
    
    Use a trained source separation model for prediction and plot the separation results
    
    >>> from soundscape_IR.soundscape_viewer import source_separation
    >>> # Load a saved model and perform source separation
    >>> model=source_separation(filename='model.mat')
    >>> model.prediction(input_data=sound_predict.data, f=sound_predict.f)
    >>> 
    >>> # View individual reconstructed spectrogram
    >>> model.plot_nmf(plot_type = 'separation', source = 1)
    >>> model.plot_nmf(plot_type = 'separation', source = 2)
    
    Apply adaptive and semi-supervised source separation in prediction
    
    >>> from soundscape_IR.soundscape_viewer import source_separation
    >>> # Enable adaptive SS by using adaptive_alpha
    >>> # Enable semi-supervised SS by using additional_basis
    >>> model=source_separation(filename='model.mat')
    >>> model.prediction(input_data=sound_predict.data, f=sound_predict.f, adaptive_alpha=0.05, additional_basis=2)
    
    Apply adaptive source separation for the target source, but not for the noise source.
    
    >>> from soundscape_IR.soundscape_viewer import source_separation
    >>> # Enable adaptive SS for 1st source, not for 2nd source
    >>> model=source_separation(filename='model.mat')
    >>> model.prediction(input_data=sound_predict.data, f=sound_predict.f, adaptive_alpha=[0.25, 0], additional_basis=0)
        
    References
    ----------
    .. [1] Kwon, K., Shin, J. W., & Kim, N. S. (2015). NMF-Based Speech Enhancement Using Bases Update. IEEE Signal Processing Letters, 22(4), 450–454. https://doi.org/10.1109/LSP.2014.2362556
    
    .. [2] Lin, T.-H., Fang, S.-H., & Tsao, Y. (2017). Improving biodiversity assessment via unsupervised separation of biological sounds from long-duration recordings. Scientific Reports, 7(1), 4547. https://doi.org/10.1038/s41598-017-04790-7
    
    .. [3] Lin, T.-H., & Tsao, Y. (2020). Source separation in ecoacoustics: A roadmap towards versatile soundscape information retrieval. Remote Sensing in Ecology and Conservation, 6(3), 236–247. https://doi.org/10.1002/rse2.141
    
    .. [4] Smaragdis, P., Raj, B. & Shashanka, M. (2007). Supervised and semi-supervised separation of sounds from single-channel mixtures. Independent Component Analysis and Signal Separation, 414–421. https://doi.org/10.1007/978-3-540-74494-8_52

    """
    def __init__(self, feature_length=1, basis_num=60, filename=None):
        self.basis_num=basis_num
        self.feature_length=feature_length
        if filename:
            self.load_model(filename)
        
    def reconstruct(self, source=None):
        if source:
            output = np.dot(self.W[:,self.W_cluster==source-1], self.H[self.W_cluster==source-1,:])
        else:
            output = np.dot(self.W, self.H)
        temp=np.zeros((len(self.f), self.H.shape[1]+1-self.feature_length))
        for x in range(self.feature_length):
            temp=temp+output[(self.feature_length-(x+1))*len(self.f):(self.feature_length-x)*len(self.f),x:self.H.shape[1]+1-self.feature_length+x]
        output=np.divide(temp,self.feature_length)
        output[np.isnan(output)]=0
        return output
    
    def nmf_output(self, data, time_vec, baseline=0):
        self.original_level = 10*np.log10((10**(data.T[:,1:]/10)).sum(axis=1))
        separation=np.zeros(self.source_num, dtype=object)
        relative_level=np.zeros(self.source_num, dtype=object)
        matrix_shape=data.shape

        # Use a ratio mask (e.g., S1 = V*((W1*H1)/(W*H))) to separate different sources
        source0 = self.reconstruct() #np.dot(self.W, self.H)
        for run in range(self.source_num):
            source = self.reconstruct(source=run+1) #np.dot(self.W[:,self.W_cluster==run],self.H[self.W_cluster==run,:])
            mask=np.divide(source,source0)
            mask[np.isnan(mask)]=0
            separation[run] = np.hstack((np.reshape(time_vec,(-1,1)), np.multiply(data,mask).T+baseline))
            relative_level[run] = 10*np.log10((10**(separation[run][:,1:]/10)).sum(axis=1))
        self.separation=separation
        self.relative_level=relative_level
        
    def plot_nmf(self, plot_type='W', source=None, time_range=None, fig_width=14, fig_height=6):
        """
        Generate a figure to show the content of basis functions or encoding vectors learned in a source separation model. 
        
        Alternatively, plot the reconstructed spectrogram of each sound source. 

        Parameters
        ----------
        plot_type : {'W', 'H', 'separation'}, default = 'W'
            Type of content for plotting. 
            
            Set to 'W' for plotting basis functions, set to 'H' for plotting encoding vectors, and set to 'separation' for plotting a reconstructed spectrogram.
            
        source : None or int ≥ 1
            Source number (start from 1), with a maximum value according to the number of sources learned. 
            
            For ``plot_type = {'W', 'H'}``, this method will plot all basis functions if ``source`` is not set. For ``plot_type='separation'``, ``source`` must be specified.
        
        time_range : None or list of 2 scalars [min, max]
            Time range to plot 'H' and 'separation'.
        
        fig_width, fig_height : float > 0
            Figure width and height.
        
        """
        # Choose source according to W_cluster
        if source:
            W_list=np.where(self.W_cluster==source-1)[0]
        else:
            W_list=np.arange(self.W.shape[1])

        # Only display part of the result
        if not plot_type=='W':
            H_list=np.arange(len(self.time_vec))
            if time_range:
                H_list=np.where((self.time_vec>=time_range[0])*(self.time_vec<time_range[1])==1)[0]
            x_lim=[self.time_vec[H_list[0]][0], self.time_vec[H_list[-1]][0]]

        # Prepare W
        if self.W.shape[0]>len(self.f):
            W=np.vstack((np.full((len(self.f),len(W_list)),np.nan), self.W[:,W_list])).T.reshape(1,-1)
            W=W.reshape((-1,len(self.f))).T
        elif self.W.shape[0]==len(self.f):
            W=np.array(self.W[:,W_list])

        # Plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if plot_type=='W':
            im = ax.imshow(W[:,1:], origin='lower',  aspect='auto', cmap=cm.jet,
                          extent=[0, len(W_list), self.f[0], self.f[-1]], interpolation='none')
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Basis')
            cbar = fig.colorbar(im, ax=ax)
            #cbar.set_label('Amplitude')
            if source:
                ax.set_title('Basis functions of source %s' % source)
            else:
                ax.set_title('Basis functions')

        elif plot_type=='H':
            im = ax.imshow(self.H[W_list,:][:,H_list+int(self.feature_length/2)], origin='lower',  aspect='auto', cmap=cm.jet,
                               extent=[x_lim[0], x_lim[1], 0, len(W_list)], interpolation='none')
            ax.set_ylabel('Basis')
            ax.set_xlabel('Time')
            cbar = fig.colorbar(im, ax=ax)
            #cbar.set_label('Amplitude')
            if source:
                ax.set_title('Temporal activation of source %s' % source)
            else:
                ax.set_title('Temporal activation')

        elif plot_type=='separation':
            im = ax.imshow(self.separation[source-1][H_list,:][:,1:].T, origin='lower',  aspect='auto', cmap=cm.jet,
                              extent=[x_lim[0], x_lim[1], self.f[0], self.f[-1]], interpolation='none')
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Time')
            cbar = fig.colorbar(im, ax=ax)
            #cbar.set_label('Amplitude')
            if source:
                ax.set_title('Separation of source %s' % source)
            else:
                ax.set_title('Separation')

    def learn_feature(self, input_data, f, alpha=0, method='NMF', iter=200, show_result=False):  
        """
        This method supports the use of NMF or PC-NMF in the feature learning procedure. 
        
        Use the NMF method when a training spectrogram is clean. 
        
        Use the PC-NMF method when a training spectrogram contains significant noise. Note that PC-NMF assumes that the target source and noise display different periodicities on the input spectrogram.

        Parameters
        ----------
        input_data : ndarray of shape (time, frequency+1)
            Spectrogram data for source separation. 
            
            The first column is time, and the subsequent columns are power spectral densities associated with ``f``. Using the same spectrogram format generated from ``audio_visualization``.
    
        f : ndarray of shape (frequency,)
            Frequency of spectrogram data.
        
        alpha : float, default=0
            Constant that multiplies the regularization terms of W. 
            
            See the introduction of ``alpha_W`` in ``sklearn.decomposition.NMF``.
        
        method : {'NMF', 'PCNMF'}, default = 'NMF'
            Type of NMF method for model training. 
            
            Use the NMF method when a training spectrogram is clean. 
            
            Use the PC-NMF method when a training spectrogram contains significant noise.
            
        iter : int ≥ 1, default = 200
            Number of iterations for learning spectral features.
            
        show_result : boolean, default = False
            Plot learned basis functions and reconstructed spectrogram if set to True.
            
        Attributes
        ----------
        f : ndarray of shape (frequency,)
            Frequency of spectrogram data.
        
        time_vec : ndarray of shape (time,)
            Array of segment time of spectrogram data.
        
        W : ndarray of shape (frequency*feature_length, basis_num)
            Basis functions (spectral features) essential for reconstructing the input spectrogram.
            
        H : ndarray of shape (basis_num, time_vec)
            Encoding vectors describing the temporal activations of each basis function in the input spectrogram.
            
        source_num : int ≥ 1
            Number of sources learned in a source separation model.
        
        W_cluster : ndarray of shape (basis_num,)
            Array of source indicator of basis functions.
        
        """
        self.f=f
        self.method=method
        self.time_vec=input_data[:,0:1]
    
        if method=='NMF':
            input_data=input_data[:,1:].T
            baseline=input_data.min()
            input_data=input_data-baseline

            # Modify the input data based the feature width
            if self.feature_length>1:
                input_data=pcnmf(feature_length=self.feature_length).matrix_conv(input_data)

            # NMF-based feature learning
            self.W, self.H, _ = NMF(input_data, n_components=self.basis_num, init='random', beta_loss=2, alpha_W=alpha, l1_ratio=1, max_iter=iter)
            self.source_num = 1
            self.W_cluster=np.zeros(self.basis_num)
            if show_result:
                # Plot the spectral features(W) and temporal activations(H) learned by using the NMF
                if self.W.shape[0]>len(f):
                    W=np.vstack((np.zeros((len(f),self.W.shape[1])), self.W)).T.reshape(1,-1)
                    W=W.reshape((-1,len(f))).T
                elif self.W.shape[0]==len(f):
                    W=self.W

                # plot the features
                self.plot_nmf(plot_type='W')

        elif method=='PCNMF':
            pcnmf_model=pcnmf(feature_length=self.feature_length, basis_num=self.basis_num, 
                              alpha=alpha, beta_loss=2, sparseness=1)
            self.W, self.H, self.W_cluster = pcnmf_model.unsupervised_separation(input_data, f, source_num=2, iter=iter)
            self.source_num = 2
            if show_result:
                pcnmf_model.plot_pcnmf(source=1)
                pcnmf_model.plot_pcnmf(source=2)

    def specify_target(self, index):
        """
        This method specifies the target source from the two sound sources learned by using PC-NMF.
        
        Parameters
        ----------
        index : int ≥ 1
            Source number (start from 1) associated with target source. 
            
            In the method of ``learn_feature``, PC-NMF only learns 2 sound sources. Please set ``index`` to 1 or 2.
        
        """
        if self.method=='PCNMF':
            print("Among the 2 sources, source #" +str(index) + " is the target source.")
            if index == 2:
                self.W_cluster=np.abs(self.W_cluster-1)

    def merge(self, model):
        """
        Merge multiple source separation models. 
        
        The principle is to use one model to merge the other models trained using NMF or PC-NMF. For models trained by using PC-NMF, please specify their target sources before the merge procedure. This method gives each target source a unique source indicator but combines all noise sources into the same source indicator.
        
        Parameters
        ----------
        model : a list of models
            Source separation models trained using NMF or PC-NMF.
            
        Examples
        --------
        Train three models and combine them into one model.
    
        >>> from soundscape_IR.soundscape_viewer import source_separation
        >>> # 1st model
        >>> model_1=source_separation(feature_length=5, basis_num=10)
        >>> model_1.learn_feature(input_data=sound_1.data, f=sound_1.f, method='NMF')
        >>> 
        >>> # 2nd model
        >>> model_2=source_separation(feature_length=5, basis_num=15)
        >>> model_2.learn_feature(input_data=sound_2.data, f=sound_2.f, method='PCNMF')
        >>> model_2.specify_target(index=2) # Assuming the 2nd source is the target source
        >>> 
        >>> # 3rd model
        >>> model_3=source_separation(feature_length=5, basis_num=20)
        >>> model_3.learn_feature(input_data=sound_3.data, f=sound_3.f, method='PCNMF')
        >>> model_3.specify_target(index=1) # Assuming the 1st source is the target source
        >>>
        >>> # Merge the three models
        >>> model_1.merge([model_2, model_3])
        
        """
        current_source=np.max(self.W_cluster)
        for i in range(0, len(model)):
            self.W = np.hstack((self.W, model[i].W))
            if model[i].method=='NMF':
                self.W_cluster=np.hstack((self.W_cluster, model[i].W_cluster+i+1+current_source))
            elif model[i].method=='PCNMF':
                temp=np.array(model[i].W_cluster)+i+current_source
                temp[model[i].W_cluster==0]=0
                self.W_cluster=np.hstack((self.W_cluster, temp))
        self.source_num = int(np.max(self.W_cluster)+1)
        self.basis_num = len(self.W_cluster)

    def prediction(self, input_data, f, iter=50, adaptive_alpha=0, additional_basis=0):
        """
        Perform prediction in source separation procedures. This method supports conventional NMF, adaptive NMF, and semi-supervised NMF. 
        
        Set ``adaptive_alpha`` and ``additional_basis`` to 0 for using conventional NMF, which assumes that the testing spectrogram contains the same target sources and noise sources as the training spectrograms. Apply adaptive source separation if target sources alter their acoustic characteristics. This can be done by setting ``adaptive_alpha``. Apply semi-supervised source separation when unseen sources are encountered. This can be done by setting ``additional_basis``.
        
        ``adaptive_alpha`` can be a value (apply for all sources) or a list of scalars (apply for different sources according to the source indicator information ``W_cluster``.

        Parameters
        ----------
        input_data : ndarray of shape (time, frequency+1)
            Spectrogram data for source separation. 
            
            The first column is time, and the subsequent columns are power spectral densities associated with ``f``. Using the same spectrogram format generated from ``audio_visualization``.
    
        f : ndarray of shape (frequency,)
            Frequency of spectrogram data.
            
        iter : int ≥ 1, default = 50
            Number of iterations for predicting source behaviors.
        
        adaptive_alpha : float [0, 1) or a list of scalars, default = 0
            Ratio to update basis functions in each iteration of adaptive source separation. 
            
            The choice of ``adaptive_alpha`` depends on the prior knowledge regarding whether the trained basis functions are representative of the target sources. If ``adaptive_alpha`` equals 0, we assume that the spectral features of target sources are invariant. If ``adaptive_alpha`` equals 1, the basis functions are set to be freely updated. 
            
            Provide a list of scalars to set ``adaptive_alpha`` for different sound sources.
        
        additional_basis : int ≥ 0, default = 0
            Adding a set of basis functions initiated by random values into a source separation model to enable semi-supervised source separation. 
            
            During the iterative updating procedure, the trained basis functions are fixed (if adaptive source separation is inactivated), but the newly added basis functions can update themselves through the standard NMF update rule. For mixtures containing many new sources, a higher number of ``additional_basis`` can give more building blocks to perform spectrogram reconstruction.
        
        Attributes
        ----------
        separation : ndarray of shape (source_num,)
            Reconstructed spectrograms of sources separated by using a source separation model.
        
        relative_level : ndarray of shape (source_num,)
            Intensities of sources separated by using a source separation model. 
            
            For each source, the intensity at each time bin is an integration of signal-to-noise ratio along the frequency domain.
            
        original_level : ndarray of shape (time,)
            Time-series intensities of the input spectrogram.
        
        """
        self.f=f
        self.time_vec=input_data[:,0:1]
        self.adaptive_alpha=adaptive_alpha
        self.additional_basis=additional_basis
        input_data=input_data[:,1:].T
        baseline=input_data.min()
        input_data=input_data-baseline

        # Modify the input data based on the feature length
        data=pcnmf(feature_length=self.feature_length).matrix_conv(input_data)

        # Check the learning rate (adaptive_alpha) for each source (adaptive NMF)
        if isinstance(adaptive_alpha, int) or isinstance(adaptive_alpha, float):
            adaptive_alpha = np.ones(self.source_num)*adaptive_alpha
        else:
            if len(adaptive_alpha) != self.source_num:
                print("Error: The model has " +str(self.source_num) +" sources. Please specify adaptive_alpha for every source")
                return

        # Add additional basis for feature learning (semi-supervised NMF)
        if additional_basis>0:
            self.W_cluster=np.append(self.W_cluster, np.ones(additional_basis)*self.source_num)
            self.source_num=self.source_num+1
            self.basis_num=self.basis_num+additional_basis
            adaptive_alpha = np.append(adaptive_alpha, 1)

        # supervised NMF
        W, H = NMF_init(data, self.basis_num, init='random')
        W[:, 0:self.basis_num-additional_basis] = self.W

        violation=0
        Ht=H.T
        Ht = check_array(H.T, order='C')
        X = check_array(data, accept_sparse='csr')

        W = np.array(W)
        for run in range(iter):
            prev_W = np.array(W)
            # update H
            violation += basis_update(X.T, W=Ht, Ht=W, l1_reg=0, l2_reg=0, shuffle=False, random_state=None)
            # update W
            violation += basis_update(X, W=W, Ht=Ht, l1_reg=0, l2_reg=0, shuffle=False, random_state=None)

            for i in range(self.source_num):
                index = np.where(self.W_cluster == i)
                if adaptive_alpha[i] == 0:
                    W[:,index] = prev_W[:,index]
                else:
                    W[:,index] = prev_W[:,index]*(1-adaptive_alpha[i]) + W[:,index]*adaptive_alpha[i]

        self.W=W
        self.H=Ht.T
        self.nmf_output(input_data, self.time_vec, baseline)
        #self.time_vec=self.time_vec[:,0]

    def save_model(self, filename='NMF_model.mat', folder_id=[]):
        """
        Save basis functions and model parameters
        
        Parameters
        ----------
        filename : str, default = 'NMF_model.mat'
            Name of the mat file.
        
        folder_id : [] or str, default = []
            The folder ID of Google Drive folder for saving model.
            
            See https://ploi.io/documentation/database/where-do-i-get-google-drive-folder-id for the detial of folder ID. 
        
        """
        #import save_parameters
        nmf_model=save_parameters()
        nmf_model.supervised_nmf(self.f, self.W, self.W_cluster, self.source_num, self.feature_length, self.basis_num)
        savemat(filename, {'save_nmf':nmf_model})

        # save the result in Gdrive as a mat file
        if folder_id:
            Gdrive=gdrive_handle(folder_id, status_print=False)
            Gdrive.upload(filename, status_print=False)
      
    def model_check(self, model):
        print('Model parameters check')
        intf=model['save_nmf']['f'].item()[0][1]-model['save_nmf']['f'].item()[0][0]
        print('Minima and maxima frequancy bin:', min(model['save_nmf']['f'].item()[0]), 'Hz and', max(model['save_nmf']['f'].item()[0]), 'Hz')
        print('Frequancy resolution:' ,intf, 'Hz')
        print('Feature length:' ,self.feature_length)
        print('Number of basis:' ,self.basis_num)
        print('Number of source:' ,self.source_num)
        if np.any(np.array(model['save_nmf'][0].dtype.names)=='sparseness'):
            print('Sparseness:', self.sparseness)
  
    def load_model(self, filename, model_check=True):
        """
        Load a source separation model
        
        Parameters
        ----------
        filename : str 
            Name of the mat file.
            
        model_check : boolean, default = True
            Print model parameters if set to True.
        
        """
        model = loadmat(filename)
        self.f=model['save_nmf']['f'].item()[0]
        self.W=model['save_nmf']['W'].item()
        self.W_cluster=model['save_nmf']['W_cluster'].item()[0]
        self.source_num=model['save_nmf']['k'].item()[0][0]
        self.feature_length=model['save_nmf']['time_frame'].item()[0][0]
        self.basis_num=model['save_nmf']['basis_num'].item()[0][0]
        if np.any(np.array(model['save_nmf'][0].dtype.names)=='sparseness'):
            self.sparseness=model['save_nmf']['sparseness'].item()[0][0]
        if model_check:
            self.model_check(model)
  
    def output_selection(self, source=1, begin_date=[], end_date=[], f_range=[]):
        source=source-1
        output_data=self.separation[source]

        # f_range: Hz
        if f_range:
            f_list=(self.f>=min(f_range))*(self.f<=max(f_range))
            f_list=np.where(f_list==True)[0]
        else:
            f_list=np.arange(len(self.f))

        f=self.f[f_list]
        f_list=np.concatenate([np.array([0]), f_list+1])

        # format of begin_data: yyyymmdd
        if begin_date:
            yy=int(begin_date[0:4])
            mm=int(begin_date[4:6])
            dd=int(begin_date[6:8])
            date=datetime.datetime(yy,mm,dd)
            begin_time=date.toordinal()+366
            list=output_data[:,0:1]>=begin_time
            if end_date:
                yy=int(end_date[0:4])
                mm=int(end_date[4:6])
                dd=int(end_date[6:8])
                date=datetime.datetime(yy,mm,dd)
                end_time=date.toordinal()+366+1
            else:
                end_time=begin_time+1
            list=list*(output_data[:,0:1]<end_time)
            list=np.where(list==True)[0]
        else:
            list=np.arange(output_data.shape[0])
        output_data=output_data[:,f_list]
        output_data=output_data[list,:]

        return output_data, f
