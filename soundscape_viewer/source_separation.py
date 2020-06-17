"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from scipy.io import savemat
from scipy.fftpack import fft
from sklearn.decomposition import non_negative_factorization as NMF
from sklearn.decomposition.nmf import _initialize_nmf as NMF_init
from sklearn.decomposition.nmf import _update_coordinate_descent as basis_update
from sklearn.utils import check_array
from .utility import save_parameters
from .utility import gdrive_handle
import datetime

class pcnmf:
  """
  Periodicity-coded non-negative matrix factorization
  Tzu-Hao Lin, Shih-Hua Fang, Yu Tsao. (2017) 
  Improving biodiversity assessment via unsupervised separation of biological 
  sounds from long-duration recordings. Scientific Reports, 7: 4547.
  
  1. Initiate PC-NMF model
  2. Run blind source separation
  3. Run model-based source separation (supervised manner)
  4. Plot the separation result
  5. Save the relative strengh as csv
  
  Examples
  --------
  >>> model=pcnmf(sparseness=0, feature_length=6)
  >>> input_data, f=LTS.input_selection('median')
  >>> model.unsupervised_separation(input_data, f, source_num=2)       
  >>> model.save_model(filename='LTS_median_pcnmf.mat')
  >>> model.plot_pcnmf(0)
  >>> model.plot_pcnmf(1)
  >>> 
  >>> separation_result=pcnmf()
  >>> separation_result.load_model(filename='LTS_median_pcnmf.mat')
  >>> separation_result.supervised_separation(input_data=LTS.input_selection('median'), f=LTS.f, iter=50)
  
  >>> analysis_result=data_organize()
  >>> analysis_result.time_fill(separation_result.time_vec, separation_result.relative_level[0], header='S1_level')
  >>> analysis_result.time_fill(separation_result.time_vec, separation_result.relative_level[1], header='S2_level')
  >>> analysis_result.save_csv(filename='Separation_result.csv')
  
  >>> analysis_result.plot_diurnal(0)
  >>> analysis_result.plot_diurnal(1)
  
  Parameters
  ----------
  sparseness : float (0 or 0-1)
      Control the percentage of zero-value elements of basis functions in NMF
      0 for no sparseness constraint
      sparseness close to 1 means only several spectral bins are activated
      Default: 0
      
  feature_length : int
      Control the number of frames for feature learning. Generally, a longer frame 
      will result in a better learning performance. However, it may cost too much 
      memory and result in a very long computation time. Suggest to choose a value 
      based on the duration of target signals.
      Default: 1
      
  basis_num : int
      Number of basis functions in NMF
      Default: 60
      
  Please check scikit-learn/sklearn/decomposition/nmf.py for the explaination of 
  the other parameters (initial, alpha, beta_loss, solver)
  
  """
  def __init__(self, sparseness=0, feature_length=1, basis_num=60, 
               initial='random', alpha=1, beta_loss='frobenius', solver='cd'):
    self.sparseness=sparseness
    self.feature_length=feature_length
    self.basis_num=basis_num
    self.initial=initial
    self.alpha=alpha
    self.beta_loss=beta_loss
    self.solver=solver
    
  def unsupervised_separation(self, input_data, f, source_num=2):    
    self.source_num=source_num
    self.f=f
    self.time_vec=input_data[:,0:1]
    input_data=input_data[:,1:].T
    baseline=input_data.min()
    input_data=input_data-baseline
    print('Running periodicity-coded NMF')
    
    # Modify the input data based the feature width
    data=self.matrix_conv(input_data)

    # 1st NMF for feature learning
    print('Feature learning...')
    self.W, self.H, n_iter = NMF(data, n_components=self.basis_num, init=self.initial, 
                            update_H=True, solver=self.solver, beta_loss=self.beta_loss,
                            alpha=self.alpha, l1_ratio=self.sparseness, regularization='transformation') 

    # Transform H into periodicity
    H_std = np.subtract(self.H, np.transpose(np.matlib.repmat(np.mean(self.H, axis=1), self.H.shape[1],1)))
    P = abs(fft(H_std))
    P2=P#[:,1:100]
    P2=P2-P2.min()

    # 2nd NMF for feature clustering
    print('Periodicity learning...')
    W_p, H_p, n_iter = NMF(np.transpose(P2), n_components=source_num, init=self.initial, 
                           update_H=True, solver=self.solver, beta_loss=self.beta_loss, 
                           alpha=self.alpha, l1_ratio=0.5, regularization='components') 
    
    # Get cluster labels for individual sources
    self.W_cluster = np.argmax(H_p, axis=0)
    
    # Reconstruct individual sources
    self.pcnmf_output(input_data, self.time_vec, baseline)
    self.time_vec=self.time_vec[:,0]
    print('Done')
    
  def matrix_conv(self, input_data):
    matrix_shape=input_data.shape
    data=np.zeros((matrix_shape[0]*self.feature_length, matrix_shape[1]-1+self.feature_length))
    for x in range(self.feature_length):
      data[x*matrix_shape[0]:(x+1)*matrix_shape[0],x:matrix_shape[1]+x]=input_data
    return data

  def pcnmf_output(self, data, time_vec, baseline=0):
    self.original_level = 10*np.log10((10**(data.T[:,1:]/10)).sum(axis=1))
    separation=np.zeros(self.source_num, dtype=np.object)
    relative_level=np.zeros(self.source_num, dtype=np.object)
    matrix_shape=data.shape
    source0 = np.dot(self.W, self.H)
    for run in range(self.source_num):
      source = np.dot(self.W[:,self.W_cluster==run],self.H[self.W_cluster==run,:])
      mask=np.divide(source,source0)
      temp=np.zeros((matrix_shape))
      for x in range(self.feature_length):
        temp=temp+mask[x*matrix_shape[0]:(x+1)*matrix_shape[0],x:matrix_shape[1]+x]
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
    
    return output_data, f;
                          
  def plot_pcnmf(self, source=1):
    source=source-1
    data=self.separation[source]
    if source<self.source_num:
      W=self.W[:,self.W_cluster==source]
      if W.shape[0]>len(self.f):
        W=np.vstack((np.zeros((len(self.f),sum(self.W_cluster==source))), W)).T.reshape((1,-1))
        W=W.reshape((-1,len(self.f))).T
      fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 10))
      im = ax1.imshow(W,
                      origin='lower',  aspect='auto', cmap=cm.jet,
                      extent=[0, sum(self.W_cluster==source), 
                              np.min(self.f), np.max(self.f)], interpolation='none')

      ax1.set_title('Features')
      ax1.set_ylabel('Frequency')
      ax1.set_xlabel('Basis')
      cbar1 = fig.colorbar(im, ax=ax1)
      cbar1.set_label('Relative amplitude')

      im2 = ax2.imshow(data[:,1:].T,
                      origin='lower',  aspect='auto', cmap=cm.jet,
                      extent=[0, data.shape[0], np.min(self.f), np.max(self.f)], interpolation='none')

      ax2.set_title('Separation result')
      ax2.set_ylabel('Frequency')
      ax2.set_xlabel('Samples')
      cbar2 = fig.colorbar(im2, ax=ax2)
      cbar2.set_label('Relative amplitude')
    else:
      print('Higher than the number of separated sources, choose a smaller number.')
      
  def save_model(self, filename='PCNMF_model.mat', folder_id=[]):
    #import save_parameters
    pcnmf_model=save_parameters()
    pcnmf_model.pcnmf(self.f, self.W, self.W_cluster, self.source_num, 
                      self.feature_length, self.sparseness, self.basis_num)
    savemat(filename, {'save_pcnmf':pcnmf_model})
    print('Successifully save to '+filename)
    
    # save the result in Gdrive as a mat file
    if folder_id:
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(filename)
      
  def load_model(self, filename):
    model = loadmat(filename)
    self.W=model['save_pcnmf']['W'].item()
    self.W_cluster=model['save_pcnmf']['W_cluster'].item()[0]
    self.source_num=model['save_pcnmf']['k'].item()[0][0]
    self.sparseness=model['save_pcnmf']['sparseness'].item()[0][0]
    self.feature_length=model['save_pcnmf']['time_frame'].item()[0][0]
    self.basis_num=model['save_pcnmf']['basis_num'].item()[0][0]
  
  def supervised_separation(self, input_data, f, iter=50):
    self.f=f    
    self.time_vec=input_data[:,0:1]
    input_data=input_data[:,1:].T
    baseline=input_data.min()
    input_data=input_data-baseline
          
    # Modify the input data based the feature width
    print('Running supervised NMF')
    data=self.matrix_conv(input_data)

    # supervised NMF
    W, H = NMF_init(data, self.basis_num, init='random')

    violation=0
    Ht=H.T
    Ht = check_array(H.T, order='C')
    X = check_array(data, accept_sparse='csr')
    print('Learning temporal activitations...')
    for run in range(iter):
      W=self.W
      violation += basis_update(X.T, W=Ht, Ht=W, l1_reg=0, l2_reg=0, shuffle=False, random_state=None)
    self.H=Ht.T
    self.pcnmf_output(input_data, self.time_vec, baseline)
    self.time_vec=self.time_vec[:,0]
    print('Done')
