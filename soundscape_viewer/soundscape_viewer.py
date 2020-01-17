"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""

import numpy as np
import numpy.matlib
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .utility import save_parameters
from .utility import gdrive_handle
from .utility import matrix_operation
import datetime

class lts_viewer:
  """
  Soundscape_viewer
  1. Initiate Soundscape_viewer
  2. Load all the mat files at a specific folder (local or on Google drive)
  3. Display spectrogram
  
  Examples
  --------
  Load local data
  >>> path=r"D:\Data"
  >>> LTS=lts()
  >>> LTS.collect_folder(path)
  >>> LTS.plot_lts()
  
  Load Google drive data
  >>> folder_id='1-uKYuw_6OnYi4lI0BwgSFjbOar74QS7W'
  >>> LTS=lts()
  >>> LTS.collect_Gdrive(folder_id)
  
  """
  def __init__(self):
      print('Initialized')
      self.Result_median=np.array([])
      self.Result_mean=np.array([])
      self.Result_diff=np.array([])
      
  def assemble(self, data, time_sort=1):
      if self.Result_median.size == 0:
          self.f = data['Result']['f'].item()[0]
          self.Result_median = data['Result']['LTS_median'].item()
          self.Result_mean = data['Result']['LTS_mean'].item()
          self.Result_diff = self.Result_mean-self.Result_median
          self.Result_diff[:,0] = self.Result_mean[:,0]
      else:
          self.Result_median = np.vstack((data['Result']['LTS_median'].item(), self.Result_median))
          self.Result_mean = np.vstack((data['Result']['LTS_mean'].item(), self.Result_mean))
          self.Result_diff = self.Result_mean-self.Result_median
          self.Result_diff[:,0] = self.Result_mean[:,0]
      if time_sort == 1:
          temp = np.argsort(self.Result_mean[:,0])
          self.Result_median=self.Result_median[temp,:]
          self.Result_mean=self.Result_mean[temp,:]
          self.Result_diff=self.Result_diff[temp,:]
  
  def collect_folder(self, path, time_sort=1):
      items = os.listdir(path)
      for names in items:
        if names.endswith(".mat"):
          print('Loading file: %s' % (names))
          data = loadmat(path+'/'+names)
          self.assemble(data, time_sort)
        
  def collect_Gdrive(self, folder_id, time_sort=1):
    Gdrive=gdrive_handle(folder_id)
    Gdrive.list_query(file_extension='.mat')
    
    for file in Gdrive.file_list:
      print('Loading file: %s' % (file['title']))
      infilename=file['title']
      file.GetContentFile(file['title'])
      data = loadmat(file['title'])
      self.assemble(data, time_sort)
      os.remove(infilename)
      
  def plot_lts(self, prewhiten_percent=0, begin_date=[], end_date=[], f_range=[], fig_width=12, fig_height=18):
    # f_range: Hz
    if f_range:
        f_list=(self.f>=min(f_range))*(self.f<=max(f_range))
        f_list=np.where(f_list==True)[0]
    else:
        f_list=np.arange(len(self.f))
        f_range=[np.min(self.f), np.max(self.f)]
    
    # format of begin_data: yyyymmdd
    if begin_date:
      yy=int(begin_date[0:4])
      mm=int(begin_date[4:6])
      dd=int(begin_date[6:8])
      date=datetime.datetime(yy,mm,dd)
      begin_time=date.toordinal()+366
      list=self.Result_median[:,0]>=begin_time
      if end_date:
            yy=int(end_date[0:4])
            mm=int(end_date[4:6])
            dd=int(end_date[6:8])
            date=datetime.datetime(yy,mm,dd)
            end_time=date.toordinal()+366+1
      else:
            end_time=begin_time+1
      list=list*(self.Result_median[:,0]<end_time)
    else:
      list=range(self.Result_median.shape[0])
    
    temp,f=self.input_selection(var_name='median', prewhiten_percent=prewhiten_percent)
    temp=matrix_operation().gap_fill(time_vec=temp[list,0], data=temp[list,1:])
    temp[:,0]=temp[:,0]+693960-366
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, fig_height))
    im = ax1.imshow(temp[:,f_list+1].T,
                    origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[np.min(temp[:,0]), np.max(temp[:,0]), f_range[0], f_range[1]])
    ax1.set_title('Median-based LTS')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Date')
    ax1.xaxis_date()
    cbar1 = fig.colorbar(im, ax=ax1)
    cbar1.set_label('Amplitude')

    temp,f=self.input_selection(var_name='mean', prewhiten_percent=prewhiten_percent)
    temp=matrix_operation().gap_fill(time_vec=temp[list,0], data=temp[list,1:])
    temp[:,0]=temp[:,0]+693960-366
    
    im2 = ax2.imshow(temp[:,f_list+1].T,
                    origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[np.min(temp[:,0]), np.max(temp[:,0]), f_range[0], f_range[1]])
    ax2.set_title('Mean-based LTS')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Date')
    ax2.xaxis_date()
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('Amplitude')

    temp,f=self.input_selection(var_name='diff', prewhiten_percent=prewhiten_percent)
    temp=matrix_operation().gap_fill(time_vec=temp[list,0], data=temp[list,1:])
    temp[:,0]=temp[:,0]+693960-366
    
    im3 = ax3.imshow(temp[:,f_list+1].T,
                    origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[np.min(temp[:,0]), np.max(temp[:,0]), f_range[0], f_range[1]])

    ax3.set_title('Difference-based LTS')
    ax3.set_ylabel('Frequency')
    ax3.set_xlabel('Date')
    ax3.xaxis_date()
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label('Amplitude')
    
    return cbar1, cbar2, cbar3;

  def input_selection(self, var_name='median', begin_date=[], end_date=[], f_range=[], prewhiten_percent=0, threshold=0):
    if var_name=='median':
      input_data=self.Result_median
    elif var_name=='mean':
      input_data=self.Result_mean
    elif var_name=='diff':
      input_data=self.Result_diff     
    else:
      input_data=0
      print('Unknown input, please choose: median, mean, diff.')
    
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
      list=self.Result_median[:,0:1]>=begin_time
      if end_date:
            yy=int(end_date[0:4])
            mm=int(end_date[4:6])
            dd=int(end_date[6:8])
            date=datetime.datetime(yy,mm,dd)
            end_time=date.toordinal()+366+1
      else:
            end_time=begin_time+1
      list=list*(self.Result_median[:,0:1]<end_time)
      list=np.where(list==True)[0]
    else:
      list=np.arange(self.Result_median.shape[0])
    input_data=input_data[:,f_list]
    input_data=input_data[list,:]
    
    if len(input_data)>1:
      time_vec=input_data[:,0]
      if np.round(prewhiten_percent) > 0:
        matrix_shape=input_data.shape
        ambient = np.percentile(input_data, prewhiten_percent, axis=0)
        input_data = np.subtract(input_data, np.matlib.repmat(ambient, matrix_shape[0], 1))
        input_data[input_data<threshold]=threshold
        input_data[:,0]=time_vec
      return input_data, f;
    
class data_organize:
  def __init__(self):
    self.final_result=np.array([])
    self.result_header=np.array([])
    print('A new spreadsheet has been created.')
      
  def time_fill(self, time_vec, data, header):
    # fill the time series gap
    temp = np.argsort(time_vec)
    time_vec=time_vec[temp]
    
    if data.ndim>1:
        output=data[temp,:]
    else:
        output=data[temp]
    resolution=np.round((time_vec[1]-time_vec[0])*24*3600)
    n_time_vec=np.arange(np.floor(np.min(time_vec))*24*3600, 
                         np.ceil(np.max(time_vec))*24*3600,resolution)/24/3600

    if data.ndim>1:
        save_result=np.zeros((n_time_vec.size, data.shape[1]+1))
    else:
        save_result=np.zeros((n_time_vec.size, 2))
    
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
        
    
    if len(self.final_result)==0:
      self.final_result=save_result
      self.result_header=['Time', header]
    else:
      self.final_result=np.hstack((self.final_result, save_result[:,1:2]))
      self.result_header=np.hstack((self.result_header, header))
    print('Columns in the spreadsheet: ', self.result_header)
      
  def plot_diurnal(self, row=1, vmin=None, vmax=None, fig_width=16, fig_height=6):
    hr=np.unique(24*(self.final_result[:,0]-np.floor(self.final_result[:,0])))
    no_sample=len(self.final_result[:,0])-np.remainder(len(self.final_result[:,0]), len(hr))
    day=np.unique(np.floor(self.final_result[0:no_sample,0]))
    python_dt=day+693960-366

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = plt.imshow(self.final_result[:,row].reshape((len(day), len(hr))).T,
                    vmin=vmin, vmax=vmax, origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[python_dt[0], python_dt[-1], np.min(hr), np.max(hr)])
    ax.xaxis_date()
    ax.set_title(self.result_header[row])
    plt.ylabel('Hour')
    plt.xlabel('Day')
    cbar1 = plt.colorbar(im)
    return cbar1
    
  def save_csv(self, filename='Soundscape_analysis.csv',folder_id=[]):
    df = pd.DataFrame(self.final_result, columns = self.result_header) 
    df.to_csv(filename, sep=',')
    print('Successifully save to '+filename)
    
    if folder_id:
      #import Gdrive_upload
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(filename)
      
class clustering:
  """
  Clustering of audio events
  1. Initiate clustering model
  2. Run clustering
  3. Save the features of each cluster as csv
  4. Save the clustering labels as csv
  
  Examples
  --------
  >>> cluster_result=clustering(k=0.8, pca_percent=0.9, method='kmeans')
  >>> input_data, f=LTS.input_selection('median')
  >>> cluster_result.run(input_data, f)
  >>> cluster_result.save_cluster_feature(filename='Cluster_feature.csv')
  >>> 
  >>> analysis_result=data_organize()
  >>> analysis_result.time_fill(time_vec=cluster_result.time_vec, data=cluster_result.cluster, header='LTS_Median_Cluster')
  >>> analysis_result.save_csv(filename='Cluster_result.csv')
  >>>
  >>> analysis_result.plot_diurnal(1)
  
  Parameters
  ----------
  k : float (>0)
      Number of clusters (when k>1). When k <1, algorithm will search an optimal 
      cluster number so that the clustering result can explan k of data dispersion
      Default: 2
      
  pca_percent : float(0-1)
      Amount of variation for reducing feature dimensions.
      Default: 0.9
      
  method : 'kmeans'
      Clustering method.
      Default: 'kmeans'
  
  """
  def __init__(self, k=2, pca_percent=0.9, method='kmeans'):
    self.k=k
    self.pca_percent=pca_percent
    self.method=method

  def run(self, input_data, f, standardization=[]):
    self.time_vec=input_data[:,0]
    self.f=f
    input_data=input_data[:,1:]

    # standardization
    if standardization=='max-min':
      input_data=input_data-np.matlib.repmat(input_data.min(axis=1), input_data.shape[1],1).T
      input_data=np.divide(input_data, np.matlib.repmat(input_data.max(axis=1), input_data.shape[1], 1).T)
    
    # dimension reduction by PCA
    pca = PCA(n_components=self.pca_percent)
    data=pca.fit_transform(input_data)
    
    if self.method=='kmeans':
      cluster=self.run_kmeans(input_data)

    # extract scene features by percentiles
    soundscape_scene = np.zeros((np.max(cluster)+1,), dtype=np.object)
    scene = np.zeros((len(f),np.max(cluster)+2), dtype=np.object)
    scene[:,0]=f
    scene_label = np.zeros((np.max(cluster)+2,), dtype=np.object)
    scene_label[0] = 'Frequency (Hz)'
    for c in np.arange(np.max(cluster)+1):
      soundscape_scene[c] = np.percentile(input_data[cluster==c,:], [5, 25, 50, 75, 95], axis=0).T
      scene[0:len(f),c+1]=soundscape_scene[c][:,2]
      scene_label[c+1] = 'Scene'+ str(c+1)
 
    self.cluster=cluster+1
    self.soundscape_scene=soundscape_scene
    self.scene_feature = pd.DataFrame(scene, columns = scene_label) 
      
  def run_kmeans(self, data):
    if self.k<1:
      interia=[]
      k_try=1
      while k_try:
        # Create a kmeans model on our data, using k clusters.
        kmeans_model = KMeans(n_clusters=k_try, init='k-means++').fit(data)

        # Measuring the explained variation 
        interia.append(kmeans_model.inertia_)
        print("k:",k_try, ", explained variation:", 1-interia[k_try-1]/interia[0])
        if 1-interia[k_try-1]/interia[0] >= self.k:
          k_final=k_try
          break
        else:
          k_try=k_try+1
    else:
      k_final=round(self.k)

    print("Final trial: run ", k_final, " clusters")
    kmeans_model = KMeans(n_clusters=k_final, init='k-means++', n_init=10).fit(data)
    return kmeans_model.labels_
    
  def save_cluster_feature(self, filename='Cluster_feature.csv', folder_id=[]):
    # Save scene features
    self.scene_feature.to_csv(filename, sep=',')
    print('Successifully save to '+filename)
    
    if folder_id:
      #import Gdrive_upload
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(filename)    
      
  def plot_cluster_feature(self, cluster_no=1, freq_scale='linear', f_range=[], fig_width=12, fig_height=6):
    cluster_no=cluster_no-1
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.plot(self.f, self.soundscape_scene[cluster_no][:,2], color='blue', linewidth=4)
    plt.plot(self.f, self.soundscape_scene[cluster_no][:,1], color='black', linestyle='--', linewidth=1)
    plt.plot(self.f, self.soundscape_scene[cluster_no][:,3], color='black', linestyle='--', linewidth=1)
    plt.plot(self.f, self.soundscape_scene[cluster_no][:,0], color='black', linestyle=':', linewidth=1)
    plt.plot(self.f, self.soundscape_scene[cluster_no][:,4], color='black', linestyle=':', linewidth=1)
    plt.xscale(freq_scale)
    if not f_range:
        plt.xlim(np.min(self.f), np.max(self.f))
    else:
        plt.xlim(np.min(f_range), np.max(f_range))
    plt.title('Cluster '+str(cluster_no+1))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
