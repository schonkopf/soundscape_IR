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
      self.Result_PI=np.array([])
      
  def assemble(self, data, time_sort=1, f_range=[], location=0):
      if any('PI' in s for s in data['Result'].dtype.names):
        self.PI = np.array(data['Result']['PI'].item()[0])
        Result_PI = data['Result']['Result_PI'].item()
      else:
        Result_PI = np.array([])

      self.f = np.array(data['Result']['f'].item()[0])
      if f_range:
          f_list=(self.f>=min(f_range))*(self.f<=max(f_range))
          f_list=np.where(f_list==True)[0]
      else:
          f_list=np.arange(len(self.f))
      self.f=self.f[f_list]
      f_list=np.concatenate([np.array([0]), f_list+1])
      Result_median = data['Result']['LTS_median'].item()[:,f_list]
      Result_mean = data['Result']['LTS_mean'].item()[:,f_list]
      
      if self.Result_median.size == 0:
          self.Result_median = np.array(Result_median)
          self.Result_mean = np.array(Result_mean)
          self.Result_diff = self.Result_mean-self.Result_median
          self.Result_diff[:,0] = self.Result_mean[:,0]
          self.Result_PI = np.array(Result_PI)
          self.location = np.matlib.repmat(np.array([location]),Result_median.shape[0],1)
      else:
          self.Result_median = np.vstack((Result_median, self.Result_median))
          self.Result_mean = np.vstack((Result_mean, self.Result_mean))
          self.Result_diff = self.Result_mean-self.Result_median
          self.Result_diff[:,0] = self.Result_mean[:,0]
          self.Result_PI = np.vstack((Result_PI, self.Result_PI))
          self.location = np.vstack((np.matlib.repmat(np.array([location]),Result_median.shape[0],1), self.location))

      if time_sort == 1:
          temp = np.argsort(self.Result_mean[:,0])
          self.Result_median=self.Result_median[temp,:]
          self.Result_mean=self.Result_mean[temp,:]
          self.Result_diff=self.Result_diff[temp,:]
          self.location=self.location[temp]
          if self.Result_PI.shape[0]>0:
            self.Result_PI=self.Result_PI[temp,:]
  
  def LTS_check(self, data, f_range=[]):
      #set freq
      self.f = np.array(data['Result']['f'].item()[0])
      if f_range:
          f_list=(self.f>=min(f_range))*(self.f<=max(f_range))
          f_list=np.where(f_list==True)[0]
      else:
          f_list=np.arange(len(self.f))
      self.f=self.f[f_list]
      f_list=np.concatenate([np.array([0]), f_list+1])
      Result_mean = data['Result']['LTS_mean'].item()[:,f_list]
      self.Result_mean = np.array(Result_mean)
      
      #set time
      temp = self.Result_mean[:,0]
      print('LTS parameters check')
      print('Sampling rate:' ,round(data['Parameters']['sampling_freq'].item()[0][0]))
      print('Start time:', pd.to_datetime(min(temp)-693962, unit='D',origin=pd.Timestamp('1900-01-01')))
      print('End time:', pd.to_datetime(max(temp)-693962, unit='D',origin=pd.Timestamp('1900-01-01')))
      print('FFT size:' ,round(data['Parameters']['FFT_size'].item()[0][0]))
      print('Time resolution:' ,round((temp[1]-temp[0])*3600*24), 'sec')
      print('Minima and maxima frequancy bin:', min(self.f), 'Hz and ', max(self.f), 'Hz')
      print('Frequancy resolution:' ,self.f[1]-self.f[0], 'Hz')
      print('---------------------------------------------------------------')
      
  def collect_folder(self, path='.', f_range=[], time_sort=1, parameter_check=False, file_extension = '.mat', subfolder=False):
      if subfolder:
        subfolder_list = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        subfolder_path = [os.path.join(path, folder) for folder in subfolder_list]
      else:
        subfolder_list = ['Top']
        subfolder_path = [path]

      n=0
      for path in subfolder_path:
        items = os.listdir(path)
        for names in items:
          if names.endswith(file_extension):
            print('Loading file: %s' % (names))
            data = loadmat(path+'/'+names)
            if parameter_check == True:
              self.LTS_check(data, f_range)
            else:
              self.assemble(data, time_sort, f_range, location=subfolder_list[n])
        n+=1
        
  def collect_Gdrive(self, folder_id, f_range=[], time_sort=1, file_extension = '.mat', parameter_check=False, subfolder=False):
    Gdrive=gdrive_handle(folder_id)
    Gdrive.list_query(file_extension, subfolder=subfolder)
    
    n=0
    for file in Gdrive.file_list:
      print('Loading file: %s' % (file['title']))
      infilename=file['title']
      file.GetContentFile(file['title'])
      data = loadmat(file['title'])
      if parameter_check == True:
        self.LTS_check(data, f_range)
      else:
        self.assemble(data, time_sort, f_range, location=Gdrive.subfolder_list[n])
        n+=1
        os.remove(infilename)
      
  def plot_lts(self, fig_width=12, fig_height=18, gap_fill=True):
    temp,f=self.input_selection(var_name='median')
    if gap_fill:
      temp=matrix_operation().gap_fill(time_vec=temp[:,0], data=temp[:,1:], tail=[])
      temp[:,0]=temp[:,0]+693960
    temp[:,0]=temp[:,0]-366
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(fig_width, fig_height))
    im = ax1.imshow(temp.T, vmin=np.min(temp[:,1:]), vmax=np.max(temp[:,1:]),
                    origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[np.min(temp[:,0]), np.max(temp[:,0]), f[0], f[-1]], interpolation='none')
    ax1.set_title('Median-based LTS')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Date')
    ax1.xaxis_date()
    cbar1 = fig.colorbar(im, ax=ax1)
    cbar1.set_label('PSD')

    temp,f=self.input_selection(var_name='mean')
    if gap_fill:
      temp=matrix_operation().gap_fill(time_vec=temp[:,0], data=temp[:,1:], tail=[])
      temp[:,0]=temp[:,0]+693960
    temp[:,0]=temp[:,0]-366
    
    im2 = ax2.imshow(temp.T, vmin=np.min(temp[:,1:]), vmax=np.max(temp[:,1:]),
                    origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[np.min(temp[:,0]), np.max(temp[:,0]), f[0], f[-1]], interpolation='none')
    ax2.set_title('Mean-based LTS')
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Date')
    ax2.xaxis_date()
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('PSD')

    temp,f=self.input_selection(var_name='diff')
    if gap_fill:
      temp=matrix_operation().gap_fill(time_vec=temp[:,0], data=temp[:,1:], tail=[])
      temp[:,0]=temp[:,0]+693960
    temp[:,0]=temp[:,0]-366
    
    im3 = ax3.imshow(temp.T, vmin=np.min(temp[:,1:]), vmax=np.max(temp[:,1:]),
                    origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[np.min(temp[:,0]), np.max(temp[:,0]), f[0], f[-1]], interpolation='none')

    ax3.set_title('Difference-based LTS')
    ax3.set_ylabel('Frequency')
    ax3.set_xlabel('Date')
    ax3.xaxis_date()
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label('SNR')

  def input_selection(self, var_name='median', begin_date=[], end_date=[], f_range=[], prewhiten_percent=0, threshold=0, annotation=None, padding=0, annotation_target=None, gap_fill=False):
    if var_name=='median':
      input_data=self.Result_median
    elif var_name=='mean':
      input_data=self.Result_mean
    elif var_name=='diff':
      input_data=self.Result_diff
    
    if var_name=='PI':
      input_data=self.Result_PI  
      f=self.PI
    else:
      # f_range: Hz
      if f_range:
        f_list=(self.f>=min(f_range))*(self.f<=max(f_range))
        f_list=np.where(f_list==True)[0]
        f=self.f[f_list]
        f_list=np.concatenate([np.array([0]), f_list+1])
        input_data=input_data[:,f_list]
      else:
        f=self.f

    if annotation:
      df = pd.read_csv(annotation,index_col=0) 
      df = df[df['sound_type']==annotation_target]
      list=np.empty([0,1], dtype='int')
      for i in range(len(df)):
        list=np.append(list,np.where((input_data[:,0]>=df.iloc[i,2]+693960-padding)*(input_data[:,0]<=df.iloc[i,3]+693960+padding)==True)[0])

    else:
      # format of begin_date: yyyymmdd or yyyymmdd_HHMMSS
      if begin_date:
        yy=int(begin_date[0:4])
        mm=int(begin_date[4:6])
        dd=int(begin_date[6:8])
        frac=0
        if len(begin_date)>8:
          HH=int(begin_date[9:11])
          MM=int(begin_date[11:13])
          SS=int(begin_date[13:15])
          frac=((SS/60+MM)/60+HH)/24
        date=datetime.datetime(yy,mm,dd)
        begin_time=date.toordinal()+366+frac
        list=self.Result_median[:,0:1]>=begin_time
        if end_date:
          yy=int(end_date[0:4])
          mm=int(end_date[4:6])
          dd=int(end_date[6:8])
          frac=0
          if len(end_date)>8:
            HH=int(end_date[9:11])
            MM=int(end_date[11:13])
            SS=int(end_date[13:15])
            frac=((SS/60+MM)/60+HH)/24
          date=datetime.datetime(yy,mm,dd)
          end_time=date.toordinal()+366+frac
        else:
          end_time=begin_time+1
        list=list*(self.Result_median[:,0:1]<end_time)
        list=np.where(list==True)[0]
      else:
        list=np.arange(self.Result_median.shape[0])
    input_data=input_data[list,:]
    
    if len(input_data)>1:
      time_vec=input_data[:,0]
      if prewhiten_percent>0:
        input_data, ambient=matrix_operation.prewhiten(input_data, prewhiten_percent, 0)
        input_data[input_data<threshold]=threshold
        input_data[:,0]=time_vec
      if gap_fill:
        input_data=matrix_operation().gap_fill(time_vec=time_vec, data=input_data[:,1:], tail=[])
        input_data[:,0]=input_data[:,0]+693960-366
      return input_data, f
    
class data_organize:
  def __init__(self):
    self.final_result=np.array([])
    self.result_header=np.array([])
    print('A new spreadsheet has been created.')
      
  def time_fill(self, time_vec, data, header):
    save_result=matrix_operation().gap_fill(time_vec, data, tail=True)
    
    if len(self.final_result)==0:
      self.final_result=save_result
      self.result_header=['Time', header]
    else:
      self.final_result=np.hstack((self.final_result, save_result[:,1:2]))
      self.result_header=np.hstack((self.result_header, header))
    print('Columns in the spreadsheet: ', self.result_header)
      
  def remove_column(self, col):
    self.final_result=np.delete(self.final_result, col, 1)
    self.result_header=np.delete(self.result_header, col)
    print('Columns in the spreadsheet: ', self.result_header)
    
  def plot_diurnal(self, col=1, vmin=None, vmax=None, fig_width=16, fig_height=6, empty_hr_remove=False, empty_day_remove=False, reduce_resolution=1, display_cluster=0, plot=True, nan_value=0):
    hr_boundary=[np.min(24*(self.final_result[:,0]-np.floor(self.final_result[:,0]))), np.max(24*(self.final_result[:,0]-np.floor(self.final_result[:,0])))]
    if display_cluster==0:
      input_data=self.final_result[:,col]
      input_data[input_data==nan_value]=np.nan
    else:
      input_data=self.final_result[:,col]==display_cluster
      input_data[self.final_result[:,col]==nan_value]=np.nan
    input_data, time_vec=data_organize.reduce_time_resolution(input_data, self.final_result[:,0:1], reduce_resolution)
    
    hr=np.unique(24*(time_vec-np.floor(time_vec)))
    no_sample=len(time_vec)-np.remainder(len(time_vec), len(hr))
    day=np.unique(np.floor(time_vec[0:no_sample]))
    python_dt=day+693960-366

    plot_matrix=input_data.reshape((len(day), len(hr))).T
    if empty_hr_remove:
      list=np.where(np.sum(plot_matrix, axis=1)>0)[0]
      plot_matrix=plot_matrix[list, :]
      hr=hr[list]
    if empty_day_remove:
      list=np.where(np.sum(plot_matrix, axis=0)>0)[0]
      plot_matrix=plot_matrix[:, list]
      day=day[list]
    if plot:
      fig, ax = plt.subplots(figsize=(fig_width, fig_height))
      im = plt.imshow(plot_matrix, vmin=vmin, vmax=vmax, origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[python_dt[0], python_dt[-1], np.min(hr_boundary), np.max(hr_boundary)], interpolation='none')
      ax.xaxis_date()
      ax.set_title(self.result_header[col])
      plt.ylabel('Hour')
      plt.xlabel('Day')
      cbar1 = plt.colorbar(im)

    plot_matrix=np.hstack((day[:,None]+693960, plot_matrix.T))
    return plot_matrix, hr

  def reduce_time_resolution(input_data, time_vec, reduce_resolution):
    if reduce_resolution>1:
      input_data=np.nanmean(input_data.reshape((int(len(input_data)/reduce_resolution), -1)), axis=1)
      time_vec=np.mean(time_vec.reshape((len(input_data), -1)), axis=1)
    return input_data, time_vec
    
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
    input_data[np.isnan(input_data)]=0
    if pca_percent>0:
      pca = PCA(n_components=self.pca_percent)
      input_data=pca.fit_transform(input_data)
    
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
