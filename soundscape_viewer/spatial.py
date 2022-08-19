import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

class spatial_mapping():
  def __init__(self, data, gps, gps_utc=0):
    df=pd.DataFrame(data)
    df[0]=pd.to_datetime(df[0]-693962,unit='D',origin=pd.Timestamp('1900-01-01'),utc=True)
    df=df.rename(columns={0:'Time'})
    self.data=df

    if type(gps)==str:
      if gps[-3:].lower()=='csv':
        self.gps=pd.read_csv(gps)
    if 'time' in self.gps:
      self.gps.time=pd.to_datetime(self.gps.time)+timedelta(hours=gps_utc)    
        
  def extract_fragments(self, fragments, mean=True, fragment_method='time'):
    # fragments: a csv file contains beginning and ending time of recording sessions
    # Or providing a time interval (seconds) for separating recording sessions
    
    if type(fragments)==str:
      if fragments[-3:].lower()=='csv':
        slice_df=pd.read_csv(fragments, sep=',')
        if fragment_method=='time':
          slice_df['Begin_time']=pd.to_datetime(slice_df['Begin_time'],utc=True)
          slice_df['End_time']=pd.to_datetime(slice_df['End_time'],utc=True)
    elif type(fragments)==int:
      segment_list=np.diff(self.data['Time'])
      slice_df=pd.DataFrame()
      slice_df['Begin_time']=self.data['Time'].values[np.concatenate(([0],np.where(segment_list>timedelta(seconds=fragments))[0]+1))]
      slice_df['End_time']=self.data['Time'].values[np.concatenate((np.where(segment_list>timedelta(seconds=fragments))[0],[len(self.data)-1]))]
      slice_df['Begin_time']=pd.to_datetime(slice_df['Begin_time'],utc=True)
      slice_df['End_time']=pd.to_datetime(slice_df['End_time'],utc=True)
      
    ndf=pd.DataFrame()
    if fragment_method=='time':
      for i in range(0, len(slice_df)):
        temp=self.data['Time']-slice_df['Begin_time'][i]
        data_list=np.where(temp>=timedelta(seconds=0))[0]
        temp2=self.data['Time']-slice_df['End_time'][i]
        data_list2=np.where(temp2<=timedelta(seconds=0))[0]
        if mean:
          result=np.mean(self.data.loc[data_list[temp[data_list].argmin()]:data_list2[temp2[data_list2].argmax()]+1])
          result['Time']=self.data.loc[data_list[temp[data_list].argmin()],'Time']
        else:
          result=self.data.loc[data_list[temp[data_list].argmin()]:data_list2[temp2[data_list2].argmax()]+1]
        ndf=ndf.append(result,ignore_index=True)
    elif fragment_method=='site':
      site_list=np.unique(fragments)
      for site in site_list:
        list_site=np.where(fragments==site)[0]
        if mean:
          result=np.mean(self.data.loc[list_site])
          result['Site']=site
        else:
          result=self.data.loc[list_site]
          result['Site']=fragments[list_site]
        ndf=ndf.append(result,ignore_index=True)
    self.data=ndf
    self.fragment_method=fragment_method
            
  def gps_mapping(self, tolerance=60):
    # tolerance: seconds

    for i in range(0,len(self.data)):
      if self.fragment_method=='time':
        location=(self.gps['time']-self.data['Time'][i]).abs().argmin()
        if (self.gps.loc[location, 'time']-self.data['Time'][i])>timedelta(seconds=tolerance):
          self.data.loc[i, 'Latitude']=np.nan
          self.data.loc[i, 'Longitude']=np.nan
        else:
          self.data.loc[i, 'Latitude']=self.gps.loc[location, 'lat']
          self.data.loc[i, 'Longitude']=self.gps.loc[location, 'lon']
      elif self.fragment_method=='site':
        self.data.loc[i, 'Latitude']=self.gps.loc[np.where(self.gps['site']==self.data['Site'][i])[0][0], 'lat']
        self.data.loc[i, 'Longitude']=self.gps.loc[np.where(self.gps['site']==self.data['Site'][i])[0][0], 'lon']
    if self.fragment_method=='time':
      self.data=self.data.drop(columns=['Time'])
    elif self.fragment_method=='site':
      self.data=self.data.drop(columns=['Site'])
        
  def save_csv(self, filename='Spatial_mapping.csv'):
    self.data.to_csv(filename, sep=',')
    print('Successifully save to '+filename)
    
  def plot_map(self, plot_col=1, plot_type='contour', mapping_resolution=10, contour_levels=15, bounding_box=[], vmin=None, vmax=None, shapefile=None, filename='Map.png'):
    # mapping resolution: meters

    x=self.data['Longitude'][:]
    y=self.data['Latitude'][:]
    z=self.data[plot_col][:]
    mapping_resolution=mapping_resolution/1000/1.852/60
    xx=np.arange(np.floor(np.min(x)/mapping_resolution)*mapping_resolution,np.ceil(np.max(x)/mapping_resolution)*mapping_resolution+mapping_resolution,mapping_resolution)
    yy=np.arange(np.floor(np.min(y)/mapping_resolution)*mapping_resolution,np.ceil(np.max(y)/mapping_resolution)*mapping_resolution+mapping_resolution,mapping_resolution)
    grid_x, grid_y = np.meshgrid(xx, yy, indexing='ij')

    fig, ax1 = plt.subplots(figsize=(int(np.round(10*(np.max(x)-np.min(x))/(np.max(y)-np.min(y)))), int(np.round(10*(np.max(y)-np.min(y))/(np.max(x)-np.min(x))))))
    if plot_type=='contour' or plot_type=='both':
      self.grid = griddata(np.hstack((x[:,None],y[:,None])), z, (grid_x, grid_y), method='cubic')
      if vmin:
        self.grid[self.grid<vmin]=vmin
      if vmax:
        self.grid[self.grid>vmax]=vmax
      im = ax1.contourf(self.grid.T, extent=([np.min(x),np.max(x),np.min(y),np.max(y)]), levels=contour_levels, vmin=vmin, vmax=vmax, extend='both')
      cbar= fig.colorbar(im)
    if plot_type=='scatter' or plot_type=='both':
      if plot_type=='both':
        im = ax1.scatter(x, y, c='', marker='o', edgecolors='k', vmin=vmin, vmax=vmax)
      if plot_type=='scatter':
        im = ax1.scatter(x, y, c=z, vmin=vmin, vmax=vmax)
        cbar= fig.colorbar(im)
      ax1.xaxis.set_major_locator(MaxNLocator(5))
      ax1.yaxis.set_major_locator(MaxNLocator(5))
      ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
      ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    if shapefile:
      import geopandas
      map_load = geopandas.read_file(shapefile).to_crs({'init': 'epsg:4326'})
      map_load.plot(ax=ax1, facecolor='gray', edgecolor='0.5')
      if len(bounding_box)>0:
        _,_=plt.xlim((bounding_box[0],bounding_box[1]))
        _,_=plt.ylim((bounding_box[2],bounding_box[3]))
      else:
        _,_=plt.xlim((np.min(x),np.max(x)))
        _,_=plt.ylim((np.min(y),np.max(y)))

    self.x=xx
    self.y=yy
    plt.savefig(filename,dpi=300)
