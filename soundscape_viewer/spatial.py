import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go

class spatial_mapping():
  def __init__(self, data, gps, fragments=None, resolution=None, tolerance=60, mean=True, gps_utc=0):
    data=pd.DataFrame(data,columns=['Time',1])
    data['Time']=pd.to_datetime(data['Time']-693962,unit='D',origin=pd.Timestamp('1900-01-01'),utc=True)
    gps=pd.read_csv(gps)
    gps.time=pd.to_datetime(gps.time)+timedelta(hours=gps_utc)
    if type(fragments)!='NoneType':
        self.data=self.extract_fragments(data, gps, fragments, resolution, tolerance, mean, fragment_method='time')

  def add_transect(self, data, gps, fragments=None, resolution=None, tolerance=60, mean=True, gps_utc=0):
    data=pd.DataFrame(data,columns=['Time',1])
    data['Time']=pd.to_datetime(data['Time']-693962,unit='D',origin=pd.Timestamp('1900-01-01'),utc=True)
    gps=pd.read_csv(gps)
    gps.time=pd.to_datetime(gps.time)+timedelta(hours=gps_utc)
    if type(fragments)!='NoneType':
        self.data=self.data.append(self.extract_fragments(data, gps, fragments, resolution, tolerance, mean, fragment_method='time'),ignore_index=True)
        
  def extract_fragments(self, data, gps, fragments, resolution=None, tolerance=60, mean=True, fragment_method='time'):
    # fragments: a csv file contains beginning and ending time of recording sessions
    # Or providing a time interval (seconds) for separating recording sessions
    if type(fragments)==int:
        segment_list=np.diff(data['Time'])
        slice_df=pd.DataFrame()
        slice_df['Begin_time']=data['Time'].values[np.concatenate(([0],np.where(segment_list>timedelta(seconds=fragments))[0]+1))]
        slice_df['End_time']=data['Time'].values[np.concatenate((np.where(segment_list>timedelta(seconds=fragments))[0],[len(data)-1]))]
        slice_df['Begin_time']=pd.to_datetime(slice_df['Begin_time'],utc=True)
        slice_df['End_time']=pd.to_datetime(slice_df['End_time'],utc=True)
    else:
        slice_df=fragments
        if fragment_method=='time':
          slice_df['Begin_time']=pd.to_datetime(slice_df['Begin_time']-2,unit='D',origin=pd.Timestamp('1900-01-01'),utc=True)
          slice_df['End_time']=pd.to_datetime(slice_df['End_time']-2,unit='D',origin=pd.Timestamp('1900-01-01'),utc=True)
        
    ndf=pd.DataFrame()
    if fragment_method=='time':
      for i in range(0, len(slice_df)):
        temp=data['Time']-slice_df['Begin_time'][i]
        data_list=np.where(temp>=timedelta(seconds=0))[0]
        temp2=data['Time']-slice_df['End_time'][i]
        data_list2=np.where(temp2<=timedelta(seconds=0))[0]
        if mean:
          fragment_data=data.loc[data_list[temp[data_list].argmin()]:data_list2[temp2[data_list2].argmax()]+1]
          if not resolution:
            result=np.mean(fragment_data)
            result['Time']=data.loc[data_list[temp[data_list].argmin()],'Time']
          else:
            fragment_data.set_index('Time', inplace=True)
            result=fragment_data.resample(f'{resolution}S', origin='start').mean()
            result.reset_index(drop=False, inplace=True)
        else:
          result=data.loc[data_list[temp[data_list].argmin()]:data_list2[temp2[data_list2].argmax()]+1]
        ndf=ndf.append(result,ignore_index=True)
    elif fragment_method=='site':
      site_list=np.unique(fragments)
      for site in site_list:
        list_site=np.where(fragments==site)[0]
        if mean:
          result=np.mean(data.loc[list_site])
          result['Site']=site
        else:
          result=data.loc[list_site]
          result['Site']=fragments[list_site]
        ndf=ndf.append(result,ignore_index=True)
    ndf=ndf.dropna().reset_index(drop=True)
    data=self.gps_mapping(ndf, gps, fragment_method, tolerance)
    return data
            
  def gps_mapping(self, data, gps, fragment_method='time', tolerance=60):
    # tolerance: seconds
    for i in range(0,len(data)):
      if fragment_method=='time':
        location=(gps['time']-data['Time'][i]).abs().argmin()
        if (gps.loc[location, 'time']-data['Time'][i])>timedelta(seconds=tolerance):
          data.loc[i, 'Latitude']=np.nan
          data.loc[i, 'Longitude']=np.nan
        else:
          data.loc[i, 'Latitude']=gps.loc[location, 'lat']
          data.loc[i, 'Longitude']=gps.loc[location, 'lon']
      elif fragment_method=='site':
        data.loc[i, 'Latitude']=gps.loc[np.where(gps['site']==data['Site'][i])[0][0], 'lat']
        data.loc[i, 'Longitude']=gps.loc[np.where(gps['site']==data['Site'][i])[0][0], 'lon']
    data['Time']=(data['Time']-pd.Timestamp("1900-01-01",tz=0)).dt.total_seconds()/24/3600+693962
    return data

  def add_site(self, data_to_add, x, y):
    df=pd.DataFrame({1:data_to_add[:,1],'Time':data_to_add[:,0],'Longitude':np.matlib.repmat(x,data_to_add.shape[0],1)[:,0],'Latitude':np.matlib.repmat(y,data_to_add.shape[0],1)[:,0]})
    self.data=self.data.append(df,ignore_index=True)

  def hour_day_convert(self):
    self.data['Hour']=24*(self.data['Time']-np.floor(self.data['Time']))
    self.data['Day']=np.floor(self.data['Time'])

  def depth_extraction(self, depth_matrix, lat, lon):
    depth=np.array([])
    for n in np.arange(self.data.shape[0]):
        i=np.argmin(np.abs(self.data['Longitude'][n]-lon))
        j=np.argmin(np.abs(self.data['Latitude'][n]-lat))
        depth=np.append(depth, depth_matrix[j,i])
    self.data['Depth']=depth
    
  def plot_map(self, input_data, plot_type='contour', mapping_resolution=10, contour_levels=15, bounding_box=[], title=None, vmin=None, vmax=None, shapefile=None, colorbar=False):
    x=input_data['Longitude']
    y=input_data['Latitude']
    z=input_data[1]
    # mapping resolution: meters
    mapping_resolution=mapping_resolution/1000/1.852/60
    xx=np.arange(np.floor(np.min(x)/mapping_resolution)*mapping_resolution,np.ceil(np.max(x)/mapping_resolution)*mapping_resolution+mapping_resolution,mapping_resolution)
    yy=np.arange(np.floor(np.min(y)/mapping_resolution)*mapping_resolution,np.ceil(np.max(y)/mapping_resolution)*mapping_resolution+mapping_resolution,mapping_resolution)
    grid_x, grid_y = np.meshgrid(xx, yy, indexing='ij')

    #fig, ax = plt.subplots(figsize=(int(np.round(10*(np.max(x)-np.min(x))/(np.max(y)-np.min(y)))), int(np.round(10*(np.max(y)-np.min(y))/(np.max(x)-np.min(x))))))
    if plot_type=='contour' or plot_type=='both':
      self.grid = griddata(np.hstack((x[:,None],y[:,None])), z, (grid_x, grid_y), method='cubic')
      if vmin:
        self.grid[self.grid<vmin]=vmin
      if vmax:
        self.grid[self.grid>vmax]=vmax
      im = ax.contourf(self.grid.T, extent=([np.min(x),np.max(x),np.min(y),np.max(y)]), levels=contour_levels, vmin=vmin, vmax=vmax, extend='both')
      if colorbar:
        cbar= fig.colorbar(im)
    if plot_type=='scatter' or plot_type=='both':
      if plot_type=='both':
        im = ax.scatter(x, y, marker='o', edgecolors='k', facecolors='none')
      if plot_type=='scatter':
        im = ax.scatter(x, y, c=z, vmin=vmin, vmax=vmax)
        if colorbar:
          cbar= fig.colorbar(im)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_title(title)
    
    if shapefile:
      import geopandas
      map_load = geopandas.read_file(shapefile).to_crs({'init': 'epsg:4326'})
      map_load.plot(ax=ax, facecolor='gray', edgecolor='0.5')
      if len(bounding_box)>0:
        _,_=plt.xlim((bounding_box[0],bounding_box[1]))
        _,_=plt.ylim((bounding_box[2],bounding_box[3]))
      else:
        _,_=plt.xlim((np.min(x),np.max(x)))
        _,_=plt.ylim((np.min(y),np.max(y)))
    return fig, ax

  def interactive_map(self, input_data, plot_type='markers+lines', vmin=None, vmax=None, title=None, fig_width=800, fig_height=600, html_name=None):
    t=pd.to_datetime(input_data['Time']-693962,unit='D',origin=pd.Timestamp('1900-01-01'),utc=False).round('S')
    x=input_data['Longitude']
    y=input_data['Latitude']
    z=input_data[1]
    
    fig = go.Figure()
    fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=12,
        mapbox_center={"lat": np.mean(y), "lon": np.mean(x)},
        title=title)

    fig.add_trace(go.Scattermapbox(lon=x, lat=y, text=t,
        mode=plot_type, marker=dict(color=z, size=9, colorscale='Viridis', cmin=vmin, cmax=vmax, showscale=True),
        hovertemplate=
        "<b>%{text}</b><br><br>" +
        "Value: %{marker.color:,.2f}<br>""<extra></extra>",))
    fig.update_layout(width=fig_width, height=fig_height)
    fig.show()
    if html_name:
        fig.write_html(file=html_name)
    return fig
