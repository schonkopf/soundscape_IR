"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""

import plotly.graph_objects as go
import pandas as pd

def interactive_matrix(input_data, f, vmin=None, vmax=None, x_title=None, y_title=None, x_date=True, figure_title=None, html_save=False, html_name='Interactive_matrix.html'):
  if x_date:
    fig = go.Figure(data=go.Heatmap(z=input_data[:,1:].T, 
        x=pd.to_datetime(input_data[:,0]-693962, unit='D',origin=pd.Timestamp('1900-01-01')), 
        y=f,
        colorscale='Jet', zmin = vmin, zmax = vmax))
  else:
    fig = go.Figure(data=go.Heatmap(z=input_data[:,1:].T, 
        x=input_data[:,0], 
        y=f,
        colorscale='Jet', zmin = vmin, zmax = vmax))

  fig.update_layout(title=figure_title, yaxis_title=y_title, xaxis_title=x_title)
  fig.show()

  if html_save:
    fig.write_html(file=html_name)
