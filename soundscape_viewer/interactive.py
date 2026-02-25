"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def interactive_matrix(input_data, f, vmin=None, vmax=None, x_title=None, y_title=None, x_date=True, figure_title=None, figure_plot=True, html_save=False, html_name='Interactive_matrix.html', fig_width=None, fig_height=None):
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
  
    fig.update_layout(title=figure_title, yaxis_title=y_title, xaxis_title=x_title, width=fig_width, height=fig_height)
    if figure_plot:
        fig.show()
    else:
        html_save=True
    if html_save:
        fig.write_html(file=html_name)
    return fig

def feature_diversity_map(umap_2d, W_input, f_input, interval=1, figure_title=None, html_name=None):
    # Create feature matrix
    pick_umap1=np.arange(np.floor(umap_2d['Umap1'].min())-interval/2,np.ceil(umap_2d['Umap1'].max())+interval/2,interval)
    pick_umap2=np.arange(np.floor(umap_2d['Umap2'].min())-interval/2,np.ceil(umap_2d['Umap2'].max())+interval/2,interval)

    ky=0
    for y in pick_umap2:
        kx=0
        ind_y=np.where((umap_2d['Umap2']>=y)*(umap_2d['Umap2']<(y+interval))==1)[0]
        W=np.zeros((W_input.shape[0],len(pick_umap1)))+np.nan
        for x in pick_umap1:
            ind=np.where((umap_2d['Umap1'][ind_y]>=x)*(umap_2d['Umap1'][ind_y]<(x+interval))==1)[0]
            if len(ind)>0:
                W[:,kx]=np.mean(W_input[:,ind_y[ind]],axis=1)
            kx+=1

        W=np.vstack((np.full((len(f_input),W.shape[1]),np.nan), W)).T.reshape(1,-1)
        if ky==0:
            W_full=W.reshape((-1,len(f_input))).T
        else:
            W_full=np.vstack((W_full, np.full((1,W_full.shape[1]),np.nan), W.reshape((-1,len(f_input))).T))
        ky+=1

    # Interactive visualization
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=umap_2d['Umap1'], y=umap_2d['Umap2'], mode='markers',
                marker=dict(opacity=0.7, size=marker_size, color=umap_2d['Cluster'], coloraxis='coloraxis1'), hovertemplate=umap_2d['Time']), row=1, col=2)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'white'}, width=fig_width, height=fig_height)
    fig.update_xaxes(title='UMAP 1', range=[np.floor(pick_umap1[0]), np.ceil(pick_umap1[-1])], showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black', mirror=True, row=1, col=2)
    fig.update_yaxes(title='UMAP 2', range=[np.floor(pick_umap2[0]), np.ceil(pick_umap2[-1])], showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black', mirror=True, row=1, col=2)

    fig.add_trace(go.Heatmap(z=W_full, coloraxis='coloraxis2',
                x=np.arange(pick_umap1[0],pick_umap1[-1]+interval,(pick_umap1[-1]-pick_umap1[0]+interval)/(W_full.shape[1])),
                y=np.arange(pick_umap2[0],pick_umap2[-1]+interval,(pick_umap2[-1]-pick_umap2[0]+interval)/(W_full.shape[0]))), row=1, col=1)
    fig.update_xaxes(title_text='UMAP 1', range=[np.floor(pick_umap1[0]), np.ceil(pick_umap1[-1])], showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black', mirror=True, row=1, col=1)
    fig.update_yaxes(title_text='UMAP 2', range=[np.floor(pick_umap2[0]), np.ceil(pick_umap2[-1])], showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black', mirror=True, row=1, col=1)
    fig.update_layout(title=figure_title, coloraxis1=dict(colorscale='Viridis'), coloraxis2=dict(colorscale='Jet', showscale=False))
    if html_save:
        fig.write_html(file=html_name)
    return fig
