#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 18:08:04 2022

@author: safishajjouz
"""

# basic plot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import quantstats as qs


def scatter_plot_simple(df, x:str ,y:str):
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")
    
    # expects as index the labels 
    
    plt.figure(figsize=[15,7])
    sns.regplot(data=df, x=x, y=y,
            fit_reg=False, marker="o", 
            color="skyblue", scatter_kws={'s':400})
    
    # add annotations one by one with a loop
    for line in range(0,df.shape[0]):
        plt.text(df[x][line], 
              df[y][line], df.index[line], 
              horizontalalignment='left', 
              size='medium', 
              color='black', 
              weight='semibold')
    
    plt.show()
    

def correlation_matrix(df, corr_limit = None, *kwargs):
    
    # check if df is a dataframe 
    if not isinstance(df, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")
    
    
    df_corr = df.corr()
    
    if corr_limit:
        df_corr = df_corr[df_corr<corr_limit]
    
    plt.figure(figsize=(18, 10))
    heatmap = sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True, cmap='BrBG', *kwargs)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);




def monthly_heatmap(returns, annot_size=10, figsize=(10, 5),
                    cbar=True, square=False,
                    compounded=True, eoy=True,
                    grayscale=False, fontname='Arial',
                    ylabel=True, savefig=None, show=True):

    # colors, ls, alpha = _core._get_colors(grayscale)
    cmap = 'gray' if grayscale else 'RdYlGn'

    returns = qs.stats.monthly_returns(returns, eoy=eoy,
                                     compounded=compounded) * 100

    fig_height = len(returns) / 3

    if figsize is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0]*1.04, max([fig_height, figsize[1]]))
       
        
   
    fig, ax = plt.subplots(figsize=figsize)
    
    # plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title('Monthly Returns (%)\n', 
                 fontsize=14, y=.995,
                 fontname=fontname, 
                 fontweight='bold', 
                 color='black', 
                 pad=20)

    # _sns.set(font_scale=.9)
    
    ax = sns.heatmap(returns, ax=ax, annot=True, center=0,
                      annot_kws={"size": annot_size},
                      fmt="0.2f", linewidths=0.5,
                      square=square, cbar=cbar, cmap=cmap,
                      cbar_kws={'format': '%.0f%%'})
    # _sns.set(font_scale=1)

    # align plot to match other
    if ylabel:
        ax.set_ylabel('Years', fontname=fontname,
                      fontweight='bold', fontsize=12)
        ax.yaxis.set_label_coords(-.1, .5)

    ax.tick_params(colors="#808080")
    plt.xticks(rotation=0, fontsize=annot_size*1.2)
    plt.yticks(rotation=0, fontsize=annot_size*1.2)

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return 

# fig = px.scatter(df_perf_python, x="AnnualizedStandardDeviation", y="AnnualizedReturn", 
#                 color = 'AnnualizedSharpe', size ='AnnualizedSharpe', template='plotly_dark',
#                   hover_data=['Company', 'ticker'], 
#                   color_continuous_scale=px.colors.sequential.Viridis, title = 'S&P 500 Companies Perfomance the Last 4 months')
# fig.show()
# fig.write_html("s_p_perf.html")


#rp.plot_dendrogram(returns=ret_log, codependence='pearson',
#                        linkage='ward', max_k=20, bins_info = 3,
#                        leaf_order=True, ax=None)



### Quick Check on Correlations (linear and Tails)
#df_corr = df_rolling.corr()
#df_corr = df_corr[df_corr<0.4]
#plt.figure(figsize=(18, 10))
#heatmap = sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
#heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

### Hierarchical Clustering 
#rp.plot_clusters(returns=ret, codependence="spearman",
 #                     linkage='ward', max_k=10,
  #                    leaf_order=True, dendrogram=True, ax=None)
  
# rp.plot_clusters(returns=df_rolling, codependence="pearson",
#                       linkage='ward', max_k=20,
#                       leaf_order=True, dendrogram=True, ax=None)

# rp.plot_dendrogram(returns=ret_simple, codependence='pearson',
#                         linkage='complete', max_k=20,
#                         leaf_order=True, ax=None)

# rp.plot_network(returns=ret, codependence="spearman",
#                      linkage="ward", k=None, max_k=10,
#                      alpha_tail=0.05, leaf_order=True,
#                      kind='spring', ax=None)