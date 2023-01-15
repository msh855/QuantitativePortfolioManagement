#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:35:08 2022

@author: safishajjouz
"""
import pandas as pd

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

# libraries for dendrogram

import scipy.stats as stats
from sklearn import cluster, metrics
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples, silhouette_score


def ts_clustering(df: pd.DataFrame, number_of_clusters: int = None,
                  algo: str = ['dtw', 'softdtw'][0],
                  plot_bar_center: bool = True):
    data = df.transpose().to_numpy()

    data = TimeSeriesScalerMeanVariance().fit_transform(data)

    if number_of_clusters is None:
        number_of_clusters = math.ceil(math.sqrt(len(data)))  # rule of thump for number of clusters

    model = TimeSeriesKMeans(n_clusters=number_of_clusters, metric=algo,
                             max_iter=100, random_state=0)

    y_pred = model.fit_predict(data)

    # clear results
    labels = model.labels_
    fancy_names_for_labels = [f"{label + 1}" for label in labels]
    df_clusters = pd.DataFrame(zip(df.transpose().index, fancy_names_for_labels),
                               columns=["asset", "Cluster"]).sort_values(by="Cluster").set_index("asset")

    if plot_bar_center:
        plt.figure(figsize=(10, 10))
        for yi in range(number_of_clusters):
            plt.subplot(number_of_clusters, number_of_clusters, yi + 1)
            for xx in data[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
                plt.plot(model.cluster_centers_[yi].ravel(), "r-")
                plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                         transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("Euclidean $k$-means")

    df_cluster_centers = []
    for yi in range(number_of_clusters):
        series_temp = pd.Series(model.cluster_centers_[yi].ravel())
        df_cluster_centers.append(series_temp)

    df_cluster_centers = pd.concat(df_cluster_centers, axis=1)
    df_cluster_centers.columns = df_cluster_centers.columns + 1
    df_cluster_centers.index = df.index
    df_cluster_centers.index = pd.to_datetime(df_cluster_centers.index, format="%d/%m/%Y")

    return df_clusters, df_cluster_centers


def detect_regimes(df: pd.DataFrame, series: str, optimal_clusters: int = 3,
                   metric="dtw", plot_regimes=True) -> pd.DataFrame:
    X = df[[series]]
    X_scaled = normalize(X, axis=0)

    # after optimal clustering 

    smooth_km = TimeSeriesKMeans(n_clusters=optimal_clusters, metric=metric,
                                 max_iter=10, random_state=33)
    smooth_km.fit(X_scaled)

    # present reslults 
    regime_name = series + '_Regime'
    X[regime_name] = list(smooth_km.labels_ + 1)

    if plot_regimes:
        X.reset_index().plot.scatter(x=X.index.name, y=series,
                                     c=regime_name,
                                     cmap="viridis", figsize=(16, 10), sharex=False);

        plt.ylabel(series, fontsize=20)
        plt.xlabel('Year', fontsize=20)
        plt.legend(fontsize=15)
        plt.title('Regimes',
                  fontsize=20)

    return X


def dendrogram(df: pd.DataFrame, title: str = 'Dendrogram'):
    '''
     This function produce a dendrogram based on the average of each columns

    :param df:
    :param title:
    :return:
    '''

    # Standardising the data
    X = pd.DataFrame(df.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pd.DataFrame(X))

    # Transformed the arrays of scaled values into a DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # find best
    # hier_comp = linkage(X_scaled, method='complete', metric='euclidean')
    # hier_average = linkage(X_scaled, method='average', metric='euclidean')
    hier_ward = linkage(X_scaled, method='ward', metric='euclidean')

    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=14)
    plt.xlabel('Distance', fontsize=10)
    plt.ylabel('Stock', fontsize=10)
    dendrogram(
        hier_ward,
        orientation='right',
        #   leaf_rotation=90.,
        leaf_font_size=20,
        labels=X_scaled.index.values,
        color_threshold=3
    )
    plt.yticks(fontsize=11)
    plt.show()
