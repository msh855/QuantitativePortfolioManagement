import pandas as pd
import numpy as np
from myScripts.functions.download_and_prepare_data import make_sma,  mom_production, get_bbq_ccy
from medh.momentum import BaseMomentum

# plotting
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('QtAgg')
import matplotlib.dates as mdates
import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.barycenters import dtw_barycenter_averaging
import math
import sys

matplotlib.style.use('seaborn')


def vol_adj_spots(df_returns, vol_window=60):
    raw_vol = BaseMomentum.std_vol(df_returns, window=vol_window)
    # Vol normalise returns
    adj_return = df_returns / raw_vol * np.sqrt(252)
    adj_spot = adj_return.cumsum()

    return adj_spot, raw_vol


def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[:, 1]

    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated


# download data
# ======================================================================================================================
ccy = 'CADUSD'
df = get_bbq_ccy(ccy)  # get_historical_spots(ccy=ccy).dropna()
df['return'] = np.log(df[ccy]).diff()
df['return_cum'] = df['return'].cumsum()
df.dropna(inplace=True)

# isolate currency
# ======================================================================================================================
spot = df[[ccy]]
spot.columns = ['Spot']

# smooth returns
# ======================================================================================================================
ret_cum_adj, raw_vol = vol_adj_spots(df[['return']])
df_spot_sm = make_sma(df=ret_cum_adj.dropna(), spans=[16, 32, 64, 128, 256])
ret_cum_adj.columns = ['return_cum_risk_adj']
raw_vol.columns = ['ret_vol']

## Cluster Time Series
# ======================================================================================================================
df_cleaned_sm = df_spot_sm.copy()
data = df_cleaned_sm.dropna().transpose().to_numpy()
data = TimeSeriesScalerMeanVariance().fit_transform(data)  # standarise
nclusters = math.ceil(math.sqrt(len(data)))  # rule of thump for number of clusters

algo = ['dtw', 'softdtw', 'euclidean'][0]
model = TimeSeriesKMeans(n_clusters=nclusters, metric=algo, max_iter=100, dtw_inertia=True)
y_pred = model.fit_predict(data)


plt.plot(dtw_barycenter_averaging(data, max_iter=5))

# find optimal clusters
# ======================================================================================================================

df_scores = []
k_values_to_try = np.arange(2, 10)
algos = ['dtw', 'softdtw', 'euclidean']

for algo in algos:
    for n_clusters in k_values_to_try:
        # Perform clustering.
        kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
                                  metric=algo,
                                  max_iter=1000, random_state=0)
        labels_clusters = kmeans.fit_predict(data)

        # Calculate various scores, and save them for further reference.
        silhouette = tslearn.clustering.silhouette_score(data, labels_clusters, random_state=0)  # default metric dwt
        tmp_scores = {"n_clusters": n_clusters,
                      "silhouette_score": silhouette,
                      "Clustering_algo": algo
                      }
        df_scores.append(tmp_scores)

df_sc = df_scores
pd.DataFrame(df_sc).set_index("n_clusters").groupby('Clustering_algo').plot()

# optimal clustering
optimal_n_cluster = df_scores[df_scores['silhouette_score'] == max(df_scores['silhouette_score'])].index[0]
model = TimeSeriesKMeans(n_clusters=optimal_n_cluster, metric=algo, max_iter=100, random_state=0)
y_pred = model.fit_predict(data)

plt.figure(figsize=(10, 10))
for yi in range(optimal_n_cluster):
    plt.subplot(optimal_n_cluster, optimal_n_cluster, yi + 1)
    for xx in data[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

# present reslults
labels = model.labels_
fancy_names_for_labels = [f"Cluster {label + 1}" for label in labels]
df_clusters = pd.DataFrame(zip(df_cleaned_sm.transpose().index, fancy_names_for_labels),
                           columns=["fund", "Cluster"]).sort_values(by="Cluster").set_index("fund")
df_clusters
