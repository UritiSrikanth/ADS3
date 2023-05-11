# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:17:16 2023

@author: Srikanth
"""

from sklearn.metrics.cluster import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa


def plot_cluster(x, y, data, title='', centers=None, **kwargs):
    """ plot data from a clustering algorithm using dataframe column names

    Args:
        x, y : str
            names of variables in ``data``
        data : pandas.Dataframe
            desired plotting data
        title : str, optional
            title of plot
        centers : array-like or pd.DataFrame, optional
            if provided, plots the given centers of the determined groups
        **kwargs : keyword arguments, optional
            arguments to pass to plt.scatter

    Returns:
        ax : matplotlib Axes
            the Axes object with the plot drawn onto it.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = data[kwargs.get('c')]
    nlabels = labels.nunique()
    bounds = np.arange(labels.min(), nlabels + 1)

    # 20 distinct colors, more visible and differentible than tab20
    # https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    cset = ['#3cb44b', '#ffe119', '#4363d8', '#e6194b',
            '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
            '#fabebe', '#008080', '#e6beff', '#800000', '#aaffc3']  # take 14

    cm = (mpl.colors.ListedColormap(cset, N=nlabels) if labels.min() == 0
          else mpl.colors.ListedColormap(['#000000'] + cset, N=nlabels + 1))

    sct = ax.scatter(x, y, data=data, cmap=cm, edgecolors='face', **kwargs)

    if centers is not None:
        if isinstance(centers, np.ndarray):
            for g in centers[:,
                             [data.columns.get_loc(x),
                              data.columns.get_loc(y)]]:
                ax.plot(*g, '*r', markersize=12, alpha=0.6)

        if isinstance(centers, pd.DataFrame):
            ax.scatter(x, y, data=centers, marker='D', c=centers.index.values, cmap=cm,
                       # scale ♦ size by Life_Ladder score
                       s=np.exp(centers['Life_Ladder']) * 75,
                       # s=(labels.value_counts().sort_index()/len(labels))*np.sqrt(nlabels)*200,
                       # #scale ♦ sizes by n
                       edgecolors='black', linewidths=1, alpha=0.7)

        ax.set_title('(color=group, ♦size=Happiness, ♦loc = group center)')

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    fig.suptitle(title, fontsize=14)
    # 'Magic' numbers for colorbar spacing
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    norm = mpl.colors.BoundaryNorm(bounds, cm.N)
    cb = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cm,
        norm=norm,
        ticks=bounds + 0.5,
        boundaries=bounds)
    cb.set_ticklabels(bounds)

    plt.savefig(title + '.png')
    return ax


def plot_boxolin(x, y, data):
    """ Plot a box plot and a violin plot.

    Args:
        x,y : str
            columns in `data` to be plotted. x is the 'groupby' attribute.
        data : pandas.DataFrame
            DataFrame containing `x` and `y` columns

    Returns:
        axes : matplotlib Axes
            the Axes object with the plot drawn onto it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    # could use sns.boxplot, but why not try something different
    whr_grps.boxplot(column=y, by=[x], ax=axes[0])
    sns.violinplot(x, y, data=whr_grps, scale='area', ax=axes[1])
    axes[0].set_title(None)
    axes[0].set_ylabel(axes[1].get_ylabel())
    axes[1].set_ylabel(None)
    plt.show()
    return axes


def cluster(model, X, **kwargs):
    """ Run a clustering model and return predictions.

    Args:
        model : {sklearn.cluster, sklearn.mixture, or hdbscan}
            Model to fit and predict
        X : pandas.DataFrame
            Data used to fit `model`
        **kwargs : `model`.fit_predict() args, optional
            Keyword arguments to be passed into `model`.fit_predict()
    Returns:
        (labels,centers) : tuple(array, pandas.DataFrame)
            A tuple containing cluster labels and a DataFrame of cluster centers formated with X columns
    """
    clust_labels = model.fit_predict(X, **kwargs)
    centers = X.assign(**{model.__class__.__name__: clust_labels}  # assign a temp column to X with model name
                       ).groupby(model.__class__.__name__, sort=True).mean()  # group on temp, gather mean of labels

    return (clust_labels, centers)


def score_clusters(X, labels):
    """ Calculate silhouette, calinski-harabasz, and davies-bouldin scores

    Args:
        X : array-like, shape (``n_samples``, ``n_features``)
            List of ``n_features``-dimensional data points. Each row corresponds
            to a single data point.

        labels : array-like, shape (``n_samples``,)
            Predicted labels for each sample.
    Returns:
        scores : dict
            Dictionary containing the three metric scores
    """
    scores = {'silhouette': silhouette_score(X, labels),
              'calinski_harabasz': calinski_harabasz_score(X, labels),
              'davies_bouldin': davies_bouldin_score(X, labels)
              }
    return scores


# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)
RS = 404  # Random state/seed
pd.set_option("display.max_columns", 30)  # Increase columns shown
whr = pd.read_excel('Chapter2OnlineData.xls')
print(whr.columns)
full_colnames = [
    'Country',
    'Year',
    'Life_Ladder',
    'Log_GDP',
    'Social_support',
    'Life_Expectancy',
    'Freedom',
    'Generosity',
    'Corruption_Perception',
    'Positive_affect',
    'Negative_affect',
    'Confidence_natGovt',
    'Democratic_Quality',
    'Delivery_Quality',
    'sdLadder',
    'cvLadder',
    'giniIncWB',
    'giniIncWBavg',
    'giniIncGallup',
    'trust_Gallup',
    'trust_WVS81_84',
    'trust_WVS89_93',
    'trust_WVS94_98',
    'trust_WVS99_2004',
    'trust_WVS2005_09',
    'trust_WVS2010_14']
core_col = full_colnames[:9]
ext_col = full_colnames[:14] + full_colnames[17:19]
whr.columns = full_colnames
whr.columns = whr.columns.str.replace(
    'Most people can be trusted',
    'trust_in_people')
whr.columns = whr.columns.str.replace(' ', '_')
whr.columns = whr.columns.str.replace('[(),]', '')  # Strip parens and commas
whr.columns
# print(whr.iloc[np.r_[0:3,-3:0]])
whr_ext = whr[ext_col].copy()
whr_ext.groupby('Country').Year.count().describe()
# print(whr_ext)
# Get latest year indices
latest_idx = whr_ext.groupby('Country').Year.idxmax()
whrl = whr_ext.iloc[latest_idx].set_index('Country')
# Check NAs in the core data set
print(whrl[whrl[core_col[1:]].isna().any(axis=1)])

imputer = IterativeImputer(estimator=BayesianRidge(
), random_state=RS, max_iter=15).fit(whr_ext.iloc[:, 1:])
whrl_imp = pd.DataFrame(
    imputer.transform(whrl),
    columns=whrl.columns,
    index=whrl.index)
# Impute on latest forward filled data
whrffl_imp = pd.DataFrame(
    imputer.transform(whrl),
    columns=whrl.columns,
    index=whrl.index)
ss = StandardScaler()
whrX = pd.DataFrame(
    ss.fit_transform(
        whrffl_imp.drop(
            columns='Year')), columns=whrffl_imp.drop(
                columns='Year').columns, index=whrffl_imp.index)
print(whrX.head())
whr_grps = whrX.copy()

distortions = []
for n in range(2, 10):
    model = KMeans(n_clusters=n, random_state=RS).fit(whrX)
    distortions.append(model.inertia_)
    labs = model.labels_
    print(f'n_clusters: {n}\n', score_clusters(whrX, labs))


km = KMeans(n_clusters=3, random_state=RS)
clabels_km, cent_km = cluster(km, whrX)
whr_grps['KMeans'] = clabels_km
cent_km
plot_cluster(
    'Log_GDP',
    'Corruption_Perception',
    whr_grps,
    centers=cent_km,
    title='K-Means Cluster',
    c='KMeans')


ac = AgglomerativeClustering(
    n_clusters=3,
    affinity='euclidean',
    linkage='ward')
clabels_ac, cent_ac = cluster(ac, whrX)
whr_grps['AgglomerativeClustering'] = clabels_ac
cent_ac

plot_cluster(
    'Log_GDP',
    'Corruption_Perception',
    whr_grps,
    centers=cent_ac,
    title='Agglomerative Cluster',
    c='AgglomerativeClustering')


db = DBSCAN(eps=0.3)
clabels_db, cent_db = cluster(db, whrX)
whr_grps['DBSCAN'] = clabels_db
cent_db
plot_cluster(
    'Log_GDP',
    'Corruption_Perception',
    whr_grps,
    centers=cent_db,
    title='DBSCAN',
    c='DBSCAN')
