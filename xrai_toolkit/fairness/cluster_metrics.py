import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
import joblib
import warnings
from yellowbrick.cluster import SilhouetteVisualizer
warnings.filterwarnings('ignore')

def silhouette_score_visualiser(num_clusters, X_train, Y_train):
    '''
    The Silhouette Coefficient is used when the ground-truth about the dataset is unknown and computes the density of clusters computed by the model. The score is computed by averaging the silhouette coefficient for each sample, computed as the difference between the average intra-cluster distance and the mean nearest-cluster distance for each sample, normalized by the maximum value. This produces a score between 1 and -1, where 1 is highly dense clusters and -1 is completely incorrect clustering.

    The Silhouette Visualizer displays the silhouette coefficient for each sample on a per-cluster basis, visualizing which clusters are dense and which are not. This is particularly useful for determining cluster imbalance, or for selecting a value for 
    by comparing multiple visualizers. 
    Parameters
    ----------
    num_clusters : int
        Number of clusters to be used in KMeans.
    X_train : array-like
        Training data.
    Y_train : array-like
        Training labels.
    
    Returns
    -------
    silhouette_score visualisation
    '''
    list_clusters = [num_clusters - 2, num_clusters - 1, num_clusters, num_clusters + 1, num_clusters + 2]
    fig, ax = plt.subplots(3, 2, figsize=(15,8))
    for i in list_clusters:
        '''
        Create KMeans instance for different number of clusters
        '''
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(X_train)
    
    return fig

def rand_index(num_clusters, X_train, Y_train, show=False):
    """
    The Rand Index computes a similarity measure between two clusters by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.

    Parameters
    ----------
    num_clusters: number of clusters used to fit the model
    X_train: training data
    Y_train: training labels
    show: whether to show the plot or not

    Returns
    -------
    rand_dict: dictionary of rand index values for each number of clusters
    fig: plotly figure of rand index vs number of clusters

    """

    rand_dict = {}
    rand_dict['list_clusters'] = [num_clusters - 2, num_clusters - 1, num_clusters, num_clusters + 1, num_clusters + 2]

    for i in rand_dict['list_clusters']:
        model=KMeans(n_clusters=i)
        model.fit(X_train)
        if 'rand_index' in rand_dict.keys():
            rand_dict['rand_index'].append(metrics.rand_score(Y_train.to_numpy().flatten(), model.labels_))
        
        else: 
            rand_dict['rand_index'] = [metrics.rand_score(Y_train.to_numpy().flatten(), model.labels_)]

    #plot plotly graph
    fig = px.line(x=rand_dict['list_clusters'], y=rand_dict['rand_index'], markers=True, title='Rand Index vs Number of Clusters')
    
    if show:
        fig.show()

    else:
        return rand_dict, fig

    return rand_dict

def adjusted_rand_index(num_clusters, X_train, Y_train, show=False):
    """
    The Rand Index but adjusted for chance. A score above 0.7 is considered a good match.

    Parameters
    ----------
    num_clusters: number of clusters used to fit the model
    X_train: training data
    Y_train: training labels
    show: whether to show the plot or not

    Returns
    -------
    adj_rand_dict: dictionary of adjusted rand index values for each number of clusters
    fig: plotly figure of adjusted rand index vs number of clusters

    """

    adj_rand_dict = {}
    adj_rand_dict['list_clusters'] = [num_clusters - 2, num_clusters - 1, num_clusters, num_clusters + 1, num_clusters + 2]

    for i in adj_rand_dict['list_clusters']:
        model=KMeans(n_clusters=i)
        model.fit(X_train)
        if 'adj_rand_index' in adj_rand_dict.keys():
            adj_rand_dict['adj_rand_index'].append(metrics.adjusted_rand_score(Y_train.to_numpy().flatten(), model.labels_))
        
        else: 
            adj_rand_dict['adj_rand_index'] = [metrics.adjusted_rand_score(Y_train.to_numpy().flatten(), model.labels_)]

    #plot plotly graph
    fig = px.line(x=adj_rand_dict['list_clusters'], y=adj_rand_dict['adj_rand_index'], markers=True, title='Adjusted Rand Index vs Number of Clusters')
    
    if show:
        fig.show()

    else:
        return adj_rand_dict, fig

    return adj_rand_dict

def mutual_info(num_clusters, X_train, Y_train, show=False):
    """
    A measure of the similarity between 2 labels of the same data. The AMI returns a value of 1 when the two partitions are identical (ie perfectly matched). Random partitions (independent labellings) have an expected AMI around 0 on average hence can be negative.

    Parameters
    ----------
    num_clusters: number of clusters used to fit the model
    X_train: training data
    Y_train: training labels
    figshow: whether to show the plot or not

    Returns
    -------
    MI_dict: dictionary of mutual information for each number of clusters
    fig: plotly figure of MI vs number of clusters

    """

    mutual_info = {}
    mutual_info['list_clusters'] = [num_clusters - 2, num_clusters - 1, num_clusters, num_clusters + 1, num_clusters + 2]

    for i in mutual_info['list_clusters']:
        model=KMeans(n_clusters=i)
        model.fit(X_train)
        if 'mutual_info' in mutual_info.keys():
            mutual_info['mutual_info'].append(metrics.adjusted_mutual_info_score(Y_train.to_numpy().flatten(), model.labels_))
        
        else: 
            mutual_info['mutual_info'] = [metrics.adjusted_mutual_info_score(Y_train.to_numpy().flatten(), model.labels_)]

    #plot plotly graph
    fig = px.line(x=mutual_info['list_clusters'], y=mutual_info['mutual_info'], markers=True, title='Mutual Information Score vs Number of Clusters')
    
    if show:
        fig.show()

    else:
        return mutual_info, fig

    return mutual_info

def CH_index(num_clusters, X_train, Y_train, show=False):
    """
    Calinski-Harabasz Index. The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion. The C-H Index is a great way to evaluate the performance of a Clustering algorithm as it does not require information on the ground truth labels. The higher the Index, the better the performance.
    Parameters
    ----------
    num_clusters: number of clusters used to fit the model
    X_train: training data
    Y_train: training labels
    show: whether to show the plot or not

    Returns
    -------
    CH_dict: dictionary of Calinski Harabasz Index for each number of clusters
    fig: plotly figure of CH vs number of clusters

    """

    CH_dict = {}
    CH_dict['list_clusters'] = [num_clusters - 2, num_clusters - 1, num_clusters, num_clusters + 1, num_clusters + 2]

    for i in CH_dict['list_clusters']:
        model=KMeans(n_clusters=i)
        model.fit(X_train)
        if 'CH' in CH_dict.keys():
            CH_dict['CH'].append(metrics.calinski_harabasz_score(X_train, model.labels_))
        
        else: 
            CH_dict['CH'] = [metrics.calinski_harabasz_score(X_train, model.labels_)]

    #plot plotly graph
    fig = px.line(x=CH_dict['list_clusters'], y=CH_dict['CH'], markers=True, title='Calinski-Harabasz index vs Number of Clusters')
    
    if show:
        fig.show()

    else:
        return CH_dict, fig

    return CH_dict

def db_index(num_clusters, X_train, Y_train, show=False):
    """
    The Davies-Bouldin Index is defined as the average similarity measure of each cluster with its most similar cluster. Similarity is the ratio of within-cluster distances to between-cluster distances. In this way, clusters which are farther apart and less dispersed will lead to a better score.

    The minimum score is zero, and differently from most performance metrics, the lower values the better clustering performance.

    Similarly to the Silhouette Score, the D-B Index does not require the a-priori knowledge of the ground-truth labels, but has a simpler implementation in terms of fomulation than Silhouette Score.
    
    Parameters
    ----------
    num_clusters: number of clusters used to fit the model
    X_train: training data
    Y_train: training labels
    show: whether to show the plot or not

    Returns
    -------
    DB_dict: dictionary of Davies-Bouldin Index for each number of clusters
    fig: plotly figure of DB index vs number of clusters

    """

    DB_dict = {}
    DB_dict['list_clusters'] = [num_clusters - 2, num_clusters - 1, num_clusters, num_clusters + 1, num_clusters + 2]

    for i in DB_dict['list_clusters']:
        model=KMeans(n_clusters=i)
        model.fit(X_train)
        if 'DB' in DB_dict.keys():
            DB_dict['DB'].append(metrics.davies_bouldin_score(X_train, model.labels_))
        
        else: 
            DB_dict['DB'] = [metrics.davies_bouldin_score(X_train, model.labels_)]

    #plot plotly graph
    fig = px.line(x=DB_dict['list_clusters'], y=DB_dict['DB'], markers=True, title='Davies-Bouldin index vs Number of Clusters')
    
    if show:
        fig.show()

    else:
        return DB_dict, fig

    return DB_dict