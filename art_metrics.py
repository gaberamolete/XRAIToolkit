#import libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from art.metrics import PDTP, SHAPr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier, ScikitlearnDecisionTreeClassifier
from art.estimators.classification import SklearnClassifier, BlackBoxClassifier, XGBoostClassifier
import plotly.graph_objects as go

def pdtp_generate_samples(num_samples, x_train):
    sample_indexes = np.random.randint(0, x_train.shape[0], num_samples)
    return sample_indexes

    
def pdtp_metric(x_train, y_train, art_extra_classifiers_dict, key, threshold_value, sample_indexes, num_iter=10):
    """
    Calculates the PDTP metric for a given classifier and dataset and whether PDTP breaches the threshold value 
    
    Parameters
    ----------
    x: pd.DataFrame
    y: pd.DataFrame
    art_extra_classifiers: dictionary of ART classifiers and extra classifiers for the given models. Key is the given model
    key: string of model name
    num_samples: number of samples to compute PDTP on. If not supplied, PDTP will be computed for 50 samples in x.
    num_iter(int): the number of iterations of PDTP computation to run for each sample. If not supplied, defaults to 10. The result is the average across iterations.
    
    Returns: PDTP metric and boolean value indicating if (True) average pdtp is above threshold and (False) average pdtp is not above threshold
    """
    art_classifier = art_extra_classifiers_dict[key][0]
    extra_classifier = art_extra_classifiers_dict[key][1]
    #calculate the PDTP for the ART classifier
    pdtp_art = PDTP(art_classifier, extra_classifier, x_train, y_train, indexes=sample_indexes, num_iter=num_iter)

    if np.average(pdtp_art[0]) > threshold_value:
        return pdtp_art, True, sample_indexes

    else: 
        return pdtp_art, False, sample_indexes
    

def SHAPr_metric(x_train, y_train, x_test, y_test, art_extra_classifiers_dict, key, threshold_value):
    """
    Calculates the SHAPr metric for a given classifier and dataset
    
    Parameters
    ----------
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame
    art_extra_classifiers_dict: dictionary of ART classifiers and extra classifiers for the given models. Key is the name of given model
    key: string of model name
    
    Returns: SHAPr metric and boolean value indicating if (True) average SHAPr is above threshold and (False) average SHAPr is not above threshold
    """
    
    #calculate the SHAPr for the ART classifier\
    art_classifier = art_extra_classifiers_dict[key][0]
    shapr_art = SHAPr(art_classifier, x_train, y_train, x_test, y_test)

    if np.average(shapr_art) > threshold_value:
        return shapr_art, True
    
    else:
        return shapr_art, False
    



def visualisation(pdtp_art, shapr_art, pdtp_threshold_value, shapr_threshold_value):
    """
    Plot the PDTP and SHAPr metrics, average and threshold values

    Parameters
    ----------
    pdtp_art: array of PDTP scores
    shapr_art: array of SHAPr scores
    pdtp_threshold_value: threshold value for PDTP
    shapr_threshold_value: threshold value for SHAPr
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, pdtp_art[0].shape[0]), y=pdtp_art[0][:pdtp_art[0].shape[0]] / pdtp_art[0].shape[0], mode='markers',
                            marker=dict(color = 'blue'), name='pdtp'))

    fig.add_trace(go.Scatter(x=np.arange(1,pdtp_art[0].shape[0]), y=[np.mean(pdtp_art[0][:pdtp_art[0].shape[0]] / pdtp_art[0].shape[0])]*pdtp_art[0].shape[0], mode='lines', name='pdtp average'))

    fig.add_hline(y = pdtp_threshold_value)

    fig.update_layout(title='PDTP scores for all samples', xaxis_title='Sample #',
                    yaxis_title='Normalized leakage score')
    fig.show()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x = np.arange(1, shapr_art.shape[0]), y = shapr_art, mode = 'markers', 
                    marker = dict(color = 'red'), name = 'shapr'))
    fig2.add_trace(go.Scatter(x = np.arange(1, shapr_art.shape[0]), y = [np.mean(shapr_art)]*shapr_art.shape[0], 
                    mode = 'lines', marker = dict(color='black'), name = 'shapr average'))
    fig.add_hline(y = shapr_threshold_value)

    fig2.update_layout(title='SHAPr scores for all samples', xaxis_title='Sample #',
                    yaxis_title='Normalized leakage score')
    fig2.show()

    return fig, fig2