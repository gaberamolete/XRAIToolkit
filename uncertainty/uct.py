import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.tools as tls
import plotly.graph_objects as go
import random
import scipy.stats
from scipy.stats import randint, uniform
from itertools import product
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
import uncertainty_toolbox as uct

def uct_manipulate_data(X_train, X_test, Y_train, Y_test, model):
    """
    Generates the appropriate arrays for the Uncertainty Toolbox

    Parameters
    -----------
    X_train: training dataset for X values
    X_test: test dataset for X values
    Y_train: training dataset for Y values
    Y_test: test dataset for Y values
    model: model that was trained
    
    Returns
    --------
    uct_data_dict: dictionary of the data that is needed for the Uncertainty Toolbox
    y_pred: predicted values from the model
    y_std: standard deviation of the predicted values
    y_mean: mean of the predicted values
    y_true: 1D Array of labels in the held out dataset
    y2_mean: 1D array of the predicted means for the held out dataset.
    y2_std: 1D array of the predicted standard deviations for the held out dataset.

    """
    uct_data_dict = {}
    uct_data_dict['y_pred'] = model.predict(X_train)
    uct_data_dict['y_std'] = model.predict(X_train).std()
    uct_data_dict['y_mean'] = model.predict(X_train).mean()
    uct_data_dict['y_true'] = Y_train.to_numpy()
    uct_data_dict['y2_mean'] = np.subtract(uct_data_dict['y_pred'], uct_data_dict['y_mean'])
    
    y2_std = np.subtract(uct_data_dict['y_pred'], uct_data_dict['y_std'])
    uct_data_dict['y2_std'] = np.abs(y2_std)
    uct_data_dict['X_train'] = X_train
    uct_data_dict['X_test'] = X_test
    
    return uct_data_dict

def uct_get_all_metrics(uct_data_dict):
    """
    Calculate all metrics for uncertainty evaluations

    Parameters
    -----------
    uct_data_dict: dictionary of the data that is needed for the Uncertainty Toolbox

    Returns
    --------
    metrics: dictionary of all metrics calculated by the Uncertainty Toolbox
    
    """
    metrics = {}
    y2_mean = uct_data_dict['y2_mean']
    y2_std = uct_data_dict['y2_std']
    y_true = uct_data_dict['y_true']
    
    uct_metrics = uct.metrics.get_all_metrics(y2_mean, y2_std, y_true)
    
    return uct_metrics

def uct_plot_adversarial_group_calibration(uct_metrics, show=False):
    """
    Plot the adversarial group calibration plots by varying group size from 0% to 100% of dataset size and recording the worst group calibration error for each group size

    Parameters
    -----------
    uct_metrics: dictionary of all metrics calculated by the Uncertainty Toolbox
    show: Boolean value to determine if the plot should be shown or not. Default to False.

    Returns
    --------
    fig: plotly figure of the adversarial group calibration
    
    """
    
    uct_adversarial_group_calibration = uct_metrics['adv_group_calibration']
    x = uct_adversarial_group_calibration['ma_adv_group_cal']['group_sizes']
    y = uct_adversarial_group_calibration['ma_adv_group_cal']['adv_group_cali_mean']
    y_stdev = uct_adversarial_group_calibration['ma_adv_group_cal']['adv_group_cali_stderr']
    
    #plot in plotly
    fig = go.Figure()
    fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=y_stdev,
                color='red',
                visible=True)
        ))

    
    fig.update_layout(title='Adversarial Group Calibration', xaxis_title = 'Group Size', yaxis_title = 'Calibration Error of Worst Group')

    if show:
        fig.show()
    
    else:
        return fig

def uct_plot_average_calibration(uct_data_dict, uct_metrics, show=False):
    """
    Plot the observed proportion vs prediction proportion of outputs falling into a range of intervals, and display miscalibration area.

    Parameters
    -----------
    uct_data_dict: dictionary of the data that is needed for the Uncertainty Toolbox
    uct_metrics: dictionary of all metrics calculated by the Uncertainty Toolbox
    show: Boolean value to determine if the plot should be shown or not. Default to False.

    Returns
    --------
    fig: plotly figure of the average calibration
    """
    y2_mean = uct_data_dict['y2_mean']
    y2_std = uct_data_dict['y2_std']
    y_true = uct_data_dict['y_true']
    uct_avg_cali = uct_metrics['avg_calibration']
    miscal_area = uct_avg_cali['miscal_area']
    uct_proportions_list = uct.metrics_calibration.get_proportion_lists(y2_mean, y2_std, y_true)

    #plot in plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uct_proportions_list[0], y=uct_proportions_list[1]))
    fig.add_trace(go.Scatter(x = [0,1], y = [0,1]))
    fig.update_layout(xaxis_title = "Predicted Proportions in Interval", 
                    yaxis_title = "Observed Proportions in Interval",
                    title = "Average Calibration")
    # add annotation
    fig.add_annotation(dict(font=dict(color='blue',size=15),
                                            x=0.8,
                                            y=-0.2,
                                            showarrow=False,
                                            text="Miscalibration Area:" + str(round(miscal_area,5)),
                                            textangle=0,
                                            xanchor='left',
                                            xref="paper",
                                            yref="paper"))

    fig.update_layout(showlegend=False)
    
    if show:
        fig.show()

    else:
        return fig

def uct_plot_ordered_intervals(X_train, X_test, Y_train, Y_test, uct_data_dict, uct_metrics, non_neg, show=False):
    
    """
    Plot predictions and predictive intervals versus true values, with points ordered by true value along x-axis.

    Parameters
    -----------
    X_train: training dataset for X values
    X_test: test dataset for X values
    Y_train: training dataset for Y values
    Y_test: test dataset for Y values
    uct_data_dict: dictionary of the data that is needed for the Uncertainty Toolbox
    uct_metrics: dictionary of all metrics calculated by the Uncertainty Toolbox
    non_neg: Boolean value whether target_feature should be non_negative
    show: Boolean value to determine if the plot should be shown or not. Default to False.

    Returns
    --------
    fig: plotly figure of the ordered intervals

    """
    
    y2_mean = uct_data_dict['y2_mean']
    y2_std = uct_data_dict['y2_std']
    if non_neg:
        prediction_intervals = uct.metrics_calibration.get_prediction_interval(y2_mean, y2_std, quantile=0.60)
        prediction_intervals_upper = prediction_intervals.upper
        prediction_intervals_lower = [x if x > 0 else 0 for x in prediction_intervals.lower]

    else:
        prediction_intervals = uct.metrics_calibration.get_prediction_interval(y2_mean, y2_std, quantile=0.60)
        prediction_intervals_upper = prediction_intervals.upper
        prediction_intervals_lower = prediction_intervals.lower
    #0.50 means interval between 25th and 75th percentile
    #0.5-q/2 and 0.5+q/2
    
    #plot in plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(name = 'Predicted Values', x=Y_train.index.values, y = y2_mean, error_y= dict(
        type='data',
        symmetric=False,
        array = prediction_intervals.upper,
        arrayminus = prediction_intervals.lower,
        color = 'purple',
        thickness = 0.5,
        width = 1),
        mode='markers', 
        marker = dict(
            color = 'blue')))

    fig.add_trace(go.Scatter(name = 'Observed Values', 
                            x=Y_train.index.values, 
                            y = Y_train, mode='markers', 
                            marker = dict(
                                color = 'green',
                                opacity = 0.1)))
                            
    fig.update_layout(xaxis_title = "Index(Ordered by Observed Values)", 
                    yaxis_title = "Predicted Values and Intervals", 
                    title = 'Ordered Prediction Intervals')

    if show:
        fig.show()

    else:
        return fig

def uct_plot_XY(X_train, X_test, Y_train, Y_test, uct_data_dict, uct_metrics, column, target_feature, non_neg, show=False):

    """
    Plot one-dimensional inputs with associated predicted values, predictive uncertainties, and true values.

    Parameters
    -----------
    X_train: training dataset for X values
    X_test: test dataset for X values
    Y_train: training dataset for Y values
    Y_test: test dataset for Y values
    uct_data_dict: dictionary of the data that is needed for the Uncertainty Toolbox
    uct_metrics: dictionary of all metrics calculated by the Uncertainty Toolbox
    column: x column to be plotted
    target_feature: name of target feature
    non_neg: Boolean value whether target_feature should be non_negative
    show: Boolean value to determine if the plot should be shown or not. Default to False.

    Returns
    --------
    fig: plotly figure of the XY plot
    """
    
    y2_mean = uct_data_dict['y2_mean']
    y2_std = uct_data_dict['y2_std']
    X_column = X_train.loc[:,column].to_numpy().flatten()

    #prediction intervals
    if non_neg:
        prediction_intervals = uct.metrics_calibration.get_prediction_interval(y2_mean, y2_std, quantile=0.60)
        prediction_intervals_upper = prediction_intervals.upper
        prediction_intervals_lower = [x if x > 0 else 0 for x in prediction_intervals.lower]

    else:
        prediction_intervals = uct.metrics_calibration.get_prediction_interval(y2_mean, y2_std, quantile=0.60)
        prediction_intervals_upper = prediction_intervals.upper
        prediction_intervals_lower = prediction_intervals.lower

    fig = go.Figure()
    fig.add_trace(go.Scatter(name = "Predictions", x=X_column, y = y2_mean, 
                    error_y = dict(
                    type='data',
                    symmetric=False,
                    array = prediction_intervals_upper,
                    arrayminus = prediction_intervals_lower,
                    color='purple',
                    thickness=0.5,
                    width = 1),
                    mode='markers',
                    marker=dict(
                    color='blue'), ))

    fig.add_trace(go.Scatter(name="Observations", x=X_column, y = Y_train, 
                            mode='markers', 
                            marker=dict(
                                color='green',
                                opacity=0.2)))

    fig.update_layout(xaxis_title = column, yaxis_title = target_feature, title='Confidence Band')

    if show:
        fig.show()
    
    else:
        return fig