import numpy as np
import pandas as pd

import chart_studio.plotly as py
import plotly.tools as tls
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import randint, uniform, norm, binom
from itertools import product

from sklearn.preprocessing import MinMaxScaler

import torch
from netcal.scaling import TemperatureScaling, LogisticCalibration, BetaCalibration
from netcal.binning import HistogramBinning, BBQ, ENIR
from netcal.binning import IsotonicRegression as IR_Class
from netcal.metrics import ECE, MCE, ACE, MMCE, Miscalibration
from netcal.metrics import NLL, PinballLoss, PICP, QCE, ENCE, UCE
from netcal.presentation import ReliabilityDiagram, ReliabilityRegression, ReliabilityQCE
from netcal.regression import GPBeta, GPNormal, GPCauchy, VarianceScaling
from netcal.regression import IsotonicRegression as IR_Reg
from netcal import cumulative_moments

from typing import Union, Iterable, List, Dict

def calib_lc(y_pred, y_true, reg = False):
    """
    Logistic Calibration, also known as Platt scaling, trains an SVM and then trains the parameters of an additional sigmoid function to map the SVM outputs into probabilities.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    lc: Logistic Calibration object.
    lc_calibrated: Recalibrated values under Logistic Calibration, given via array.
    """
    
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    lc = LogisticCalibration()
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    lc.fit(y_pred, y_true)
    lc_calibrated = lc.transform(y_pred)
    if reg:
        lc_calibrated = scaler.inverse_transform(lc_calibrated.reshape(-1, 1)).flatten()
    return lc, lc_calibrated

def calib_bc(y_pred, y_true, reg = False):
    """
    Beta Calibration, a well-founded and easily implemented improvement on Platt scaling for binary classifiers. Assumes that per-class scores of classifier each follow a beta distribution.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    bc: Beta Calibration object.
    bc_calibrated: Recalibrated values under Beta Calibration, given via array.
    """
    
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    bc = BetaCalibration()
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    bc.fit(y_pred, y_true)
    bc_calibrated = bc.transform(y_pred)
    if reg:
        bc_calibrated = scaler.inverse_transform(bc_calibrated.reshape(-1, 1)).flatten()
    return bc, bc_calibrated

def calib_temp(y_pred, y_true, reg = False):
    """
    Temperature Scaling, a single-parameter variant of Platt Scaling.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    temperature: Temperature Scaling object.
    temp_calibrated: Recalibrated values under Temperature Scaling, given via array.
    """
    
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    temperature = TemperatureScaling()
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    temperature.fit(y_pred, y_true)
    temp_calibrated = temperature.transform(y_pred)
    if reg:
        temp_calibrated = scaler.inverse_transform(temp_calibrated.reshape(-1, 1)).flatten()
    return temperature, temp_calibrated

def calib_hb(y_pred, y_true, bins = 100, reg = False):
    """
    Histogram binning, where each prediction is sorted into a bin and assigned a calibrated confidence estimate.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    hb: Histogram Binning object.
    hb_calibrated: Recalibrated values under Histogram Binning, given via array.
    """
    
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    hb = HistogramBinning(bins = bins)
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    hb.fit(y_pred, y_true)
    hb_calibrated = hb.transform(y_pred)
    if reg:
        hb_calibrated = scaler.inverse_transform(hb_calibrated.reshape(-1, 1)).flatten()
    return hb, hb_calibrated

def calib_ir(y_pred, y_true, reg = False):
    """
    Isotonic Regression, similar to Histogram Binning but with dynamic bin sizes and boundaries. A piecewise constant function gets for to ground truth labels sorted by given confidence estimates.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    ir: Isotonic Regression object.
    ir_calibrated: Recalibrated values under Isotonic Regression, given via array.
    """
    
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    ir = IR_Class()
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    ir.fit(y_pred, y_true)
    ir_calibrated = ir.transform(y_pred)
    if reg:
        ir_calibrated = scaler.inverse_transform(ir_calibrated.reshape(-1, 1)).flatten()
    return ir, ir_calibrated

def calib_bbq(y_pred, y_true, score = 'AIC', reg = False):
    """
    Bayesian Binning into Quantiles (BBQ). Utilizes multiple Histogram Binning instances with different amounts of bins, and computes a weighted sum of all methods to obtain a well-calibrated confidence estimate. Not recommended for regression outputs.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    bbq: Bayesian Binning into Quantiles object.
    bbq_calibrated: Recalibrated values under Bayesian Binning into Quantiles, given via array.
    """
    
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    bbq = BBQ(score_function = score)
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    bbq.fit(y_pred, y_true)
    bbq_calibrated = bbq.transform(y_pred)
    if reg:
        bbq_calibrated = scaler.inverse_transform(bbq_calibrated.reshape(-1, 1)).flatten()
    return bbq, bbq_calibrated

def calib_enir(y_pred, y_true, score = 'AIC'):
    """
    Ensemble of Near Isotonic Regression models (ENIR). Allows a violation of monotony restrictions. Using the modified Pool-Adjacent-Violaters Algorithm (mPAVA), this method builds multiple Near Isotonic Regression Models and weights them by a certain score function. Not recommended for regression outputs.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    enir: ENIR object.
    enir_calibrated: Recalibrated values under ENIR, given via array.
    """
    
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    enir = ENIR(score_function = score)
    enir.fit(y_pred, y_true)
    enir_calibrated = enir.transform(y_pred)
    return enir, enir_calibrated

def calib_ece(y_pred, y_true, n_bins = 10, equal_intervals: bool = True, sample_threshold: int = 1, reg = False):
    """
    Expected Calibration Error (ECE), which divides the confidence space into several bins and measures the observed accuracy in each bin. The bin gaps between observed accuracy and bin confidence are summed up and weighted by the amount of samples in each bin.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    n_bins: Number of bins used for the internal binning. Defaults to 10.
    equal_intervals: bool. If True, the bins have the same width. If False, the bins are splitted to equalize
    the number of samples in each bin. Defaults to True.
    sample_threshold: int. bins with an amount of samples below this threshold are not included into the miscalibration. Defaults to 1.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    ece_score: Expected Calibration Error
    """
    
    ece = ECE(n_bins, equal_intervals, sample_threshold = sample_threshold)
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    ece_score = ece.measure(y_pred, y_true)
    # ece_freq = ece.frequency(y_pred, y_true)
    if reg:
        ece_score = scaler.inverse_transform(np.array(ece_score).reshape(-1, 1))[0][0]
    
    return ece_score #, ece_freq

def calib_mce(y_pred, y_true, n_bins = 10, equal_intervals: bool = True, sample_threshold: int = 1, reg = False):
    """
    Maximum Calibration Error (MCE), denotes the highest gap over all bins.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    n_bins: Number of bins used for the internal binning. Defaults to 10.
    equal_intervals: bool. If True, the bins have the same width. If False, the bins are splitted to equalize
    the number of samples in each bin. Defaults to True.
    sample_threshold: int. bins with an amount of samples below this threshold are not included into the miscalibration. Defaults to 1.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    mce_score: Maximum Calibration Error
    """
    
    mce = MCE(n_bins, equal_intervals, sample_threshold = sample_threshold)
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    mce_score = mce.measure(y_pred, y_true)
    # mce_freq = mce.frequency(y_pred, y_true)
    if reg:
        mce_score = scaler.inverse_transform(np.array(mce_score).reshape(-1, 1))[0][0]
    return mce_score #, mce_freq

def calib_ace(y_pred, y_true, n_bins = 10, equal_intervals: bool = True, sample_threshold: int = 1, reg = False):
    """
    Average Calibration Error (ACE), denotes the average miscalibration where each bin gets weighted equally.
    
    Parameters
    ------------
    y_pred: Array or series of predicted values.
    y_true: Array or Series with ground truth labels.
    n_bins: Number of bins used for the internal binning. Defaults to 10.
    equal_intervals: bool. If True, the bins have the same width. If False, the bins are splitted to equalize
    the number of samples in each bin. Defaults to True.
    sample_threshold: int. bins with an amount of samples below this threshold are not included into the miscalibration. Defaults to 1.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    ace_score: Average Calibration Error
    """
    
    ace = ACE(n_bins, equal_intervals, sample_threshold = sample_threshold)
    if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    if reg:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((y_pred, y_true)).reshape(-1, 1))
        y_pred = scaler.transform(y_pred.reshape(-1, 1))
        y_true = scaler.transform(y_true.reshape(-1, 1))
    ace_score = ace.measure(y_pred, y_true)
    # ace_freq = ace.frequency(y_pred, y_true)
    if reg:
        ace_score = scaler.inverse_transform(np.array(ace_score).reshape(-1, 1))[0][0]
    return ace_score # , ace_freq

def calib_nll(y_pred_means, y_pred_stds, y_true, reduction = 'mean'):
    """
    Negative Log Likelihood, measures the quality of a predicted probability distribution with respect to the ground truth.
    
    Parameters
    ------------
    y_pred_means: Array or series of the predicted values subtracted by the mean of the predicted values.
    y_pred_stds: Array or series of the predicted values subtracted by the standard deviation of the predicted values.
    y_true: Array or Series with ground truth labels.
    reduction: str, one of 'none', 'mean', 'batchmean', 'sum' or 'batchsum', default: 'mean'
    Specifies the reduction to apply to the output:
    - none : no reduction is performed. Return NLL for each sample and for each dim separately.
    - mean : calculate mean over all samples and all dimensions.
    - batchmean : calculate mean over all samples but for each dim separately.
                  If input has covariance matrices, 'batchmean' is the same as 'mean'.
    - sum : calculate sum over all samples and all dimensions.
    - batchsum : calculate sum over all samples but for each dim separately.
                 If input has covariance matrices, 'batchsum' is the same as 'sum'.
                 
    Returns
    -----------
    nll_score: Negative Log Likelihood
    """
    
    if isinstance(y_pred_means, pd.Series) or isinstance(y_pred_means, pd.DataFrame):
        y_pred_means = y_pred_means.to_numpy()
    if isinstance(y_pred_stds, pd.Series) or isinstance(y_pred_stds, pd.DataFrame):
        y_pred_stds = y_pred_stds.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    nll = NLL()
    nll_score = nll.measure((y_pred_means, y_pred_stds), y_true, reduction = reduction)
    return nll_score

def calib_pl(y_pred_means, y_pred_stds, y_true, quantiles = np.linspace(0.1, 0.9, 10), reduction = 'mean'):
    """
    Pinball loss is a quantile-based calibration, and is a synonym for Quantile Loss. Tests for quantile calibration of a probabilistic regression model. This is an asymmetric loss that measures the quality of the predicted quantiles.
    
    Parameters
    ------------
    y_pred_means: Array or series of the predicted values subtracted by the mean of the predicted values.
    y_pred_stds: Array or series of the predicted values subtracted by the standard deviation of the predicted values.
    y_true: Array or Series with ground truth labels.
    quantiles: Array of quantile scores in [0, 1] of size q to compute the x-valued quantile boundaries for.
    reduction: reduction : str, one of 'none', 'mean' or 'sum' default: 'mean'
    Specifies the reduction to apply to the output:
    - none : no reduction is performed. Return quantile loss for each sample, each
             quantile and for each dim separately.
    - mean : calculate mean over all quantiles, all samples and all dimensions.
    - sum : calculate sum over all quantiles, all samples and all dimensions
                 
    Returns
    -----------
    pl_score: Pinball Loss
    """
    
    if isinstance(y_pred_means, pd.Series) or isinstance(y_pred_means, pd.DataFrame):
        y_pred_means = y_pred_means.to_numpy()
    if isinstance(y_pred_stds, pd.Series) or isinstance(y_pred_stds, pd.DataFrame):
        y_pred_stds = y_pred_stds.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    pl = PinballLoss()
    pl_score = pl.measure((y_pred_means, y_pred_stds), y_true, q = quantiles, reduction = reduction)
    return pl_score

def calib_picp(y_pred_means, y_pred_stds, y_true, quantiles = np.linspace(0.1, 0.9, 10), reduction = 'mean'):
    """
    Prediction Interval Coverage Probability (PICP), a quantile-based calibration. The is used for Bayesian models to determine quality of the uncertainty estimates. In Bayesian mode, an uncertainty estimate is attached to each sample. The PICP measures the probability that the true (observed) accuracy falls into the  𝑝%
  prediction interval. Returns the PICP and the Mean Prediction Interval Width (MPIW).
    
    Parameters
    ------------
    y_pred_means: Array or series of the predicted values subtracted by the mean of the predicted values.
    y_pred_stds: Array or series of the predicted values subtracted by the standard deviation of the predicted values.
    y_true: Array or Series with ground truth labels.
    quantiles: Array of quantile scores in [0, 1] of size q to compute the x-valued quantile boundaries for.
    reduction: reduction : str, one of 'none', 'mean' or 'sum' default: 'mean'
    Specifies the reduction to apply to the output:
    - none : no reduction is performed. Return quantile loss for each sample, each
             quantile and for each dim separately.
    - mean : calculate mean over all quantiles, all samples and all dimensions.
    - sum : calculate sum over all quantiles, all samples and all dimensions
                 
    Returns
    -----------
    picp_score: Prediction Interval Coverage Probability
    mpiw_score: Mean Prediction Interval Width
    """
    
    if isinstance(y_pred_means, pd.Series) or isinstance(y_pred_means, pd.DataFrame):
        y_pred_means = y_pred_means.to_numpy()
    if isinstance(y_pred_stds, pd.Series) or isinstance(y_pred_stds, pd.DataFrame):
        y_pred_stds = y_pred_stds.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    picp = PICP()
    picp_score, mpiw_score = picp.measure((y_pred_means, y_pred_stds), y_true, q = quantiles, reduction = reduction)
    return picp_score, mpiw_score

def calib_qce(y_pred_means, y_pred_stds, y_true, bins = 10, quantiles = np.linspace(0.1, 0.9, 10 - 1), reduction = 'mean'):
    """
    Quantile Calibration Error (QCE), a quantile-based calibration. Returns the Marginal Quantile Calibration Error (M-QCE), which measures the gap between predicted quantiles and observed quantile coverage for multivariate distributions. This is based on the Normalized Estimation Error Squared (NEES), known from object tracking.
    
    Parameters
    ------------
    y_pred_means: Array or series of the predicted values subtracted by the mean of the predicted values.
    y_pred_stds: Array or series of the predicted values subtracted by the standard deviation of the predicted values.
    y_true: Array or Series with ground truth labels.
    bins: Number of bins used for the internal binning. Defaults to 10.
    quantiles: Array of quantile scores in [0, 1] of size q to compute the x-valued quantile boundaries for.
    reduction: reduction : str, one of 'none', 'mean' or 'sum' default: 'mean'
    Specifies the reduction to apply to the output:
    - none : no reduction is performed. Return quantile loss for each sample, each
             quantile and for each dim separately.
    - mean : calculate mean over all quantiles, all samples and all dimensions.
    - sum : calculate sum over all quantiles, all samples and all dimensions
                 
    Returns
    -----------
    qce_score: Marginal Quantile Calibration Error
    """
    
    if isinstance(y_pred_means, pd.Series) or isinstance(y_pred_means, pd.DataFrame):
        y_pred_means = y_pred_means.to_numpy()
    if isinstance(y_pred_stds, pd.Series) or isinstance(y_pred_stds, pd.DataFrame):
        y_pred_stds = y_pred_stds.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    qce = QCE(bins)
    qce_score = qce.measure((y_pred_means, y_pred_stds), y_true, q = quantiles, reduction = reduction)
    return qce_score

def calib_ence(y_pred_means, y_pred_stds, y_true, bins = 10):
    """
    Expected Normalized Calibration Error (ENCE), a variance-based calibration. Used for normal distributions, where we measure the quality of the predicted variance/stddev estimates. We require that the predicted variance matches the observed error variance, which is equivalent to the Mean Squared Error. ENCE applies a binning scheme with  𝐵
  bins over the predicted standard deviation  𝜎𝑦(𝑋)
  and measures the absolute (normalized) difference between RMSE and RMV. 
    
    Parameters
    ------------
    y_pred_means: Array or series of the predicted values subtracted by the mean of the predicted values.
    y_pred_stds: Array or series of the predicted values subtracted by the standard deviation of the predicted values.
    y_true: Array or Series with ground truth labels.
    bins: Number of bins used for the internal binning. Defaults to 10.
                 
    Returns
    -----------
    ence_score: Expected Normalized Calibration Error
    """
    
    if isinstance(y_pred_means, pd.Series) or isinstance(y_pred_means, pd.DataFrame):
        y_pred_means = y_pred_means.to_numpy()
    if isinstance(y_pred_stds, pd.Series) or isinstance(y_pred_stds, pd.DataFrame):
        y_pred_stds = y_pred_stds.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    ence = ENCE(bins = bins)
    ence_score = ence.measure((y_pred_means, y_pred_stds), y_true)
    return ence_score.item() # returns 0D array

def calib_uce(y_pred_means, y_pred_stds, y_true, bins = 10):
    """
    Uncertainty Calibration Error (UCE), a variance-based calibration. Used for normal distributions, where we measure the quality of the predicted variance/stddev estimates. We require that the predicted variance matches the observed error variance, which is equivalent to the Mean Squared Error. UCE applies a binning scheme with  𝐵
  bins over the predicted variance  𝜎2𝑦(𝑋)
  and measures the absolute difference between MSE and MV. 
    
    Parameters
    ------------
    y_pred_means: Array or series of the predicted values subtracted by the mean of the predicted values.
    y_pred_stds: Array or series of the predicted values subtracted by the standard deviation of the predicted values.
    y_true: Array or Series with ground truth labels.
    bins: Number of bins used for the internal binning. Defaults to 10.
                 
    Returns
    -----------
    uce_score: Uncertainty Calibration Error
    """
    
    if isinstance(y_pred_means, pd.Series) or isinstance(y_pred_means, pd.DataFrame):
        y_pred_means = y_pred_means.to_numpy()
    if isinstance(y_pred_stds, pd.Series) or isinstance(y_pred_stds, pd.DataFrame):
        y_pred_stds = y_pred_stds.to_numpy()
    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    uce = UCE(bins = bins)
    uce_score = uce.measure((y_pred_means, y_pred_stds), y_true)
    return uce_score.item() # returns 0D array

def calib_metrics(y_true, calibs: Dict, n_bins = 100, reg = False):
    """
    Outputs all calibration metrics for an array of predicted values.
    
    Parameters
    ------------
    y_true: Array or Series with ground truth labels.
    calibs: Dict, key-pair value of Calibration technique and calibrated values in array
    n_bins: Number of bins used for the internal binning. Defaults to 10.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    
    Returns
    ------------
    calibs_df: DataFrame of calculated calibration metrics on given calibration arrays
    """
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
        
    eces = []
    mces = []
    aces = []
    nlls = []
    pls = []
    picps = []
    mpiws = []
    qces = []
    ences = []
    uces = []

    for calib in calibs.values():
        if isinstance(calib, pd.Series):
            calib = calib.to_numpy()
        
        calib_mean = calib.mean()
        calib_std = calib.std()
        calib_means = calib - calib_mean
        calib_stds = abs(calib - calib_std)
        
        # ECE
        if reg:
            ece = calib_ece(calib, y_true, n_bins, reg = True)
        if not reg:
            ece = calib_ece(calib, y_true, n_bins)
        eces.append(ece)
        
        # MCE
        if reg:
            cmce = calib_mce(calib, y_true, n_bins, reg = True)
        if not reg:
            cmce = calib_mce(calib, y_true, n_bins)
        mce = MCE(n_bins)
        mces.append(cmce)

        # ACE
        if reg:
            cace = calib_ace(calib, y_true, n_bins, reg = True)
        if not reg:
            cace = calib_ace(calib, y_true, n_bins)
        aces.append(cace)
        
        # NLL
        nll = calib_nll(calib_means, calib_stds, y_true)
        nlls.append(nll)
        
        # Pinball Loss
        pl = calib_pl(calib_means, calib_stds, y_true)
        pls.append(pl)
        
        # PICP and MPIW
        picp, mpiw = calib_picp(calib_means, calib_stds, y_true)
        picps.append(picp)
        mpiws.append(mpiw)
        
        # QCE
        qce = calib_qce(calib_means, calib_stds, y_true, n_bins)
        qces.append(qce)
        
        # ENCE
        ence = calib_ence(calib_means, calib_stds, y_true, n_bins)
        ences.append(ence)
        
        # UCE
        uce = calib_uce(calib_means, calib_stds, y_true, n_bins)
        uces.append(uce)

    calibs_df = pd.DataFrame(data = {
        'ECE': eces,
        'MCE': mces,
        'ACE': aces,
        'NLL': nlls,
        'PL': pls,
        'PICP': picps,
        'MPIW': mpiws,
        'QCE': qces,
        'ENCE': ences,
        'UCE': uces,
    }, index = [name for name in calibs.keys()])
    return pd.DataFrame(calibs_df.round(4), index=calibs_df.index).reset_index()

def plot_reliability_diagram(y, x, calib, n_bins = 50, reg = False, title = None, error_bars = False,
                                error_bar_alpha = 0.05, scaling_eps = .0001, scaling_base = 10, **kwargs):
    """
    Plots a reliability diagram of predicted vs actual probabilities.
    
    Parameters
    ------------
    y: Array or Series with ground truth labels.
    x: Array or series of predicted values.
    calib: Calibration object.
    n_bins: Number of bins used for the internal binning. Defaults to 10.
    reg: bool, determines if y's given are from a regression or classification problem. Defaults to False.
    title: Title of figure. Defaults to None, and will auto-generate to "Reliability Diagram"
    error_bars: bool, determines if error bars will be shown in the figure. Defaults to False.
    error_bar_alpha: The alpha value to use for error bars, based on the binomial distribution. Defaults to 0.05 (95% CI).
    scaling_eps: Indicates the smallest meaningful positive probability considered. Defaults to 0.0001.
    scaling_base: Indicates the base used when scaling back and forth. Defaults to 10.
    **kwargs: additional args to be passed to the go.Scatter plotly.graphobjects call.
    
    Returns
    ------------
    fig: Plotly figure.
    area: Area between the model estimation and a hypothetical "perfect" model.
    """
    
    bins = np.linspace(0, 1, n_bins)
    
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.to_numpy()
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    
    if reg:
        scaler = MinMaxScaler()
        x = np.array(x).reshape(-1, 1)
        scaler.fit(x)
        digitized_x = np.digitize(scaler.transform(x), bins)
        digitized_x
        mean_count_array = np.array([[np.mean(y[[digitized_x == i][0].flatten()]),
                                      len(y[[digitized_x == i][0].flatten()]),
                                      np.mean(x[[digitized_x == i][0].flatten()])] 
                                      for i in np.unique(digitized_x)])
    
    if not reg:
        digitized_x = np.digitize(x, bins)
        mean_count_array = np.array([[np.mean(y[digitized_x == i]),
                                      len(y[digitized_x == i]),
                                      np.mean(x[digitized_x==i])] 
                                      for i in np.unique(digitized_x)])
    
    x_pts_to_graph = mean_count_array[:,2]
    y_pts_to_graph = mean_count_array[:,0]
    bin_counts = mean_count_array[:,1]
    
    if not reg:
        x_pts_to_graph_scaled = my_logit(x_pts_to_graph, eps=scaling_eps,
                                         base=scaling_base)
        y_pts_to_graph_scaled = my_logit(y_pts_to_graph, eps=scaling_eps,
                                         base=scaling_base)
    
    if error_bars:
        if reg:
            prob_range_mat = binom.interval(1 - error_bar_alpha, bin_counts, scaler.transform(x_pts_to_graph.reshape(-1, 1))) / bin_counts
            yerr_mat = (prob_range_mat - scaler.transform(x_pts_to_graph.reshape(-1, 1)))
            yerr_mat[0,:] = -yerr_mat[0,:]
            # print(yerr_mat)
            yerr = scaler.inverse_transform((yerr_mat[0,:] + yerr_mat[1,:]).reshape(-1 ,1)).flatten()
            yerrp = yerr.copy()
            yerrm = yerrp.copy()
            # print(yerr)
        
            # Limiting error bar to [0, int(max(max(x_pts_to_graph), max(y_pts_to_graph)))]
            for i, (yp, ym, y) in enumerate(zip(yerrp, yerrm, np.round(y_pts_to_graph, decimals = 4))):
                if (y + abs(yp)) > int(max(max(x_pts_to_graph), max(y_pts_to_graph))):
                    yerrp[i] = 1 - y
                    # print(i, yp, yerrp[i])
                if (y - abs(ym)) < 0:
                    yerrm[i] = abs(y)
                    # print(i, ym, yerrm[i])

        if not reg:
            prob_range_mat = binom.interval(1 - error_bar_alpha, bin_counts, x_pts_to_graph) / bin_counts
            yerr_mat = (my_logit(prob_range_mat, 
                                    eps=scaling_eps,
                                    base=scaling_base) -
                               my_logit(x_pts_to_graph,
                                           eps=scaling_eps,
                                           base=scaling_base))
            yerr_mat[0,:] = -yerr_mat[0,:]
            # print(yerr_mat)
            yerrp = yerr_mat[0,:] - yerr_mat[1,:]
            yerrm = yerrp.copy()
        
            # Limiting error bar to [0, 1]
            for i, (yp, ym, y) in enumerate(zip(yerrp, yerrm, np.round(my_logistic(y_pts_to_graph_scaled, base = scaling_base), decimals = 4))):
                if (y + abs(yp)) > 1:
                    yerrp[i] = 1 - y
                    # print(i, yp, yerrp[i])
                if (y - abs(ym)) < 0:
                    yerrm[i] = abs(y)
                    # print(i, ym, yerrm[i])
        
        yerrm = abs(yerrm)
        # print(yerrp, yerrm)
    
    fig = go.Figure()
    
    if reg:
        tvec = np.linspace(0, int(max(max(x_pts_to_graph), max(y_pts_to_graph))), 999)
    if not reg:
        tvec = np.linspace(0, 1, 999)
    
    fig.add_trace(go.Scatter(x = tvec, y = tvec, line = dict(width = 4, dash = 'dot'), name = 'Perfect', showlegend = True))
    
    if error_bars:
        if reg:
            fig.add_trace(go.Scatter(x = np.round(x_pts_to_graph, decimals = 4),
                                      y = np.round(y_pts_to_graph, decimals = 4),
                                      mode = 'markers', name = 'Model', showlegend = True,
                                     error_y = dict(type = 'data',
                                                    symmetric = False,
                                                    array = yerrp, 
                                                    arrayminus = yerrm,
                                                    visible = True)
                                    ))
        if not reg:
            fig.add_trace(go.Scatter(x = np.round(my_logistic(x_pts_to_graph_scaled, base = scaling_base), decimals = 4),
                                              y = np.round(my_logistic(y_pts_to_graph_scaled, base = scaling_base), decimals = 4),
                                              mode = 'markers', name = 'Model', showlegend = True,
                                             error_y = dict(type = 'data',
                                                            symmetric = False,
                                                            array = yerrp, 
                                                            arrayminus = yerrm,
                                                            visible = True)
                                             ))
    else:
        if reg:
            fig.add_trace(go.Scatter(x = np.round(x_pts_to_graph, decimals = 4),
                                  y = np.round(y_pts_to_graph, decimals = 4),
                                  mode = 'markers', name = 'Model', showlegend = True,
                                 ))
        if not reg:
            fig.add_trace(go.Scatter(x = np.round(my_logistic(x_pts_to_graph_scaled, base = scaling_base), decimals = 4),
                                          y = np.round(my_logistic(y_pts_to_graph_scaled, base = scaling_base), decimals = 4),
                                          mode = 'markers', name = 'Model', showlegend = True,
                                         ))
    if reg:
        fig.add_trace(go.Scatter(x = tvec, y = scaler.inverse_transform(calib.transform(scaler.transform(tvec.reshape(-1, 1))).reshape(-1, 1))[:,0],
                         mode = 'lines', name = 'Model Estimation', showlegend = True))
    if not reg:
        fig.add_trace(go.Scatter(x = tvec, y = calib.transform(tvec), mode = 'lines', name = 'Model Estimation', showlegend = True))
    
    fig.update_layout(autosize = False, width = 750, height = 500)
    
    # Area
    if reg:
        ya = scaler.inverse_transform(calib.transform(scaler.transform(tvec.reshape(-1, 1))).reshape(-1, 1))[:,0].flatten()
        # print(tvec, ya)
        area = np.trapz(x = tvec, y = abs(ya - tvec)) / int(max(max(x_pts_to_graph), max(y_pts_to_graph)))**2
    if not reg:
        area = np.trapz(x = tvec, y = abs(calib.transform(tvec) - tvec))
    
    # Title
    if title:
        fig.update_layout(title = f'Logit Reliability Diagram - {title}<br><sup>Miscalibration Area: {area*100:.5f}%</sup>')
    else:
        fig.update_layout(title = f'Logit Reliability Diagram<br><sup>Miscalibration Area: {area*100:.5f}%</sup>')
    
    if not reg:
        fig.update_layout(yaxis_range = [-0.1, 1.1], xaxis_range = [-0.1, 1.1])

    # Axis Labels
    if reg:
        fig.update_layout(xaxis_title = 'Actual', yaxis_title = 'Predicted')
    elif not reg:
        fig.update_layout(xaxis_title = 'Actual (Probability)', yaxis_title = 'Predicted (Probability)')
    
    fig.show()
    
    return fig, area

def my_logit(vec, base=np.exp(1), eps=1e-16):
    vec = np.clip(vec, eps, 1-eps)
    return (1/np.log(base)) * np.log(vec/(1-vec))

def my_logistic(vec, base=np.exp(1)):
    return 1/(1+base**(-vec))