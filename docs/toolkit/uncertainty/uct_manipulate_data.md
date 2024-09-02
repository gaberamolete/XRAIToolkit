---
layout: default
title: xrai_toolkit.uncertainty.uct.uct_manipulate_data
parent: Uncertainty
grand_parent: Toolkit
has_children: false
nav_order: 19
---

# xrai_toolkit.uncertainty.uct.uct_manipulate_data
**[xrai_toolkit.uncertainty.uct.uct_manipulate_data(X_train, X_test, Y_train, Y_test, model, reg)](https://github.com/gaberamolete/xrai_toolkit/blob/main/uncertainty/calibration.py)**


Generates the appropriate arrays for the Uncertainty Toolbox.


**Parameters:**
- X_train (pandas.DataFrame): training dataset for X values
- X_test (pandas.DataFrame): test dataset for X values
- Y_train (pandas.DataFrame): training dataset for Y values
- Y_test (pandas.DataFrame): test dataset for Y values
- model: model that was trained
- reg (bool): if model is a regression omodel

**Returns:**
- uct_data_dict (dict): dictionary of the data that is needed for the Uncertainty Toolbox
    - y_pred: predicted values from the model
    - y_std: standard deviation of the predicted values
    - y_mean: mean of the predicted values
    - y_true: 1D Array of labels in the held out dataset
    - y2_mean: 1D array of the predicted means for the held out dataset.
    - y2_std: 1D array of the predicted standard deviations for the held out dataset.

