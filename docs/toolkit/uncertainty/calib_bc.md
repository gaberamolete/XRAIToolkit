---
layout: default
title: xrai_toolkit.uncertainty.calibration.calib_bc
parent: Uncertainty
grand_parent: Toolkit
has_children: false
nav_order: 2
---

# xrai_toolkit.uncertainty.calibration.calib_bc
**[xrai_toolkit.uncertainty.calibration.calib_bc(y_pred,y_true,reg)](https://github.com/gaberamolete/xrai_toolkit/blob/main/uncertainty/calibration.py)**


Beta Calibration, a well-founded and easily implemented improvement on Platt scaling for binary classifiers. Assumes that per-class scores of classifier each follow a beta distribution.


**Parameters:**
- y_pred (numpy.ndarray): Array or series of predicted values.
- y_true (numpy.ndarray): Array or Series with ground truth labels.
- reg (bool): Determines if y's given are from a regression or classification problem. Defaults to False.

**Returns:**
- bc: Beta Calibration object.
- bc_calibrated: Recalibrated values under Beta Calibration, given via array.
