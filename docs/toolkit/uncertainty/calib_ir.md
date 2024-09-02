---
layout: default
title: xrai_toolkit.uncertainty.calibration.calib_ir
parent: Uncertainty
grand_parent: Toolkit
has_children: false
nav_order: 5
---

# xrai_toolkit.uncertainty.calibration.calib_ir
**[xrai_toolkit.uncertainty.calibration.calib_ir(y_pred,y_true,reg)](https://github.com/gaberamolete/xrai_toolkit/blob/main/uncertainty/calibration.py)**


Isotonic Regression, similar to Histogram Binning but with dynamic bin sizes and boundaries. A piecewise constant function gets for to ground truth labels sorted by given confidence estimates.


**Parameters:**
- y_pred (numpy.ndarray): Array or series of predicted values.
- y_true (numpy.ndarray): Array or Series with ground truth labels.
- reg (bool): Determines if y's given are from a regression or classification problem. Defaults to False.

**Returns:**
- ir: Isotonic Regression object.
- ir_calibrated: Recalibrated values under Isotonic Regression, given via array.
