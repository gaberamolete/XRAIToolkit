---
layout: default
title: xrai_toolkit.uncertainty.uct.uct_plot_adversarial_group_calibration
parent: Uncertainty
grand_parent: Toolkit
has_children: false
nav_order: 21
---

# xrai_toolkit.uncertainty.uct.uct_plot_adversarial_group_calibration
**[xrai_toolkit.uncertainty.uct.uct_plot_adversarial_group_calibration(uct_metrics, show = False)](https://github.com/gaberamolete/xrai_toolkit/blob/main/uncertainty/calibration.py)**


Plot the adversarial group calibration plots by varying group size from 0% to 100% of dataset size and recording the worst group calibration error for each group size


**Parameters:**
- uct_metrics (dict): Dictionary of all metrics calculated by the Uncertainty Toolbox
- show (bool): Boolean value to determine if the plot should be shown or not. Default to False.

**Returns:**
- fig (plotly.Figure): plotly figure of the adversarial group calibration

