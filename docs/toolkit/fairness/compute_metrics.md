---
layout: default
title: xrai_toolkit.fairness.fairness_algorithm.compute_metrics
parent: Fairness & Performance
grand_parent: Toolkit
nav_order: 4
---

# xrai_toolkit.fairness.fairness_algorithm.compute_metrics
**[xrai_toolkit.fairness.fairness_algorithm.compute_metrics(dataset_true, dataset_pred, unprivileged_groups, privileged_groups, disp = True)](https://github.com/gaberamolete/XRAIToolkit/blob/main/fairness/fairness_algorithm.py)**


Compute the fairness metrics for classification


**Parameters:**
- dataset_true (aif360.StandardDataset): The ground truth data
- dataset_pred (aif360.StandardDataset): The prediction data
- unprivileged_groups (List): List of unprivileged group
- privileged_groups (List): List of privileged group
- disp (bool): Print the metrics or not

**Returns:**
- metrics (OrderedDict): Dictionary of the different metrics and their corresponding scores.