---
layout: default
title: xrai_toolkit.stability.stability.psi_list
parent: Stability
grand_parent: Toolkit
nav_order: 1
---

# xrai_toolkit.stability.stability.psi_list
**[xrai_toolkit.stability.stability.psi_list(train, test)](https://github.com/gaberamolete/xrai_toolkit/blob/main/stability/stability.py)**


Compares the distribution of the target variable in the test dataset to a training data set that was used to develop the model.


**Parameters:**
- train: pd.DataFrame
- test: pd.DataFrame

**Returns:**
- large: List of variables with large shift
- slight: List of variable with slight shift