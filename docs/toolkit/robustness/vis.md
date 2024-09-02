---
layout: default
title: xrai_toolkit.robustness.art_metrics.visualisation
parent: Robustness
grand_parent: Toolkit
nav_order: 1
---

# xrai_toolkit.robustness.art_metrics.visualisation
**[xrai_toolkit.robustness.art_metrics.visualisation(pdtp_art, shapr_art, pdtp_threshold_value, shapr_threshold_value)](https://github.com/gaberamolete/xrai_toolkit/blob/main/robustness/art_metrics.py)**

    
Calculates the SHAPr metric for a given classifier and dataset


**Parameters:**
- pdtp_art: array of PDTP scores
- shapr_art: array of SHAPr scores
- pdtp_threshold_value: threshold value for PDTP
- shapr_threshold_value: threshold value for SHAPr

**Returns:**
- fig: plotly.Figure visualisation of PDTP
- fig2: plotly.Figure visualisation of SHAPr