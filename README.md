# Explainable and Responsible AI Toolkit (v3.0 - 2023-09)
***Developed by AI&I COE***

The **DSAI Explainable and Responsible AI (XRAI) Toolkit** is a complementary tool to the [XRAI Guidelines document](https://unionbankphilippines.sharepoint.com/:b:/s/DataScienceInsightsTeam/EbGWZEJkn7REt1zzHspu-xABsLDpD1eD6mgHMjPJypnzdA?e=wm55U7) <change this link to the GitHub link. The Toolkit provides a one-stop tool for technical tests were identified by packaging widely used open-source libraries into a single platform, according to ADI/DSAI user needs. These tools include Python libraries such as `shap`, `dalex`, `dice-ml`, `alibi`, `netcal`, `aif360`, `scikit-learn`, and UI visualization libraries such as `raiwidgets` and `plotly`. This toolkit aims to: 
- Provide a user interface to guide users step-by-step in the testing process; 
- Support certain binary classification and regression models that use tabular data 
- Produce a basic summary report to help DSAI System Developers and Owners interpret test results 
- Intended to be deployable in the user’s environment

Subsequent versions and updates will be found in the XRAI [GitHub repo](https://github.com/aboitiz-data-innovation/XRAI), accompanied by XRAI Brownbag sessions.

# Introduction
## Assumptions / Limitations
The toolkit is in its second iterations of development. As such, there are limitations which hinders the Toolkit from being able to handle certain models and display other XRAI-related features. These include:
- Only Python users for V3. An R-based Toolkit may be released according to demand and necessity for later versions 
- This is mostly intended for classification and regression models; unsupervised models may run on some functions but are not guaranteed to work or provided correct insight.
- Certain features may be discussed in Guidelines V3 but are not yet in Toolkit V3
- This toolkit does not define ethical standards. It provides a way for DSAI System Developers and Owners to demonstrate their claims about the performance of their DSAI systems according to the XRAI principles

## Inputs
Our interactive toolkit only needs two main inputs before any major analysis:
- Model (.pkl or .sav)
- Data (train, test) (.csv) 

We intend for the user to have inputs mostly on the XRAI-related functions. However, we need the user to manually input **the names of train and test file**, in addition to target variable name. Prompts will be shown later in the notebook where you will need to load. In the shared folder we have provided sample model and data (test_data.csv, train_data.csv, finalized_model.pkl). For testing for different models and data you may just replace files.

## Features found in the Toolkit
- EDA – User-friendly, no-code custom visualizations to help you analyze your data's quality and distribution
- Fairness Preprocessing – Ready-made functions to clean and catch potential bias in data prior to model development
- Fairness Metrics – Find out which factors are disadvantaged by the model, and receive recommendations on how to mitigate unfairness
- Model Performance – Statistics of predicted outcome, dataset features, and error groups
- Local Explanations – Understand how a model affects individual explanations with a variety of techniques
- Global Explanations – Understand how a model is shaped
- Stability Analysis – OOT validation, data and model quality, how data and concepts decays over time
- Outlier Analysis – resampling, pre-model and post-model views
- What-if Analysis – counterfactual measures; showing how result for individual data point may change if one of its feature values was changed
- Robustness – Check if system can function despite unexpected inputs 
- Uncertainty – Models may not always make perfect predictions, and there can be some level of doubt or variability associated with their result

# Running in Local Machine using Conda Environment
To easily create the environment for XRAI in your local machine, you can run the following in a command prompt or terminal.
```
conda env create -f environment.yml
```
The `environment.yml` is the latest environment with no dependency issues, while the `latest_env.yml` has some due to the addition of the Responsible AI library. Some components in the dashboard may not be available from `environment.yml`.

After creating an environment, you can create a kernel to enable the notebooks to use the conda environment using the following lines.
```
python -m ipykernel install --user --name=XRAI
```

# Requirements
```
python = 3.9.0
```
Ensure that you have the correct libraries and dependencies installed by checking `requirements.txt`.

