import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers

from qii.qii import QII
from qii.predictor import QIIPredictor
from qii.qoi import QuantityOfInterest

import dalex as dx

import shap

import plotly.express as px

from IPython.display import HTML, display
import warnings


def dice_exp(X_train, y_train, model, target, backend = 'sklearn', model_type = 'classifier'):
    '''
    Initialize dice experiment CF.
    
    Parameters
    ----------
    X_train: DataFrame,
    y_train: DataFrame or Series, contains target column and list of target variables
    model: model object, can be sklearn, tensorflow, or keras
    target: str, name of target variable
    backend: str, "TF1" ("TF2") for TensorFLow 1.0 (2.0), "PYT" for PyTorch implementations, "sklearn" for Scikit-Learn implementations of standard DiCE (https://arxiv.org/pdf/1905.07697.pdf). For all other frameworks and implementations, provide a dictionary with "model" and "explainer" as keys, and include module and class names as values in the form module_name.class_name. For instance, if there is a model interface class "XGBoostModel" in module "xgboost_model.py" inside the subpackage dice_ml.model_interfaces, and dice interface class "DiceXGBoost" in module "dice_xgboost" inside dice_ml.explainer_interfaces, then backend parameter should be {"model": "xgboost_model.XGBoostModel", "explainer": dice_xgboost.DiceXGBoost}.
    '''
    
    train_dataset = pd.concat([X_train, y_train],axis=1)
    
    cont = X_train.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    cat = X_train.select_dtypes(exclude = ['int64', 'float64']).columns.tolist()
    
    d = dice_ml.Data(dataframe = train_dataset, continuous_features = cont,
                    categorical_features = cat, outcome_name = target)
    
    m = dice_ml.Model(model = model, backend = backend, model_type=model_type)
    exp = dice_ml.Dice(d, m, method = 'random')
    return exp

def exp_cf(X, exp, total_CFs = 2, desired_range = None, desired_class = 'opposite', features_to_vary = 'all',
           permitted_range = None, reg = False):
    '''
    Generates counterfactuals
    
    Parameters
    ----------
    X: dataframe, rows of dataset to be analyzed via CF
    exp: object created by dice_exp()
    total_CFs: int, number of CFs to be generated
    desired_range: list of shape (2,), range of desired output for regression case
    desired_class: int or str. Desired CF class - can take 0 or 1. Default value is 'opposite' to the outcome class of query_instance for binary classification. Specify the class name for non-binary classification.
    features_to_vary: List of names of features to vary. Defaults to all.
    permitted_range: Dictionary with key-value pairs of variable name and ranges. Defaults to None.
    '''
    
    try:
        if (reg == False):
            e = exp.generate_counterfactuals(X, total_CFs = total_CFs, desired_class = desired_class,
                                            features_to_vary = features_to_vary, permitted_range = permitted_range)
            e.visualize_as_dataframe(show_only_changes = True)
            return e
        elif (reg == True):
            e = exp.generate_counterfactuals(X, total_CFs = total_CFs, desired_range = desired_range,
                                            features_to_vary = features_to_vary, permitted_range = permitted_range)
            e.visualize_as_dataframe(show_only_changes = True)
            return e
        else:
            print('Not compatible model.')
    except Exception:
        print('No counterfactuals found for any of the query points! Kindly check your configuration.')
        return 'No counterfactuals found for any of the query points! Kindly check your configuration.'

class Predictor(QIIPredictor):
    """
    Class used to initialize QII object.
    """
    def __init__(self, predictor):
        super(Predictor, self).__init__(predictor)
        
    def predict(self, x):
        # predict the label for instance x
        return self._predictor.predict(x) 

def get_feature_names(column_transformer, cat_cols):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans, cat_cols):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['y%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                
                return [name + "__" + f for f in column]
        # print(trans)
        f = [name + "__" + f for f in trans.get_feature_names(cat_cols)]
        # print(f)
        return f 
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    try:
        if type(column_transformer) == sklearn.pipeline.Pipeline:
            l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
            # print(l_transformers)
        else:
            # For column transformers, follow the original method
            l_transformers = list(column_transformer._iter(fitted=True))
            # print(l_transformers)

        for name, trans, column, _ in l_transformers: 
            if type(trans) == sklearn.pipeline.Pipeline:
                # Recursive call on pipeline
                _names = get_feature_names(trans, cat_cols)
                # print(_names)
                # if pipeline has no transformer that returns names
                if len(_names) == 0:
                    _names = [name + "__" + f for f in column]
                feature_names.extend(_names)
            else:
                feature_names.extend(get_names(trans, cat_cols))
    except:
        feature_names = list(column_transformer.feature_names_in_)
    
    return feature_names

def exp_qii(model, X, idx, preprocessor = None, method = 'banzhaf',
            plot = True, pool_size = 100, n_samplings = 50, cat_cols = None):
    '''
    An alternate variable-importance measure using Quantity of Interest Method.
    
    Parameters
    ------------
    model: model object, must be model only and have preprocessing steps beforehand.
    X: DataFrame on which `model` trains on.
    idx: int, row of DataFrame/numpy.ndarray object to be observed.
    preprocessor: preprocessor object. Defaults to None.
    cat_cols: categorical columns found in X. Defaults to None.
    method: method of QII processing. Default is `banzhaf`, but can be `shapley`.
    plot: If user wants a plot of the QII values to be automatically generated. Defaults to True.
    pool_size: no. of instances to be sampled from. Defaults to 100.
    n_samplings: no. of samplings. Defaults to 50.
    
    Returns
    ------------
    vals_df: DataFrame of features from preprocessed dataset and their corresponding QII (importance) variables.
    fig: Interactive figure object equivalent of `vals_df`.
    '''
    
    if preprocessor:
        X_proc = preprocessor.transform(X)
        try:
            feature_names = preprocessor.get_feature_names_out()
            print('Preprocessing - Normal')
        except:
            try:
                p_ind = preprocessor[-1].get_support(indices = True)
                fn = preprocessor[0].get_feature_names_out()
                feature_names = [fn[x] for x in p_ind]
                print('Preprocessing - Steps')
            except:
                try:
                    feature_names = get_feature_names(preprocessor, cat_cols)
                    print('Preprocessing (old) - Normal')
                except:
                    p_ind = preprocessor[-1].get_support(indices = True)
                    fn = get_feature_names(preprocessor[0], cat_cols)
                    feature_names = [fn[x] for x in p_ind]
                    print('Preprocessing (old) - Steps')
    else:
        X_proc = X.copy()
        feature_names = X_proc.columns.tolist()
        print('Preprocessing - None')
    
    predictor = Predictor(model)
    quantity_of_interest = QuantityOfInterest()
    n_features = X_proc.shape[1]
    
    qii = QII(X_proc, n_features, quantity_of_interest)
    
    idx = idx
    x_0 = X_proc[idx:idx + 1]
    
    print(f'QII with {method} method:')
    vals = qii.compute(x_0 = x_0, predictor = predictor, 
                       show_approx = True, evaluated_features = None,
                       data_exhaustive = False, feature_exhaustive = False,
                       method = method, pool_size = 100, n_samplings = 100)
    print(f'{method}: \n{vals}\n\n')
        
    
    vals2 = list(vals.values())
    
    vals_df = pd.DataFrame(data = {
        'columns': feature_names,
        'values': vals2
    }).sort_values(by = 'values', ascending = False)
    
    if plot:
        fig = px.bar(vals_df, y = 'columns', x = 'values', text_auto = '.4f')
        fig.update_traces(textposition = 'outside')
        fig.update_layout(autosize = True, height = vals_df.shape[0] * 25)
        fig.show()
    else:
        fig = None
    
    return vals_df, fig

def dalex_exp(model, X_train, y_train, X_test, idx):
    '''
    Dalex-related explanation object.
    
    Parameters
    ------------
    model: model object, full model, can be with preprocessing steps
    X_train: DataFrame
    y_train: DataFrame
    X_test: DataFrame
    
    Returns
    ------------
    exp: explanation object
    obs: single Dalex-enabled observation
    '''
    
    exp = dx.Explainer(model, X_train, y_train)
    exp.predict(X_test)
    
    obs = X_test.iloc[idx, :]
    
    return exp, obs

def break_down(exp, obs, order = None, random_state = 42, N = None, labels = None):
    '''
    The plot presents variable attribution to model performance by highlighting the importance of order.
    The default function preselects variable order based on local accuracy.
    Users can also select their own variable order with `order`.
    
    Parameters
    ------------
    exp: explanation object
    obs: single Dalex-enabled observation
    order: order of variables to be input to the function. Defaults to None.
    random_state: defines the random state in which the number of observations to be sampled. Defaults to 42 for reproducibility.
    N: Number of observations to be sampled with. Defaults to None. Writing an int will have the function use N observations of data expensive.
    labels: label attached to the plot. Defaults to None.
    
    Returns
    ------------
    result: DataFrame of the results from the break down plot.
    '''
    
    if order:
        if not isinstance(order, list):
            return print("Your object is not a list.")
        order = np.array(order)
    
    bd = exp.predict_parts(obs, type = 'break_down', order = order,
                          random_state = random_state, N = N)
    if labels:
        bd.result['label'] = bd.result['label'] + f'_{label}'
    
    plot = bd.plot()
    return bd.result, plot

def interactive(exp, obs, count = 10, random_state = 42, N = None, labels = None):
    '''
    Adds interactions to the usual break-down plot.
    
    Parameters
    ------------
    exp: explanation object
    obs: single Dalex-enabled observation
    count: number of new 'interaction' variables to be made. Defaults to 10 to keep it relatively computationally inexpensive.
    random_state: defines the random state in which the number of observations to be sampled. Defaults to 42 for reproducibility.
    N: Number of observations to be sampled with. Defaults to None. Writing an int will have the function use N observations of data expensive.
    labels: label attached to the plot. Defaults to None.
    
    Returns
    ------------
    result: DataFrame of results from the interactive break down plot.
    '''
    
    inter = exp.predict_parts(obs, type = 'break_down_interactions', interaction_preference = count,
                             random_state = random_state, N = N)
    if labels:
        inter.result['label'] = inter.result['label'] + f'_{label}'
    plot = inter.plot()
    return inter.result, plot

def cp_profile(exp, obs, variables = None, var_type = 'numerical', labels = False):
    '''
    Creates a ceteris-paribus plot for a specific observation and outputs the equivalent table.
    User may specify the variables to showcase.
    Note that this can only help explain variables that are of the same data type at the same time
    i.e. you may not analyze a numerical and categorical variable in the same run.
    
    Parameters
    ------------
    exp: explanation object
    obs: single Dalex-enabled observation
    variables: list, list of variables to be explained utilizing the methods. The default is 'None', which will make it run through all variables.
    var_type: can either be 'numerical' or 'categorical'.
    labels: boolean. If True, will change label to 'PD profiles'.
    
    Returns
    ------------
    result: DataFrame of results from the local ceteris-paribus profile.
    '''
    
    if variables:
        if var_type == 'numerical':
            num = exp.data.select_dtypes(include = np.number).columns.tolist()
            for col in variables:
                if col not in num:
                    return print(f'The variable `{col}` is not a numerical column found in the explanation object.')
        if var_type == 'categorical':
            cat = exp.data.select_dtypes(exclude = np.number).columns.tolist()
            for col in variables:
                if col not in cat:
                    return print(f'The variable `{col}` is not a categorical column found in the explanation object.')

    cp = exp.predict_profile(obs)
    plot = cp.plot(variables = variables, variable_type = var_type)
    return cp.result, plot

def initiate_shap_loc(X, model, preprocessor = None, samples = 100, seed = 42, cat_cols = None):
    '''
    Initiate the shap explainer used for local explanations. Defaults to an Independent masker, but will redirect to a TabularPartitions if exceptions/errors are encountered.
    
    Parameters
    ------------
    X: DataFrame, data to be modelled.
    preprocessor: Object needed if X needs to be preprocessed (i.e. in a Pipeline) before being fed into the model. Defaults to None.
    int, number of samples to be included in the TabularPartitions method. Defaults to 100.
    seed: Random state in order to ensure reproducibility. Defaults to 42.
    
    Returns
    ------------
    explainer: Shap explainer object.
    shap_values: Generated shap values, to be used for other shap-generate graphs.
    feature_names: Names of features generated from the preprocessed X DataFrame. Useful for other shap-generated graphs.
    '''
    
    if preprocessor:
        X_proc = preprocessor.transform(X)
        try:
            feature_names = preprocessor.get_feature_names_out()
            print('Preprocessing - Normal')
        except:
            try:
                p_ind = preprocessor[-1].get_support(indices = True)
                fn = preprocessor[0].get_feature_names_out()
                feature_names = [fn[x] for x in p_ind]
                print('Preprocessing - Steps')
            except:
                try:
                    feature_names = get_feature_names(preprocessor, cat_cols)
                    print('Preprocessing (old) - Normal')
                except:
                    p_ind = preprocessor[-1].get_support(indices = True)
                    fn = get_feature_names(preprocessor[0], cat_cols)
                    feature_names = [fn[x] for x in p_ind]
                    print('Preprocessing (old) - Steps')
    else:
        X_proc = X.copy()
        feature_names = X_proc.columns.tolist()
    
    # Explain model's predictions using SHAP values
    try:
        background = shap.maskers.Independent(X_proc)
        print('Background - Independent')
    except:
        background = shap.maskers.TabularPartitions(X_proc, sample = samples)
        print('Background - Tabular Partitions')
        
    explainer = shap.Explainer(model, background, link = shap.links.logit, seed = seed)
    
    # Shap values
    # idx = np.random.randint(X_proc.shape[0], size = samples)
    shap_value_loc = explainer(pd.DataFrame(X_proc, columns = feature_names), check_additivity = False)
    
    return explainer, shap_value_loc, feature_names

def shap_waterfall(shap_value_loc, idx, feature_names = None, class_ind = None, class_names = None, reg = False, show=True):
    '''
    Returns a shap waterfall plot for a specified observation.
    
    Parameters
    ------------
    shap_value_loc: Array of shap values used for the waterfall plot. Generated from the initial local shap instance.
    idx: Index to be used for local observation.
    feature_names: List of features that correspond to the column indices in `shap_value_loc`. Defaults to None, but is highly recommended for explainability purposes.
    class_ind: int, represents index used for classification objects in determining which shap values to show. Regression models do not need this variable. Defaults to None.
    class_names: List of all class names of target feature under a classification model. This will be used with the `class_ind` to indicate what class is being shown. Defaults to None.
    reg: Indicates whether model with which `shap_values_loc` was trained on is a regression or classification model. Defaults to False.
    show: Show the plot or not. Defaults to True.
    
    Returns
    ------------
    s: Shap local waterfall plot figure.
    '''
    
    if reg == False:
        exp_shap = shap.Explanation(values = shap_value_loc.values[:, :, class_ind],
                                    base_values = shap_value_loc.base_values[:, class_ind],
                                    data = shap_value_loc.data,
                                   feature_names = feature_names)
        s = shap.plots.waterfall(exp_shap[idx],
                                 show = show
                                )
        plt.title(f'SHAP Local Waterfall plot on Index {idx}\nfor {class_names[class_ind]} class', fontsize = 16)
    else:
        exp_shap = shap.Explanation(values = shap_value_loc.values,
                                    base_values = shap_value_loc.base_values,
                                    data = shap_value_loc.data,
                                   feature_names = feature_names)
        s = shap.plots.waterfall(exp_shap[idx]
                                 , show = show
                                )
        plt.title(f'SHAP Local Waterfall plot on Index {idx}', fontsize = 16)
    
    return s

def shap_force_loc(shap_value_loc, idx, feature_names = None, class_ind = None, class_names = None, reg = False, show=True):
    '''
    Returns a shap force plot for a specified observation.
    
    Parameters
    ------------
    shap_value_loc: Array of shap values used for the force plot. Generated from the initial local shap instance.
    idx: Index to be used for local observation.
    feature_names: List of features that correspond to the column indices in `shap_value_loc`. Defaults to None, but is highly recommended for explainability purposes.
    class_ind: int, represents index used for classification objects in determining which shap values to show. Regression models do not need this variable. Defaults to None.
    class_names: List of all class names of target feature under a classification model. This will be used with the `class_ind` to indicate what class is being shown. Defaults to None.
    reg: Indicates whether model with which `shap_values_loc` was trained on is a regression or classification model. Defaults to False.
    show: Show the plot or not. Defaults to True.
    
    Returns
    ------------
    s: Shap local force plot figure.
    '''
    
    if reg == False:
        exp_shap = shap.Explanation(values = shap_value_loc.values[:, :, class_ind],
                                    base_values = shap_value_loc.base_values[:, class_ind],
                                    data = shap_value_loc.data,
                                   feature_names = feature_names)
        s = shap.plots.force(exp_shap[idx], show = show, matplotlib = True)
        plt.title(f'SHAP Local Waterfall plot on Index {idx}\nfor {class_names[class_ind]} class', fontsize = 16)
    else:
        exp_shap = shap.Explanation(values = shap_value_loc.values,
                                    base_values = shap_value_loc.base_values,
                                    data = shap_value_loc.data,
                                   feature_names = feature_names)
        s = shap.plots.force(exp_shap[idx], show = show, matplotlib = True)
        plt.title(f'SHAP Local Waterfall plot on Index {idx}', fontsize = 16)    
    return s

def shap_bar_loc(shap_value_loc, idx, feature_names = None, class_ind = None, class_names = None, reg = False, show = True):
    '''
    Returns a bar plot for a specified observation.
    
    Parameters
    ------------
    shap_value_loc: Array of shap values used for the bar plot. Generated from the initial local shap instance.
    idx: Index to be used for local observation.
    feature_names: List of features that correspond to the column indices in `shap_value_loc`. Defaults to None, but is highly recommended for explainability purposes.
    class_ind: int, represents index used for classification objects in determining which shap values to show. Regression models do not need this variable. Defaults to None.
    class_names: List of all class names of target feature under a classification model. This will be used with the `class_ind` to indicate what class is being shown. Defaults to None.
    reg: Indicates whether model with which `shap_values_loc` was trained on is a regression or classification model. Defaults to False.
    show: Show the plot or not. Defaults to True.
    
    Returns
    ------------
    s: Shap local bar plot figure.
    '''
    
    if reg == False:
        exp_shap = shap.Explanation(values = shap_value_loc.values[:, :, class_ind],
                                    base_values = shap_value_loc.base_values[:, class_ind],
                                    data = shap_value_loc.data,
                                   feature_names = feature_names)
        s = plt.figure()
        shap.plots.bar(exp_shap[idx], show = show)
        plt.title(f'SHAP Local Bar plot on Index {idx}\nfor {class_names[class_ind]} class', fontsize = 16)
    else:
        exp_shap = shap.Explanation(values = shap_value_loc.values,
                                    base_values = shap_value_loc.base_values,
                                    data = shap_value_loc.data,
                                   feature_names = feature_names)
        s = plt.figure()
        shap.plots.bar(exp_shap[idx], show = show)
        plt.title(f'SHAP Local Bar plot on Index {idx}', fontsize = 16)    
    return s    
