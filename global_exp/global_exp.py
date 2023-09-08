import dalex as dx
import numpy as np 
import shap
import pandas as pd
import warnings
from collections import Counter
import sklearn
import matplotlib.pyplot as plt

def dalex_exp(model, X_train, y_train, X_test, idx):
    '''
    Explanation object.
    
    model: model object, full model, can be with preprocessing steps
    X_train: DataFrame
    y_train: DataFrame
    X_test: DataFrame
    '''
    
    exp = dx.Explainer(model, X_train, y_train)
    exp.predict(X_test)
    
    obs = X_test.iloc[idx, :]
    
    return exp, obs

def pd_profile(exp, variables = None, var_type = 'numerical', groups = None, random_state = 42, N = 300, labels = False):
    '''
    Creates a partial-dependence plot and outputs the equivalent table. User may specify the variables to showcase.
    Note that this can only help explain variables that are of the same data type at the same time
    i.e. you may not analyze a numerical and categorical variable in the same run.
    
    exp: explanation object
    variables: list, list of variables to be explained utilizing the methods. The default is 'None', which will make it run through all variables
    var_type: can either be 'numerical' or 'categorical'
    groups: specify a single categorical variable not in the 'variables' list that will be used as a group. Defaults to 'None'
    random_state: defines the random state in which the number of observations to be sampled. Defaults to 42 for reproducibility.
    N: Number of observations to be sampled with. Defaults to 300. Writing 'None' will have the function use all data, which may be computationally expensive.
    labels: boolean. If True, will change label to 'PD profiles'
    '''
    
    if groups:
        if groups not in exp.data.select_dtypes(exclude = np.number).columns.tolist():
            print('Please specify a categorical variable in `groups`.')
            return 'Please specify a categorical variable in `groups`.'
        
    pd = exp.model_profile(type = 'partial', variable_type = var_type, groups = groups,
                              variables = variables, random_state = random_state,
                              N = N)
    if labels:
        pd.result['_label_'] = 'PD profiles'
    pd.plot()
    return pd.result, pd.plot(show=False)

def var_imp(exp, loss_function = 'rmse', groups = None, N = 1000, B = 10, random_state = 42):
    '''
    A permutation-based approach in explaining variable importance to the model.
    
    ---
    exp: explanation object
    loss_function: manner in which the function will calculate loss. Can choose from 'rmse', 'mae', 'mse', 'mad', '1-auc'. Defaults to 'rmse'.
    groups: specify a single categorical variable not in the 'variables' list that will be used as a group. Defaults to 'None'
    random_state: defines the random state in which the number of observations to be sampled. Defaults to 42 for reproducibility.
    N: Number of observations to be sampled with. Defaults to 300. Writing 'None' will have the function use all data, which may be computationally expensive.
    '''
    
    if groups:
        # Check if groups is properly inputted
        if type(groups) is not dict:
            return print('Please input a dictionary with a key-value pair of variable name and variable list for the variable `group`.')
    
        # Check if variables are in the data
        temp = []
        for v in groups.values():
            for col in v:
                if col not in exp.data.columns.tolist():
                    print(f'`{col}` - This variable is not in the data')
                    break
                temp.append(col)
                
        # Check if variables repeat
        if len(temp) > np.unique(temp).size:
            repeat = [k for k, v in Counter(temp).items() if v > 1]
            print(f'The variables {repeat} appear more than once in the groups dictionary.') 
    
    vi = exp.model_parts(variable_groups = groups, loss_function = loss_function,
                         N = N, B = B, random_state = random_state)
    vi.plot()
    return vi.result.sort_values(by = 'dropout_loss', ascending = False), vi.plot(show=False)

def ld_profile(exp, variables = None, var_type = 'numerical', groups = None, random_state = 42, N = 300, labels = False):
    '''
    Creates a local-dependence plot and outputs the equivalent table. User may specify the variables to showcase.
    Note that this can only help explain variables that are of the same data type at the same time
    i.e. you may not analyze a numerical and categorical variable in the same run.
    
    exp: explanation object
    variables: list, list of variables to be explained utilizing the methods. The default is 'None', which will make it run through all variables
    var_type: can either be 'numerical' or 'categorical'
    groups: specify a single categorical variable not in the 'variables' list that will be used as a group. Defaults to 'None'
    random_state: defines the random state in which the number of observations to be sampled. Defaults to 42 for reproducibility.
    N: Number of observations to be sampled with. Defaults to 300. Writing 'None' will have the function use all data, which may be computationally expensive.
    labels: boolean. If True, will change label to 'LD profiles'
    '''
    
    if groups:
        if groups not in exp.data.select_dtypes(exclude = np.number).columns.tolist():
            return print('Please specify a categorical variable in `groups`.')
        
    ld = exp.model_profile(type = 'conditional', variable_type = var_type, groups = groups,
                              variables = variables, random_state = random_state,
                              N = N)
    if labels:
        ld.result['_label_'] = 'LD profiles'
    ld.plot()
    return ld.result, ld.plot(show=False)

def al_profile(exp, variables = None, var_type = 'numerical', groups = None, random_state = 42, N = 300, labels = False):
    '''
    Creates a accumulated-local plot and outputs the equivalent table. User may specify the variables to showcase.
    Note that this can only help explain variables that are of the same data type at the same time
    i.e. you may not analyze a numerical and categorical variable in the same run.
    
    exp: explanation object
    variables: list, list of variables to be explained utilizing the methods. The default is 'None', which will make it run through all variables
    var_type: can either be 'numerical' or 'categorical'
    groups: specify a single categorical variable not in the 'variables' list that will be used as a group. Defaults to 'None'
    random_state: defines the random state in which the number of observations to be sampled. Defaults to 42 for reproducibility.
    N: Number of observations to be sampled with. Defaults to 300. Writing 'None' will have the function use all data, which may be computationally expensive.
    labels: boolean. If True, will change label to 'LD profiles'
    '''
    
    if groups:
        if groups not in exp.data.select_dtypes(exclude = np.number).columns.tolist():
            return print('Please specify a categorical variable in `groups`.')
        
    al = exp.model_profile(type = 'accumulated', variable_type = var_type, groups = groups,
                              variables = variables, random_state = random_state,
                              N = N)
    if labels:
        al.result['_label_'] = 'AL profiles'
    al.plot()
    return al.result, al.plot(show=False)

def compare_profiles(exp, variables = None, var_type = 'numerical', groups = None, random_state = 42, N = 300):
    '''
    Compares partial-dependence, local-dependence, and accumulated-local profiles. User may specify the variables to showcase.
    Note that this can only help explain variables that are of the same data type at the same time
    i.e. you may not analyze a numerical and categorical variable in the same run.
    
    exp: explanation object
    variables: list, list of variables to be explained utilizing the methods. The default is 'None', which will make it run through all variables
    var_type: can either be 'numerical' or 'categorical'
    groups: specify a single categorical variable not in the 'variables' list that will be used as a group. Defaults to 'None'
    random_state: defines the random state in which the number of observations to be sampled. Defaults to 42 for reproducibility.
    N: Number of observations to be sampled with. Defaults to 300. Writing 'None' will have the function use all data, which may be computationally expensive.
    labels: boolean. If True, will change label to 'LD profiles'
    '''
    
    if groups:
        if groups not in exp.data.select_dtypes(exclude = np.number).columns.tolist():
            return print('Please specify a categorical variable in `groups`.')
    
    pd = exp.model_profile(type = 'partial', variable_type = var_type, groups = groups,
                              variables = variables, random_state = random_state,
                              N = N)
    pd.result['_label_'] = pd.result['_label_'] + '_PD profiles'
    ld = exp.model_profile(type = 'conditional', variable_type = var_type, groups = groups,
                              variables = variables, random_state = random_state,
                              N = N)
    ld.result['_label_'] = ld.result['_label_'] + '_LD profiles'
    al = exp.model_profile(type = 'accumulated', variable_type = var_type, groups = groups,
                              variables = variables, random_state = random_state,
                              N = N)
    al.result['_label_'] = al.result['_label_'] + '_AL profiles'
    
    pd.plot([ld, al])
    return pd.plot([ld, al], show=False)

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
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
        # print(l_transformers)
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
        #print(l_transformers)
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans, cat_cols)
            #print(_names)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans, cat_cols))
    
    return feature_names

def initiate_shap_glob(X, model_global, preprocessor = None, samples = 100):
    """
    Initiate instance for SHAP global object. Defaults to a TreeExplainer, but will redirect to a KernelExplainer if exceptions/errors are encountered.
    
    Parameters
    ------------
    X: DataFrame, data to be modelled
    model_global: Model object to be used to create predictions. X is fed into this object. Note that this only takes in the actual model, so Pipelines with preprocessing steps will not be dealt with properly.
    preprocessor: Object needed if X needs to be preprocessed (i.e. in a Pipeline) before being fed into the model. Defaults to None.
    samples: int, number of samples to be included in the KernelExplainer method. Defaults to 100.
    
    Returns
    ------------
    explainer: Shap explainer object.
    shap_values: Generated shap values, to be used for other shap-generate graphs.
    feature_names: Names of features generated from the preprocessed X DataFrame. Useful for other shap-generated graphs.
    """
    
    if preprocessor:
        X_proc = preprocessor.transform(X)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            p_ind = preprocessor[-1].get_support(indices = True)
            fn = preprocessor.get_feature_names_out()
            feature_names = [fn[x] for x in p_ind]
    else:
        X_proc = X.copy()

    try:
        explainer = shap.TreeExplainer(model_global)
    except:
        explainer = shap.KernelExplainer(model_global.predict, X_proc[:samples], keep_index = True)
        
    shap_values = explainer.shap_values(X_proc)
        
    return explainer, shap_values, feature_names

def shap_bar_glob(shap_values, class_ind, X_proc, feature_names = None, class_names = None, reg = False):
    '''
    Returns a bar plot of a shap value.
    '''
    if reg == False:
        if not class_names:
            return print('Please specify class names in a list.')
        s = plt.figure()
        shap.summary_plot(shap_values[class_ind], X_proc, feature_names = feature_names,
                    class_names = class_names, show = False, plot_type = 'bar')
        plt.title(f'Global SHAP Values for {class_names[class_ind]} class', fontsize = 18)
    else:
        s = plt.figure()
        shap.summary_plot(shap_values, X_proc, feature_names = feature_names,
                    show = False, plot_type = 'bar')
        plt.title('Global SHAP Values', fontsize = 18)
    #plt.show()
    return s

def shap_summary(shap_values, class_ind, X_proc, class_names, feature_names = None, reg = False):
    '''
    Returns a summary plot.
    '''
    
    if reg == False:
        s = plt.figure()
        shap.summary_plot(shap_values[class_ind], X_proc, feature_names = feature_names,
                    class_names = class_names, show = False)
        plt.title(f'Global SHAP Values for {class_names[class_ind]} class', fontsize = 18)
    else:
        s = plt.figure()
        shap.summary_plot(shap_values, X_proc, feature_names = feature_names,
                    show = False)
        plt.title('Global SHAP Values', fontsize = 18)
    #plt.show()
    return s

def shap_dependence(shap_values, class_ind, X_proc, feature_names, class_names,
                   shap_ind, int_ind = None, reg = False):
    '''
    Returns a dependence plot comparing a value and its equivalent shap values.
    The user may also compare these with another variable as an interaction index.
    '''
    
    for col in [shap_ind, int_ind]:
        if not int_ind:
            continue
        if col not in feature_names:
            return print(f'{col} is not in the feature names list. Please specify accordingly.')
        
    if reg == False:
        s, ax = plt.subplots()
        s = shap.dependence_plot(shap_ind, shap_values[class_ind], X_proc,
                            interaction_index = int_ind, feature_names = feature_names, show = False, ax=ax)
        plt.title(f'Global SHAP dependence plot for {class_names[class_ind]} class', fontsize = 16)
        return s
    else:
        s, ax = plt.subplots()
        shap.dependence_plot(shap_ind, shap_values, X_proc,
                            interaction_index = int_ind, feature_names = feature_names, show = False, ax=ax)
        plt.title(f'Global SHAP dependence plot', fontsize = 16)
    #plt.show()
    return s

def shap_force_glob(explainer, shap_values, X_proc, class_ind, class_names,
                    feature_names = None, reg = False, samples = 100):
    '''
    
    '''
    if reg == False:
        idx = np.random.randint(shap_values[class_ind].shape[0], size = samples)
        s = shap.force_plot(explainer.expected_value[class_ind], shap_values[class_ind][idx, :],
                        X_proc[idx, :], feature_names = feature_names, show=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{s.html()}</body>"
        shap.force_plot(explainer.expected_value[class_ind], shap_values[class_ind][idx, :],
                        X_proc[idx, :], feature_names = feature_names, show=True)
    else:
        idx = np.random.randint(shap_values.shape[0], size = samples)
        s = shap.force_plot(explainer.expected_value, shap_values[idx, :],
                        X_proc[idx, :], feature_names = feature_names, show=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{s.html()}</body>"
        shap.force_plot(explainer.expected_value, shap_values[idx, :],
                        X_proc[idx, :], feature_names = feature_names, show=True)
    return shap_html