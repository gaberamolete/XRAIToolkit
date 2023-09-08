import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import joblib
#import libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from art.metrics import PDTP, SHAPr
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from art.estimators.classification import SklearnClassifier, BlackBoxClassifier, XGBoostClassifier
from sklearn.base import is_classifier, is_regressor

def art_extra_classifiers(models):
    """
    Returns a dictionary of ART classifiers and extra classifiers for the given models. Key is the given model
    name 
    
    Parameters
    ----------
    models: dictionary of models
    """
    
    art_extra_classifier_dict = {}
    #ART classifiers for models 
    try:
    
        for key in models:
            class_name = models[key][-1].__class__.__name__
            model_input = models[key][-1]
            model_source = model_input.__class__.__module__.split('.')[0]

            art_extra_classifier_dict[key] = []

            #check if model is classifier or regressor
            if is_classifier(model_input):
            
                #check if model is from sklearn
                if model_source == 'sklearn':
                    art_extra_classifier_dict[key].append(SklearnClassifier(model = models[key][-1]))
                
                    if class_name == 'RandomForestClassifier':
                        extra_model = RandomForestClassifier()
                        extra_classifier = SklearnClassifier(model = extra_model)

                        art_extra_classifier_dict[key].append(extra_classifier)

                    elif class_name == 'DecisionTreeClassifier':
                        extra_model = DecisionTreeClassifier()
                        extra_classifier = SklearnClassifier(model = extra_model)
                        art_extra_classifier_dict[key].append(extra_classifier)
            
                else:
                    art_extra_classifier_dict[key].append('Model Not Supported')            

            elif is_regressor(model_input):
                #check if model is from sklearn
                if model_source == 'sklearn':
                    art_extra_classifier_dict[key].append(ScikitlearnRegressor(model = models[key][-1]))

                    if class_name == 'DecisionTreeRegressor':
                        extra_model = DecisionTreeRegressor()
                        extra_classifier = ScikitlearnRegressor(model = extra_model)
                        art_extra_classifier_dict[key].append(extra_classifier)

                    if class_name == 'LinearRegressor':
                        extra_model = LinearRegression()
                        extra_classifier = ScikitlearnRegressor(model = extra_model)
                        art_extra_classifier_dict[key].append(extra_classifier)

    except: 
        
        for key in models:
            class_name = models[key].__class__.__name__
            model_input = models[key]
            model_source = model_input.__class__.__module__.split('.')[0]

            art_extra_classifier_dict[key] = []

            #check if model is classifier or regressor
            if is_classifier(model_input):
            
                #check if model is from sklearn
                if model_source == 'sklearn':
                    art_extra_classifier_dict[key].append(SklearnClassifier(model = models[key][-1]))
                
                    if class_name == 'RandomForestClassifier':
                        extra_model = RandomForestClassifier()
                        extra_classifier = SklearnClassifier(model = extra_model)

                        art_extra_classifier_dict[key].append(extra_classifier)

                    elif class_name == 'DecisionTreeClassifier':
                        extra_model = DecisionTreeClassifier()
                        extra_classifier = SklearnClassifier(model = extra_model)
                        art_extra_classifier_dict[key].append(extra_classifier)
            
                else:
                    art_extra_classifier_dict[key].append('Model Not Supported')            

            elif is_regressor(model_input):
                #check if model is from sklearn
                if model_source == 'sklearn':
                    art_extra_classifier_dict[key].append(ScikitlearnRegressor(model = models[key]))

                    if class_name == 'DecisionTreeRegressor':
                        extra_model = DecisionTreeRegressor()
                        extra_classifier = ScikitlearnRegressor(model = extra_model)
                        art_extra_classifier_dict[key].append(extra_classifier)

                    if class_name == 'LinearRegressor':
                        extra_model = LinearRegression()
                        extra_classifier = ScikitlearnRegressor(model = extra_model)
                        art_extra_classifier_dict[key].append(extra_classifier)
                        
    return art_extra_classifier_dict 