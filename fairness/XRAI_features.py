'''
Required functions for User interface (UI)
Inputs: data sets and model

'''

# Import required libraries xrai features
from responsibleai import RAIInsights
# From responsibleai.feature_metadata import FeatureMetadata
from raiutils.cohort import Cohort, CohortFilter, CohortFilterMethods

# XRAI feature
def xrai_features(model, train_data, test_data_sample, target_feature, task_type = 'classification', categorical_features = None):
    '''
    input: model, train_data, test_data_sample, target_feature
        :test_sample_data- here you can either use entire test data or can take random sample of test data for better visualization
    output: 
    :rai_insights- explainer() to give explaination of data statistics, model overview and feature importance
    error_analysis() it gives detailed analysis of automated and customised error segments
    cohort_list - default and required cohorts/groups are true_y values, pred_y values and Cohort on index of the row in the dataset
    '''

    # Add 's1' as an identity feature
    '''
    param identity_feature_name: Name of the feature which helps to uniquely identify a row or instance in user input dataset. 
    Feature metadata for the train/test dataset to identify different kinds of features in the dataset.
    '''
    # feature_metadata = FeatureMetadata(identity_feature_name='s1')


    # RAI_Insights 
    '''
    This is responsibleai in-built library function that requires input- train, test data, model and target_feature
    We can choose task as classification or regression
    If we would like to get better understanding of categorical features we have scope to provide categorical features along with category
    This function manily takes input parameter and create a class for further features analysis

    Defines the top-level Model Analysis API.
    Use RAIInsights to analyze errors, explain the most important features, data exploration 
    and model evaluation in a single API.
    ''' 
    rai_insights = RAIInsights(model, train_data, test_data_sample, target_feature,
                               task_type = task_type, categorical_features = categorical_features)

    # define XRAI features
    # Interpretability : Data statistics Exploration, feature importance, and model overview
    # rai_insights.explainer.add()
    # Error Analysis
    rai_insights.error_analysis.add()

    # compute features
    rai_insights.compute()
    

    # Cohort analysis
    '''
    Here we can also create custom cohort considering different variables and their their subgrouprs
    For example: age>40 and age<70 
                 or combination of 2 or more variables subgroups
    We have same capability in UI as well. 
    In coming version we are going to add capabilities via code.
    '''
    # Cohort on index of the row in the dataset
    cohort_filter_index = CohortFilter(method=CohortFilterMethods.METHOD_LESS,arg=[20],column='Index')
    user_cohort_index = Cohort(name='Cohort Index')
    user_cohort_index.add_cohort_filter(cohort_filter_index)

    # Cohort on predicted target value
    cohort_filter_predicted_y = CohortFilter(method=CohortFilterMethods.METHOD_INCLUDES,arg=[1],column='Predicted Y')
    user_cohort_predicted_y = Cohort(name='Cohort Predicted Y')
    user_cohort_predicted_y.add_cohort_filter(cohort_filter_predicted_y)

    # Cohort on true target value
    cohort_filter_true_y = CohortFilter(method=CohortFilterMethods.METHOD_INCLUDES,arg=[1],column='True Y')
    user_cohort_true_y = Cohort(name='Cohort True Y')
    user_cohort_true_y.add_cohort_filter(cohort_filter_true_y)

    cohort_list = [user_cohort_index, user_cohort_predicted_y, user_cohort_true_y]

    return rai_insights, cohort_list
