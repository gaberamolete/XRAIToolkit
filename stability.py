import numpy as np
import pandas as pd

#from river import drift
#from skmultiflow.drift_detection import PageHinkley
from scipy import stats

from evidently.test_suite import TestSuite
from evidently.test_preset import *
from evidently.tests import *
from evidently.tests.utils import approx
from evidently.report import Report
from evidently.metric_preset import *
from evidently.metrics import *
from evidently import ColumnMapping

import alibi
from alibi.utils import gen_category_map
from alibi_detect.cd import MMDDrift, FETDrift, CVMDrift, ChiSquareDrift, TabularDrift, ClassifierDrift
from alibi_detect.cd import ClassifierUncertaintyDrift, RegressorUncertaintyDrift
from alibi_detect.saving import save_detector, load_detector

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    
    def psi(expected_array, actual_array, buckets):
        
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            
            return input
        
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
        
        def sub_psi(e_perc, a_perc):
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
 
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value
 
        for i in range(0, len(expected_percents)):
            psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]))
                               
        return psi_value

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        psi_values = psi(expected, actual, buckets)

    return psi_values

def psi_list(train, test):
    """
    Compares the distribution of the target variable in the test dataset to a training data set that was used to develop the model
    
    Parameters
    ----------
    train: pd.DataFrame
    test: pd.DataFrame
    """
    
    psi_list = []
    top_feature_list=train.columns
    
    large = []
    slight = []
    
    for feature in top_feature_list:
        # Assuming you have a validation and training set
        psi_t = calculate_psi(train[feature], test[feature])
        psi_list.append(psi_t)      
        print('Stability index for column',feature,'is',psi_t)
        if(psi_t <= 0.1):
            print('There is no change or shift in the distributions of both datasets for column {}.\n'.format(feature))
        elif(psi_t > 0.2):
            print('This indicates a large shift in the distribution has occurred between both datasets for column {}.\n'.format(feature))
            large.append(feature)
        else:
            print('This indicates a slight change or shift has occurred for column {}.\n'.format(feature))
            slight.append(feature)
            
    if (len(large) == 0) and (len(slight) == 0):
        print("There is no change or shift in the distributions of both datasets for all columns")
        
    if (len(large) != 0):
        print("There is/are indications that a large shift has occurred between both datasets for column {}".format(large))
        
    if (len(slight) != 0):
        print("There is/are indications that a slight shift has occurred between both datasets for column {}".format(slight))
            
    return large, slight

def generate_psi_df(train, test):
    top_feature_list=train.columns
    df = pd.DataFrame(index=top_feature_list,columns=["Feature","PSI Value", "Shift"])
    for feature in top_feature_list:
        # Assuming you have a validation and training set
        df["Feature"][feature] = feature
        psi_t = calculate_psi(train[feature], test[feature])
        df["PSI Value"][feature] = psi_t
        if(psi_t <= 0.1):
            df["Shift"][feature] = "No Shift"
        elif(psi_t > 0.2):
            df["Shift"][feature] = "Large Shift"
        else:
            df["Shift"][feature] = "Slight Shift"
    return df
        

# def PageHinkley(train, test):
#     """
#     Detects data drift by computing the observed values and their mean up to the current moment. Page-Hinkley does not signal warning zones, only change detections.
#     This detector implements the CUSUM control chart for detecting changes. This implementation also supports the two-sided Page-Hinkley test to detect increasing and decreasing changes in the mean of the input values.
    
#     Parameters:
#     ----------
#     train: pd.DataFrame()
#     test: pd.DataFrame()
#     """
    
#     # Initialize Page-Hinkley
#     ph = drift.PageHinkley()
    
#     index_col = []
#     col_col = []
#     val_col = []
    
#     # Update drift detector and verify if change is detected
#     for col in train.columns:
#         data_stream=[]
#         a = np.array(train[col])
#         b = np.array(test[col])
#         data_stream = np.concatenate((a,b))
#         for i, val in enumerate(data_stream):
#             in_drift, in_warning = ph.update(val)
#             if in_drift:
#                 print(f"Change detected at index {i} for column: {col} with input value: {val}")
#                 index_col.append(i)
#                 col_col.append(col)
#                 val_col.append(val)
    
#     ph_df = pd.DataFrame(data = {
#         'index': index_col,
#         'column': col_col,
#         'value': val_col
#     })
    
#     if (len(ph_df['column']) != 0):
#         print()
#         print("In summary, there is/are data drift happenning at columns [column name, frequencies]:")
#         print(ph_df['column'].value_counts())
#     else:
#         print()
#         print('There is no data drift detected at all columns')
    
#     return ph_df

def ks(train, test, p_value = 0.05):
    """
    The K-S test is a nonparametric test that compares the cumulative distributions of two data sets, in this case, the training data and the post-training data. The null hypothesis for this test states that the data distributions from both the datasets are same. If the null is rejected then we can conclude that there is adrift in the model.
    
    Parameters:
    ----------
    train: pd.DataFrame()
    test: pd.DataFrame()
    p_value: float, defaults to 0.05
    """
    rejected_cols = []
    p_vals = []
    
    for col in train.columns:
        testing = stats.ks_2samp(train[col], test[col])
        p_values = testing[1].round(decimals=4)
        if testing[1] < p_value:
            p_values = testing[1].round(decimals=4)
            print("Result: Column rejected", col, 'at p-value', p_values)
            rejected_cols.append(col)
        p_vals.append(p_values)

    print(f"At {p_value}, we rejected {len(rejected_cols)} column(s) in total")
    
    ks_df = pd.DataFrame(data = {
        'columns': train.columns.tolist(),
        'p_values': p_vals
    })
    
    return ks_df, rejected_cols

###### EVIDENTLY ######


def mapping_columns(current_data, reference_data, model, target, prediction = None, id_cols = None, datetime_cols = None,
                  num_cols = None, cat_cols = None):
    """
    Mapping column types for other Evidently-based functions.
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    model: Model used to train data.
    target: str, name of the target column in both the `current_data` and `reference_data`.
    prediction: str, name of the prediction column in both the `current_data` and `reference_data`. Defaults to None.
    id_cols: List of ID columns found in the data. Defaults to None.
    datetime_cols: List of datetime columns found in the data. Defaults to None.
    num_cols: List of numerical columns found in the data. Defaults to None.
    cat_cols: List of categorical columns found in the data. Defaults to None.
    
    Returns
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    column_mapping: Object that tells other functions how to treat each column based on data type.
    """
    
    column_mapping = ColumnMapping()
    column_mapping.target = target
    
    if not prediction:
        current_data['prediction'] = model.predict(current_data.drop(target, axis=1))
        reference_data['prediction'] = model.predict(reference_data.drop(target, axis = 1))
    else:
        column_mapping.prediction = [prediction]
    
    if id_cols:
        column_mapping.id = id_cols
    if datetime_cols:
        column_mapping.datetime_features = datetime_cols
    if num_cols:
        column_mapping.numerical_features = num_cols
    if cat_cols:
        column_mapping.categorical_featuresc = cat_cols
        
    return current_data, reference_data, column_mapping

def data_drift_dataset_report(current_data, reference_data, report_format = 'json', column_mapping = None,
                     drift_share = 0.5, stattest = None, stattest_threshold = None, cat_stattest = None, cat_stattest_threshold = None,
                     num_stattest = None, num_stattest_threshold = None, per_column_stattest = None, per_column_stattest_threshold = None):
    """
    This evaluates the data drift in each individual column of the current dataset, and determines if the dataset has definite data drift.
    
    Drift detection methods that can be used for `stattest` and other equivalent parameters are as follows:
    - `ks`: Kolmogorov-Smirnov test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `chisquare`: Chi-Square test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `z`: Z-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `wasserstein`: Normalized Wasserstein distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `kl_div`: Kullback-Leibler, numerical and categorical. Returns divergence drift detected when greater than or equal to threshold (default is 0.1).
    - `psi`: Population Stability Index, numerical and categorical. Returns psi_value drift detected when greater than or equal to threshold (default is 0.1).
    - `jensenshannon`: Jensen-Shannon distance, numerical and categorical. Default method for categorical, if there are > 1000 objects. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `anderson`: Anderson-Darling test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `fisher_exact`: Fisher's Exact test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `cramer_von_mises`: Cramer-Von-Mises test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `g-test`: G-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `hellinger`: Normalized Hellinger distance, numerical and categorical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `mannw`: Mann-Whitney U-rank test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `ed`: Energy distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `es`: Epps-Singleton tes, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `t_test`: T-Test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `emperical_mmd`: Emperical-MMD, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `TVD`: Total-Variation-Distance, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    A combination of these tests are used if `stattest` and other equivalent parameters are not explicitly specified.
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type.
    drift_share: Threshold to determine when dataset achieves data drift. Calculated by dividing the number of columns with detected data drift over all total columns. Defaults to 0.5.
    stattest: Defines the drift detection method for a given column (if a single column is tested), or all columns in the dataset (if multiple columns are tested).
    stattest_threshold: Sets the drift threshold in a given column or all columns. The threshold meaning varies based on the drift detection method, e.g., it can be the value of a distance metric or a p-value of a statistical test.
    cat_stattest: Sets the drift method for all categorical columns in the dataset.
    cat_stattest_threshold: Sets the threshold for all categorical columns in the dataset.
    num_stattest: Sets the drift method for all numerical columns in the dataset.
    num_stattest_threshold: Sets the threshold for all numerical columns in the dataset.
    per_column_stattest: Sets the drift method for the listed columns (accepts a dictionary).
    per_column_stattest_threshold: Sets the threshold for the listed columns (accepts a dictionary).
    
    Returns
    ------------
    data_drift_report: Interactive visualization object containing the data drift report
    ddr: Report 
    """
    data_drift_report = Report(metrics=[
        DatasetDriftMetric(drift_share = drift_share, stattest = stattest, stattest_threshold = stattest_threshold,
                           cat_stattest = cat_stattest, cat_stattest_threshold = cat_stattest_threshold,
                           num_stattest = num_stattest, num_stattest_threshold = num_stattest_threshold,
                           per_column_stattest = per_column_stattest, per_column_stattest_threshold = per_column_stattest_threshold),
        DataDriftTable(stattest = stattest, stattest_threshold = stattest_threshold,
                       cat_stattest = cat_stattest, cat_stattest_threshold = cat_stattest_threshold,
                       num_stattest = num_stattest, num_stattest_threshold = num_stattest_threshold,
                       per_column_stattest = per_column_stattest, per_column_stattest_threshold = per_column_stattest_threshold),
    ])

    data_drift_report.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if report_format == 'json':
        ddr = data_drift_report.json()
    elif report_format == 'dict':
        ddr = data_drift_report.as_dict()
    elif report_format == 'DataFrame':
        ddr = data_drift_report.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        ddr = None
    
    return data_drift_report, ddr

def data_drift_column_report(current_data, reference_data, column, report_format = 'json', 
                             stattest = None, stattest_threshold = None):
    """
    This evaluates the data drift in one column of the current dataset.

    Drift detection methods that can be used for `stattest` and other equivalent parameters are as follows:
    - `ks`: Kolmogorov-Smirnov test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `chisquare`: Chi-Square test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `z`: Z-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `wasserstein`: Normalized Wasserstein distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `kl_div`: Kullback-Leibler, numerical and categorical. Returns divergence drift detected when greater than or equal to threshold (default is 0.1).
    - `psi`: Population Stability Index, numerical and categorical. Returns psi_value drift detected when greater than or equal to threshold (default is 0.1).
    - `jensenshannon`: Jensen-Shannon distance, numerical and categorical. Default method for categorical, if there are > 1000 objects. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `anderson`: Anderson-Darling test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `fisher_exact`: Fisher's Exact test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `cramer_von_mises`: Cramer-Von-Mises test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `g-test`: G-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `hellinger`: Normalized Hellinger distance, numerical and categorical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `mannw`: Mann-Whitney U-rank test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `ed`: Energy distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `es`: Epps-Singleton tes, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `t_test`: T-Test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `emperical_mmd`: Emperical-MMD, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `TVD`: Total-Variation-Distance, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    A combination of these tests are used if `stattest` and other equivalent parameters are not explicitly specified.

    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    column: Column from both current and reference data to detect drift on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    stattest: Defines the drift detection method for a given column (if a single column is tested), or all columns in the dataset (if multiple columns are tested).
    stattest_threshold: Sets the drift threshold in a given column or all columns. The threshold meaning varies based on the drift detection method, e.g., it can be the value of a distance metric or a p-value of a statistical test.

    Returns
    ------------
    data_drift_report: Interactive visualization object containing the data drift report
    ddr: Report 
    """
    
    data_drift_report = Report(metrics = [
        ColumnDriftMetric(column, stattest = stattest, stattest_threshold = stattest_threshold),
        ColumnValuePlot(column),
    ])
    data_drift_report.run(current_data = current_data, reference_data = reference_data)
    
    if report_format == 'json':
        ddr = data_drift_report.json()
    elif report_format == 'dict':
        ddr = data_drift_report.as_dict()
    elif report_format == 'DataFrame':
        ddr = data_drift_report.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        ddr = None
        
    return data_drift_report, ddr

def data_drift_dataset_test(current_data, reference_data, test_format = 'json', column_mapping = None, stattest = None, stattest_threshold = None, 
                            cat_stattest = None, cat_stattest_threshold = None, num_stattest = None, num_stattest_threshold = None, per_column_stattest = None, 
                            per_column_stattest_threshold = None):
    """
    This compares the distribution of each column in the current dataset to the reference and tests the number and share of drifting features against a defined condition.
    
    Drift detection methods that can be used for `stattest` and other equivalent parameters are as follows:
    - `ks`: Kolmogorov-Smirnov test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `chisquare`: Chi-Square test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `z`: Z-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `wasserstein`: Normalized Wasserstein distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `kl_div`: Kullback-Leibler, numerical and categorical. Returns divergence drift detected when greater than or equal to threshold (default is 0.1).
    - `psi`: Population Stability Index, numerical and categorical. Returns psi_value drift detected when greater than or equal to threshold (default is 0.1).
    - `jensenshannon`: Jensen-Shannon distance, numerical and categorical. Default method for categorical, if there are > 1000 objects. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `anderson`: Anderson-Darling test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `fisher_exact`: Fisher's Exact test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `cramer_von_mises`: Cramer-Von-Mises test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `g-test`: G-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `hellinger`: Normalized Hellinger distance, numerical and categorical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `mannw`: Mann-Whitney U-rank test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `ed`: Energy distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `es`: Epps-Singleton tes, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `t_test`: T-Test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `emperical_mmd`: Emperical-MMD, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `TVD`: Total-Variation-Distance, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    A combination of these tests are used if `stattest` and other equivalent parameters are not explicitly specified.
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type.
    stattest: Defines the drift detection method for a given column (if a single column is tested), or all columns in the dataset (if multiple columns are tested).
    stattest_threshold: Sets the drift threshold in a given column or all columns. The threshold meaning varies based on the drift detection method, e.g., it can be the value of a distance metric or a p-value of a statistical test.
    cat_stattest: Sets the drift method for all categorical columns in the dataset.
    cat_stattest_threshold: Sets the threshold for all categorical columns in the dataset.
    num_stattest: Sets the drift method for all numerical columns in the dataset.
    num_stattest_threshold: Sets the threshold for all numerical columns in the dataset.
    per_column_stattest: Sets the drift method for the listed columns (accepts a dictionary).
    per_column_stattest_threshold: Sets the threshold for the listed columns (accepts a dictionary).
    
    Returns
    ------------
    data_drift_test: Interactive visualization object containing the data drift test
    ddt: Test results 
    """
    
    data_drift_test = TestSuite(tests = [
        TestNumberOfDriftedColumns(stattest = stattest, stattest_threshold = stattest_threshold,
                                   cat_stattest = cat_stattest, cat_stattest_threshold = cat_stattest_threshold,
                                   num_stattest = num_stattest, num_stattest_threshold = num_stattest_threshold,
                                   per_column_stattest = per_column_stattest, per_column_stattest_threshold = per_column_stattest_threshold),
        TestShareOfDriftedColumns(stattest = stattest, stattest_threshold = stattest_threshold,
                                  cat_stattest = cat_stattest, cat_stattest_threshold = cat_stattest_threshold,
                                  num_stattest = num_stattest, num_stattest_threshold = num_stattest_threshold,
                                  per_column_stattest = per_column_stattest, per_column_stattest_threshold = per_column_stattest_threshold),
        TestAllFeaturesValueDrift(stattest = stattest, stattest_threshold = stattest_threshold,
                                  cat_stattest = cat_stattest, cat_stattest_threshold = cat_stattest_threshold,
                                  num_stattest = num_stattest, num_stattest_threshold = num_stattest_threshold,
                                  per_column_stattest = per_column_stattest, per_column_stattest_threshold = per_column_stattest_threshold)
    ])
    
    data_drift_test.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if test_format == 'json':
        ddt = data_drift_test.json()
    elif test_format == 'dict':
        ddt = data_drift_test.as_dict()
    elif test_format == 'DataFrame':
        ddt = data_drift_test.as_pandas()
    else:
        print("Test format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        ddt = None
        
    return data_drift_test, ddt

def data_drift_column_test(current_data, reference_data, column, test_format = 'json', stattest=None, stattest_threshold=None):
    """
    This compares the distribution of an identified column in the current dataset to the reference and tests for data drift.
    
    Drift detection methods that can be used for `stattest` and other equivalent parameters are as follows:
    - `ks`: Kolmogorov-Smirnov test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `chisquare`: Chi-Square test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `z`: Z-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `wasserstein`: Normalized Wasserstein distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `kl_div`: Kullback-Leibler, numerical and categorical. Returns divergence drift detected when greater than or equal to threshold (default is 0.1).
    - `psi`: Population Stability Index, numerical and categorical. Returns psi_value drift detected when greater than or equal to threshold (default is 0.1).
    - `jensenshannon`: Jensen-Shannon distance, numerical and categorical. Default method for categorical, if there are > 1000 objects. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `anderson`: Anderson-Darling test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `fisher_exact`: Fisher's Exact test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `cramer_von_mises`: Cramer-Von-Mises test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `g-test`: G-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `hellinger`: Normalized Hellinger distance, numerical and categorical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `mannw`: Mann-Whitney U-rank test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `ed`: Energy distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `es`: Epps-Singleton tes, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `t_test`: T-Test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `emperical_mmd`: Emperical-MMD, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `TVD`: Total-Variation-Distance, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    A combination of these tests are used if `stattest` and other equivalent parameters are not explicitly specified.
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    column: Column from both current and reference data to detect drift on.
    test_format: Specify the format to output the test object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    
    Returns
    ------------
    data_drift_test: Interactive visualization object containing the data drift test
    ddt: Test results 
    """
    
    data_drift_test = TestSuite(tests = [
        TestColumnDrift(column_name = column, stattest = stattest, stattest_threshold = stattest_threshold)
    ])
    
    data_drift_test.run(current_data = current_data, reference_data = reference_data)
    
    if test_format == 'json':
        ddt = data_drift_test.json()
    elif test_format == 'dict':
        ddt = data_drift_test.as_dict()
    elif test_format == 'DataFrame':
        ddt = data_drift_test.as_pandas()
    else:
        print("Test format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        ddt = None
        
    return data_drift_test, ddt

def data_quality_dataset_report(current_data, reference_data, report_format = 'json', column_mapping = None,
                               adt = 0.95, act = 0.95):
    """
    Calculate various descriptive statistics, the number and share of missing values per column, and correlations between columns in the dataset.
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type.
    adt: Almost Duplicated Threshold, for when values look to be very similar to each other. Defaults to 0.95.
    act: Almost Constant Threshold, for when a column exhibits almost no variance. Defaults to 0.95.
    
    Returns
    ------------
    data_quality_report: Interactive visualization object containing the data quality report
    dqr: Report
    """
    
    data_quality_report = Report(metrics = [
        DatasetSummaryMetric(almost_duplicated_threshold = adt, almost_constant_threshold = act),
        DatasetMissingValuesMetric(),
        DatasetCorrelationsMetric()
    ])
    
    data_quality_report.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if report_format == 'json':
        dqr = data_quality_report.json()
    elif report_format == 'dict':
        dqr = data_quality_report.as_dict()
    elif report_format == 'DataFrame':
        dqr = data_quality_report.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        dqr = None
        
    return data_quality_report, dqr

def data_quality_column_report(current_data, reference_data, column, report_format = 'json', quantile = 0.75,  values_list = None):
    """
    For an identified column:
    - Calculates various descriptive statistics,
    - Calculates number and share of missing values
    - Plots distribution histogram,
    - Calculates quantile value and plots distribution,
    - Calculates correlation between defined column and all other columns
    - If categorical, calculates number of values in list / out of the list / not found in defined column
    - If numerical, calculates number and share of values in specified range / out of range in defined column, and plots distributions
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    column: Column from both current and reference data to detect drift on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    quantile: Float, from [0, 1] to showcase quantile value in distribution. Defaults to 0.75.
    values_list: List of values to showcase for either a range (numerical, e.g. [10, 20] for an `Age` column) or list (categorical, e.g. ['High School', 'Post-Graduate'] for an `Education` column). Defaults to None.
    
    Returns
    ------------
    data_quality_report: Interactive visualization object containing the data quality report
    dqr: Report
    """
    
    if values_list:
        if np.issubdtype(current_data[column], np.number): # If column data type is numerical
            data_quality_report = Report(metrics = [
                ColumnSummaryMetric(column_name = column),
                ColumnMissingValuesMetric(column_name = column),
                ColumnDistributionMetric(column_name = column),
                ColumnValuePlot(column_name = column),
                ColumnQuantileMetric(column_name = column, quantile = quantile),
                ColumnValueRangeMetric(column_name = column, values = values_list)
            ])
        else: # If column data type is object/categorical
            data_quality_report = Report(metrics = [
                ColumnSummaryMetric(column_name = column),
                ColumnMissingValuesMetric(column_name = column),
                ColumnDistributionMetric(column_name = column),
                ColumnValuePlot(column_name = column),
                ColumnQuantileMetric(column_name = column, quantile = quantile),
                ColumnValueListMetric(column_name = column, values = values_list)
            ])
    else:
        data_quality_report = Report(metrics = [
            ColumnSummaryMetric(column_name = column),
            ColumnMissingValuesMetric(column_name = column),
            ColumnDistributionMetric(column_name = column),
            ColumnValuePlot(column_name = column),
            ColumnQuantileMetric(column_name = column, quantile = quantile)
        ])
        
    data_quality_report.run(current_data = current_data, reference_data = reference_data)
    
    if report_format == 'json':
        dqr = data_quality_report.json()
    elif report_format == 'dict':
        dqr = data_quality_report.as_dict()
    elif report_format == 'DataFrame':
        dqr = data_quality_report.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        dqr = None
        
    return data_quality_report, dqr

def data_quality_dataset_test(current_data, reference_data, test_format = 'json', column_mapping = None):
    """
    For all columns in a dataset:
    - Tests number of rows and columns against reference or defined condition
    - Tests number and share of missing values in the dataset against reference or defined condition
    - Tests number and share of columns and rows with missing values against reference or defined condition
    - Tests number of differently encoded missing values in the dataset against reference or defined condition
    - Tests number of columns with all constant values against reference or defined condition
    - Tests number of empty rows (expects 10% or none) and columns (expects none) against reference or defined condition
    - Tests number of duplicated rows (expects 10% or none) and columns (expects none) against reference or defined condition
    - Tests types of all columns against the reference, expecting types to match
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    test_format: Specify the format to output the test object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type.
    
    Returns
    ------------
    data_quality_test: Interactive visualization object containing the data quality test
    ddt: Test results 
    """
    
    data_quality_test = TestSuite(tests = [
        TestNumberOfColumns(),
        TestNumberOfRows(),
        TestNumberOfMissingValues(),
        TestShareOfMissingValues(),
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestShareOfColumnsWithMissingValues(),
        TestShareOfRowsWithMissingValues(),
        TestNumberOfDifferentMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfEmptyRows(),
        TestNumberOfEmptyColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
    ])
    
    data_quality_test.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if test_format == 'json':
        dqt = data_quality_test.json()
    elif test_format == 'dict':
        dqt = data_quality_test.as_dict()
    elif test_format == 'DataFrame':
        dqt = data_quality_test.as_pandas()
    else:
        print("Test format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        dqt = None
        
    return data_quality_test, dqt

def data_quality_column_test(current_data, reference_data, column, test_format = 'json', n_sigmas = 2, quantile = 0.25):
    """
    For a given column in the dataset:
    - Tests number and share of missing values in a given column against reference or defined condition
    - Tests number of differently encoded missing values in a given column against reference or defined condition
    - Tests if all values in a given column are a) constant, b) unique
    - Tests the minimum and maximum value of a numerical column against reference or defined condition
    - Tests the mean, median, and standard deviation of a numerical column against reference or defined condition
    - Tests the number and share of unique values against reference or defined condition
    - Tests the most common value in a categorical column against reference or defined condition
    - Tests if the mean value in a numerical column is within expected range, defined in standard deviations
    - Tests if numerical column contains values out of min-max range, and its share against reference or defined condition
    - Tests if a categorical variable contains values out of the list, and its share against reference or defined condition
    - Computes a quantile value and compares to reference or defined condition
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    column: Column from both current and reference data to detect drift on.
    test_format: Specify the format to output the test object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    n_sigmas: # of sigmas (standard deviations) to test on mean value of a numerical column. Defaults to 2.
    quantile: Float, from [0, 1] to showcase quantile value in distribution. Defaults to 0.75.
    
    Returns
    ------------
    data_quality_test: Interactive visualization object containing the data quality test
    ddt: Test results 
    """
    
    if np.issubdtype(current_data[column], np.number): # If column data type is numerical:
        data_quality_test = TestSuite(tests = [
            TestColumnNumberOfMissingValues(column_name = column),
            TestColumnShareOfMissingValues(column_name = column),
            TestColumnNumberOfDifferentMissingValues(column_name = column),
            TestColumnAllConstantValues(column_name = column),
            TestColumnAllUniqueValues(column_name = column),
            TestColumnValueMin(column_name = column),
            TestColumnValueMax(column_name = column),
            TestColumnValueMean(column_name = column),
            TestColumnValueMedian(column_name = column),
            TestColumnValueStd(column_name = column),
            TestNumberOfUniqueValues(column_name = column),
            TestUniqueValuesShare(column_name = column),
            TestMeanInNSigmas(column_name = column, n_sigmas = n_sigmas),
            TestValueRange(column_name = column),
            TestNumberOfOutRangeValues(column_name = column),
            TestShareOfOutRangeValues(column_name = column),
            TestColumnQuantile(column_name = column, quantile = quantile),
        ])
    else: # If column data type is object/categorical
        data_quality_test = TestSuite(tests = [
            TestColumnNumberOfMissingValues(column_name = column),
            TestColumnShareOfMissingValues(column_name = column),
            TestColumnNumberOfDifferentMissingValues(column_name = column),
            TestColumnAllConstantValues(column_name = column),
            TestColumnAllUniqueValues(column_name = column),
            TestNumberOfUniqueValues(column_name = column),
            TestUniqueValuesShare(column_name = column),
            TestMostCommonValueShare(column_name = column),
            TestValueList(column_name = column),
            TestNumberOfOutListValues(column_name = column),
            TestShareOfOutListValues(column_name = column),
            TestColumnQuantile(column_name = column, quantile = quantile),
        ])
    
    data_quality_test.run(current_data = current_data, reference_data = reference_data)
    
    if test_format == 'json':
        dqt = data_quality_test.json()
    elif test_format == 'dict':
        dqt = data_quality_test.as_dict()
    elif test_format == 'DataFrame':
        dqt = data_quality_test.as_pandas()
    else:
        print("Test format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        dqt = None
        
    return data_quality_test, dqt

def target_drift_report(current_data, reference_data, report_format = 'json', column_mapping = None, stattest = None, stattest_threshold = None,
                        cat_stattest = None, cat_stattest_threshold = None, num_stattest = None, num_stattest_threshold = None,
                        per_column_stattest = None, per_column_stattest_threshold = None):
    """
    
    Drift detection methods that can be used for `stattest` and other equivalent parameters are as follows:
    - `ks`: Kolmogorov-Smirnov test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `chisquare`: Chi-Square test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `z`: Z-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `wasserstein`: Normalized Wasserstein distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `kl_div`: Kullback-Leibler, numerical and categorical. Returns divergence drift detected when greater than or equal to threshold (default is 0.1).
    - `psi`: Population Stability Index, numerical and categorical. Returns psi_value drift detected when greater than or equal to threshold (default is 0.1).
    - `jensenshannon`: Jensen-Shannon distance, numerical and categorical. Default method for categorical, if there are > 1000 objects. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `anderson`: Anderson-Darling test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `fisher_exact`: Fisher's Exact test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `cramer_von_mises`: Cramer-Von-Mises test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `g-test`: G-test, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `hellinger`: Normalized Hellinger distance, numerical and categorical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `mannw`: Mann-Whitney U-rank test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `ed`: Energy distance, only numerical. Returns distance drift detected when greater than or equal to threshold (default is 0.1).
    - `es`: Epps-Singleton tes, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `t_test`: T-Test, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `emperical_mmd`: Emperical-MMD, only numerical. Returns p_value drift detected when less than threshold (default is 0.05).
    - `TVD`: Total-Variation-Distance, only categorical. Returns p_value drift detected when less than threshold (default is 0.05).
    A combination of these tests are used if `stattest` and other equivalent parameters are not explicitly specified.
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type.
    stattest: Defines the drift detection method for a given column (if a single column is tested), or all columns in the dataset (if multiple columns are tested).
    stattest_threshold: Sets the drift threshold in a given column or all columns. The threshold meaning varies based on the drift detection method, e.g., it can be the value of a distance metric or a p-value of a statistical test.
    cat_stattest: Sets the drift method for all categorical columns in the dataset.
    cat_stattest_threshold: Sets the threshold for all categorical columns in the dataset.
    num_stattest: Sets the drift method for all numerical columns in the dataset.
    num_stattest_threshold: Sets the threshold for all numerical columns in the dataset.
    per_column_stattest: Sets the drift method for the listed columns (accepts a dictionary).
    per_column_stattest_threshold: Sets the threshold for the listed columns (accepts a dictionary).
    
    Returns
    ------------
    target_drift_report: Interactive visualization object containing the target drift report
    tdr: Report 
    """
    
    td_report = Report(metrics = [
        TargetDriftPreset(stattest = stattest, stattest_threshold = stattest_threshold,
                          cat_stattest = cat_stattest, cat_stattest_threshold = cat_stattest_threshold,
                          num_stattest = num_stattest, num_stattest_threshold = num_stattest_threshold,
                          per_column_stattest = per_column_stattest, per_column_stattest_threshold = per_column_stattest_threshold)
    ])
       
    td_report.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if report_format == 'json':
        tdr = td_report.json()
    elif report_format == 'dict':
        tdr = td_report.as_dict()
    elif report_format == 'DataFrame':
        tdr = td_report.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        tdr = None
        
    return td_report, tdr

def regression_performance_report(current_data, reference_data, report_format = 'json', column_mapping = None,
                                 top_error = 0.05, columns = None):
    """
    For a regression model, the report shows the following:
    - Calculates various regression performance metrics, including Mean Error, MAPE, MAE, etc.
    - Visualizes predicted vs. actual values in a scatter plot and line plot
    - Visualizes the model error (predicted - actual) and absolute percentage error in respective line plots
    - Visualizes the model error distribution in a histogram
    - Visualizes the quantile-quantile (Q-Q) plot to estimate value normality
    - Calculates and visualizes the regression performance metrics for different groups -- top-X% with overestimation, top-X% with underestimation
    - Plots relationship between feature values and model quality per group (for top-X% error groups)
    - Calculates the number of instances where model returns a different output for an identical input
    - Calculates the number of instances where there is a different target value/label for an identical input
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type. Defaults to None.
    top_error: Threshold for creating groups of instances with top percentiles in a) overestimation and b) underestimation. Defaults to 0.05.
    columns: List of columns to showcase in the error bias table. Defaults to None, which showcases all columns.
    
    Returns
    ------------
    regression_report: Interactive visualization object containing the regression report
    rpr: Report 
    """
    
    regression_report = Report(metrics = [
        RegressionQualityMetric(),
        RegressionPredictedVsActualScatter(),
        RegressionPredictedVsActualPlot(),
        RegressionErrorPlot(),
        RegressionAbsPercentageErrorPlot(),
        RegressionErrorDistribution(),
        RegressionErrorNormality(),
        RegressionTopErrorMetric(top_error = top_error),
        RegressionErrorBiasTable(columns = columns, top_error = top_error),
        ConflictTargetMetric(),
        ConflictPredictionMetric()
    ])
    
    regression_report.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if report_format == 'json':
        rpr = regression_report.json()
    elif report_format == 'dict':
        rpr = regression_report.as_dict()
    elif report_format == 'DataFrame':
        rpr = regression_report.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        rpr = None
        
    return regression_report, rpr

def regression_performance_test(current_data, reference_data, test_report = 'json', column_mapping = None, approx_val = None, rel_val = 0.1):
    """
    Computes the following tests on regression data, failing if +/- a percentage (%) of scores over reference data is achieved:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Mean Error (ME) and tests if it is near zero
    - Mean Absolute Percentage Error (MAPE)
    - Absolute Maximum Error
    - R2 Score (coefficient of determination)
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type. Defaults to None.
    approx_val: Dictionary, if user wants to specify values for each test. Defaults to None. See documentation for example on how to put parameters.
    rel_val: Relative percentage with which each test will pass or fail. Defaults to 0.1 (10%).
    
    Returns
    ------------
    regression_test: Interactive visualization object containing the regression test
    rpt: Test 
    """
    
    if approx_val:
        # # SAMPLE DICTIONARY
        # approx_val = {
        #     'mae': 30,
        #     'rmse': 4.5,
        #     'me': 15,
        #     'mape': 0.2,
        #     'ame': 50,
        #     'r2': 0.75
        # }
        
        regression_test = TestSuite(tests = [
            TestValueMAE(eq = approx(approx_val['mae'], relative = rel_val)),
            TestValueRMSE(eq = approx(approx_val['rmse'], relative = rel_val)),
            TestValueMeanError(eq = approx(approx_val['me'], relative = rel_val)),
            TestValueMAPE(eq = approx(approx_val['mape'], relative = rel_val)),
            TestValueAbsMaxError(eq = approx(approx_val['ame'], relative = rel_val)),
            TestValueR2Score(eq = approx(approx_val['r2'], relative = rel_val))
        ])
    else:
        regression_test = TestSuite(tests = [
            TestValueMAE(eq = rel_val),
            TestValueRMSE(eq = rel_val),
            TestValueMeanError(eq = rel_val),
            TestValueMAPE(eq = rel_val),
            TestValueAbsMaxError(eq = rel_val),
            TestValueR2Score(eq = rel_val)
        ])
    
    regression_test.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if test_report == 'json':
        rpt = regression_test.json()
    elif test_report == 'dict':
        rpt = regression_test.as_dict()
    elif test_report == 'DataFrame':
        rpt = regression_test.as_pandas()
    else:
        print("Test format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        rpt = None
        
    return regression_test, rpt

def classification_performance_report(current_data, reference_data, is_prob = False, report_format = 'json', column_mapping = None,
                                     probas_threshold = None, columns = None):
    """
    For a classification model, the report shows the following:
    - Calculates various classification performance metrics, such as precision, accuracy, recall, F1-score, TPR, TNR, FPR, FNR, AUROC, LogLoss
    - Calculates the number of objects for each label and plots a histogram
    - Calculates the TPR, TNR, FPR, FNR, and plots the confusion matrix
    - Calculates the classification quality metrics for each class and plots a matrix
    - For probabilistic classification, visualizes the predicted probabilities by class
    - For probabilistic classification, visualizes the probability distribution by class
    - For probabilistic classification, plots the ROC Curve
    - For probabilistic classification, plots the Precision-Recall curve
    - Calculates the Precision-Recall table that shows model quality at a different decision threshold
    - Visualizes the relationship between feature values and model quality
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    is_prob: Boolean, determines if target class is absolute or probabilistic. Defaults to False.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type. Defaults to None.
    probas_threshold: Threshold at which to determine a positive value for a class. Defaults to None. Can be set to 0.5 for probabilistic classification.
    columns: List of columns to showcase in the error bias table. Defaults to None, which showcases all columns.
    
    Returns
    ------------
    classification_report: Interactive visualization object containing the classification report
    cpr: Report 
    """
    
    if is_prob:
        classification_report = Report(metrics = [
            ClassificationQualityMetric(probas_threshold = probas_threshold),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(probas_threshold = probas_threshold),
            ClassificationQualityByClass(probas_threshold = probas_threshold),
            ClassificationClassSeparationPlot(),
            ClassificationProbDistribution(),
            ClassificationRocCurve(),
            ClassificationPRCurve(),
            ClassificationPRTable(),
            ClassificationQualityByFeatureTable(columns = columns)
        ])
    else:
        classification_report = Report(metrics = [
            ClassificationQualityMetric(probas_threshold = probas_threshold),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(probas_threshold = probas_threshold),
            ClassificationQualityByClass(probas_threshold = probas_threshold),
            #ClassificationPRTable(),
            ClassificationQualityByFeatureTable(columns = columns)
        ])
    
    classification_report.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if report_format == 'json':
        cpr = classification_report.json()
    elif report_format == 'dict':
        cpr = classification_report.as_dict()
    elif report_format == 'DataFrame':
        cpr = classification_report.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        cpr = None
        
    return classification_report, cpr

def classification_performance_test(current_data, reference_data, is_prob = False, test_format = 'json', column_mapping = None,
                                    probas_threshold = None, pred_col = 'prediction', approx_val = None, rel_val = 0.2):
    """
    Computes the following tests on classification data, failing if +/- a percentage (%) of scores over reference data is achieved:
    - Accuracy, Precision, Recall, F1 on the whole dataset
    - Precision, Recall, F1 on each class
    - Computes the True Positive Rate (TPR), True Negative Rate (TNR), False Positive Rate (FPR), False Negative Rate (FNR)
    - For probabilistic classification, computes the ROC AUC and LogLoss
    
    Parameters
    ------------
    current_data: DataFrame, inference data collected after model deployment or training.
    reference_data: DataFrame, dataset used to train your initial model on.
    is_prob: Boolean, determines if target class is absolute or probabilistic. Defaults to False.
    report_format: Specify the format to output the report object in, from `json`, `dict`, and `DataFrame`. Defaults to `json`.
    column_mapping: Input a column mapping object so the function knows how to treat each column based on data type. Defaults to None.
    probas_threshold: Threshold at which to determine a positive value for a class. Defaults to None. Can be set to 0.5 for probabilistic classification.
    pred_col: Column name of prediction column in current data. Defaults to `prediction`.
    approx_val: Dictionary, if user wants to specify values for each test. Defaults to None. See documentation for example on how to put parameters.
    rel_val: Relative percentage with which each test will pass or fail. Defaults to 0.2 (20%).
    
    Returns
    ------------
    classification_test: Interactive visualization object containing the classification test
    cpt: Test 
    """
    
    if approx_val:
        # # SAMPLE DICTIONARY
        # approx_val = {
        #     'mae': 30,
        #     'rmse': 4.5,
        #     'me': 15,
        #     'mape': 0.2,
        #     'ame': 50,
        #     'r2': 0.75
        # }
        if is_prob:
            argmax = [np.argmax(test_proba[n]) for n in range(0, len(test_proba))]
            labels = argmax.unique()

            tests = [TestRocAuc(eq = approx(approx_val['mae'], relative = rel_val)),
                     TestLogLoss(eq = approx(approx_val['mae'], relative = rel_val))] # Tests applicable for probability classification outputs
        else:
            labels = current_data[pred_col].unique()
            tests = []

        for label in labels:
            tp = TestPrecisionByClass(label = label, probas_threshold = probas_threshold,
                                     eq = approx(approx_val['mae'], relative = rel_val))
            tr = TestRecallByClass(label = label, probas_threshold = probas_threshold,
                                  eq = approx(approx_val['mae'], relative = rel_val))
            tf = TestF1ByClass(label = label, probas_threshold = probas_threshold,
                              eq = approx(approx_val['mae'], relative = rel_val))
            tests.extend([tp, tr, tf])

        tests = [
            TestAccuracyScore(eq = approx(approx_val['mae'], relative = rel_val)),
            TestPrecisionScore(eq = approx(approx_val['mae'], relative = rel_val)),
            TestRecallScore(eq = approx(approx_val['mae'], relative = rel_val)),
            TestF1Score(eq = approx(approx_val['mae'], relative = rel_val)),
            TestTPR(eq = approx(approx_val['mae'], relative = rel_val)),
            TestTNR(eq = approx(approx_val['mae'], relative = rel_val)),
            TestFPR(eq = approx(approx_val['mae'], relative = rel_val)),
            TestFNR(eq = approx(approx_val['mae'], relative = rel_val)),
        ] + tests
    else:
    
        if is_prob:
            argmax = [np.argmax(test_proba[n]) for n in range(0, len(test_proba))]
            labels = argmax.unique()

            tests = [TestRocAuc(eq = rel_val), TestLogLoss(eq = rel_val)] # Tests applicable for probability classification outputs
        else:
            labels = current_data[pred_col].unique()
            tests = []

        for label in labels:
            tp = TestPrecisionByClass(label = label, probas_threshold = probas_threshold, eq = rel_val)
            tr = TestRecallByClass(label = label, probas_threshold = probas_threshold, eq = rel_val)
            tf = TestF1ByClass(label = label, probas_threshold = probas_threshold, eq = rel_val)
            tests.extend([tp, tr, tf])

        tests = [
            TestAccuracyScore(eq = rel_val),
            TestPrecisionScore(eq = rel_val),
            TestRecallScore(eq = rel_val),
            TestF1Score(eq = rel_val),
            TestTPR(eq = rel_val),
            TestTNR(eq = rel_val),
            TestFPR(eq = rel_val),
            TestFNR(eq = rel_val),
        ] + tests

    classification_test = TestSuite(tests = tests)
    
    classification_test.run(current_data = current_data, reference_data = reference_data, column_mapping = column_mapping)
    
    if test_format == 'json':
        cpt = classification_test.json()
    elif test_format == 'dict':
        cpt = classification_test.as_dict()
    elif test_format == 'DataFrame':
        cpt = classification_test.as_pandas()
    else:
        print("Report format not correct! Please pick between 'json', 'dict', and 'DataFrame'.")
        cpt = None
        
    return classification_test, cpt


###### ALIBI DETECT ######


def cramer_von_mises(loss_ref, losses, p_val = 0.05, labels = ['No!', 'Yes!']):
    """
    Cramer-von Mises (CVM) data drift detector, which tests for any change in the distribution of continuous univariate data. Works for both regression and classification use cases.
    For multivariate data, a separate CVM test is applied to each feature, and the obtained p-values are aggregated via the Bonferroni or False Discovery Rate (FDR) corrections.
    
    Parameters
    ------------
    loss_ref: Loss function from your reference data
    losses: Dictionary, with key-value pair of `Dataset` and `loss function for that dataset`. Examples of what could be tested are {`Concept`: loss_concept, `Covariance`: loss_covariance, `6MonthsPast`: loss_6months}
    p_val: Threshold to determine whether data has drifted. Defaults to 0.05.
    labels: List, labels to detect drift. Defaults to ['No', 'Yes'].
    
    Returns
    ------------
    cvm_dict: Dictionary of datasets and p-values based on the Cramer-von-Mises detector
    """
    
    print('Cramer-von-Mises')
    print('--------------------')
    cd = CVMDrift(loss_ref.to_numpy(), p_val = p_val)
    
    cvm_dict = {}
    
    for name, loss_arr in losses.items():
        print('\n%s' % name)
        preds = cd.predict(loss_arr)
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('p-value: {}'.format(preds['data']['p_val'][0]))
        
        cvm_dict[name] = preds['data']['p_val'][0]
    
    return cvm_dict

def maximum_mean_discrepancy(X_ref, Xs, preprocessor = None, p_val = 0.05, labels = ['No!', 'Yes!']):
    """
    Maximum Mean Discrepancy (MMD) data drift detector using a permutation test. Usually used for unsupervised, non-malicious drift detection.
    Works for regression, classification, and unsupervised use cases.
    
    Parameters
    ------------
    X_ref: DataFrame, containing the reference data used for model training.
    Xs: Dictionary, with key-value pair of `Dataset Name` and `DataFrame`. Examples of what could be tested are {`Concept`: df_concept, `Covariance`: df_covariance, `6MonthsPast`: df_6months}
    preprocessor: preprocessor object, if used to preprocessed data before model training. Defaults to None.
    p_val: Threshold to determine whether data has drifted. Defaults to 0.05.
    labels: List, labels to detect drift. Defaults to ['No', 'Yes'].
    
    Returns
    ------------
    mmd_dict: Dictionary of datasets and p-values based on the Maximum Mean Discrepancy detector
    """
    
    print("Maximum Mean Discrepancy")
    print('--------------------')
    
    if preprocessor:
        cd = MMDDrift(preprocessor.transform(X_ref), p_val = p_val)
        Xs = {name: preprocessor.transform(X) for name, X in Xs.items()}
    else:
        cd = MMDDrift(X_ref, p_val = p_val)
    
    mmd_dict = {}
    
    for name, Xarr in Xs.items():
        print('\n%s' % name)
        preds = cd.predict(Xarr.to_numpy())
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('p-value: {}'.format(preds['data']['p_val'][0]))
        
        mmd_dict[name] = preds['data']['p_val'][0]
    
    return mmd_dict

def fishers_exact_test(loss_ref, losses, p_val = 0.05, labels = ['No!', 'Yes!']):
    """
    Fisher exact test (FET) data drift detector, which tests for a change in the mean of binary univariate data. Works for classification use cases only.
    For multivariate data, a separate FET test is applied to each feature, and the obtained p-values are aggregated via the Bonferroni or False Discovery Rate (FDR) corrections.
    
    Parameters
    ------------
    loss_ref: Loss function from your reference data
    losses: Dictionary, with key-value pair of `Dataset` and `loss function for that dataset`. Examples of what could be tested are {`Concept`: loss_concept, `Covariance`: loss_covariance, `6MonthsPast`: loss_6months}
    p_val: Threshold to determine whether data has drifted. Defaults to 0.05.
    labels: List, labels to detect drift. Defaults to ['No', 'Yes'].
    
    Returns
    ------------
    fet_dict: Dictionary of datasets and p-values based on the Fisher's Exact Test detector
    """
    
    print("Fisher's Exact Test")
    print('--------------------')
    cd = FETDrift(loss_ref.to_numpy(), p_val = p_val)
    
    fet_dict = {}
    
    for name, loss_arr in losses.items():
        print('\n%s' % name)
        preds = cd.predict(loss_arr.to_numpy())
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('p-value: {}'.format(preds['data']['p_val'][0]))
        
        fet_dict[name] = preds['data']['p_val'][0]
    
    return fet_dict

def categs(X, infer = False):
    """
    Generate a category map, and the categories on each categorical feature.
    
    Parameters
    ------------
    X: DataFrame of variables used to train model
    infer: Boolean, if you want the TabularDrift detector to infer the number of categories from reference data. Defaults to False.
    """
    
    df_cat_map = gen_category_map(X)
    if infer:
        df_cat_map = {f: None for f in list(df_cat_map.keys())}
    
    return df_cat_map

def tabular_drift(X_reference, X_current, preprocessor = None, p_val = 0.05, categories_per_feature = None,
                  drift_type = 'batch', labels = ['No!', 'Yes!'], feature_names = None):
    """
    Mixed-type tabular data drift detector with Bonferroni or False Discovery Rate (FDR) correction for multivariate data.
    Kolmogorov-Smirnov (K-S) univariate tests are applied to continuous numerical data and Chi-Squared (Chi2) univariate tests to categorical data.
    
    Parameters
    ------------
    X_reference: DataFrame, reference data used in initial model training.
    X_current: DataFrame, current data being for comparison.
    reprocessor: preprocessor object, if used to preprocessed data before model training. Defaults to None.
    p_val: Threshold to determine whether data has drifted. Defaults to 0.05.
    categories_per_feature: Dictionary, categories per categorical feature. Defaults to None.
    drift_type: String, either `batch` or `feature`. Defaults to `batch`.
    labels: List, labels to detect drift. Defaults to ['No', 'Yes'].
    feature_names: List of feature names of X_reference. Defaults to None.
    
    Returns
    ------------
    mtt: DataFrame, summarizes Chi-square or K-S statistics and drift detections for each variable. 
    """
    
    print("Mixed-Type Tabular Drift")
    print('--------------------')
    
    if preprocessor:
        X_reference = preprocessor.transform(X_reference)
        X_current = preprocessor.transform(X_current)
        features = preprocessor.get_feature_names_out()
        X_reference = pd.DataFrame(X_reference, columns=features)
        X_current = pd.DataFrame(X_current, columns=features)
        
    if not categories_per_feature:
        categories_per_feature = categs(X_reference, infer = True)
    
    # Initialize detector    
    cd = TabularDrift(X_reference.to_numpy(), p_val = p_val, categories_per_feature = categories_per_feature)

    preds = cd.predict(X_current.to_numpy(), drift_type = drift_type)
    labels = ['No!', 'Yes!']
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print('p-value: {}'.format(preds['data']['p_val'][0]))
    
    # Extract feature names if needed
    if not feature_names:
        if preprocessor:
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = X_reference.columns.tolist()
            
    fnames = []
    stats = []
    stat_vals = []
    p_vals = []
    
    if drift_type == 'batch':
        print('For Batch drift type:\n')
        
        # Check which of the feature-level p-values are below threshold
        print('Threshold: ', preds['data']['threshold'])
        # The preds dictionary also returns the K-S test statistics and p-value for each feature:
        for f in range(cd.n_features):
            stat = 'Chi2' if f in list(categories_per_feature.keys()) else 'K-S'
            fname = feature_names[f]
            stat_val, p_val = preds['data']['distance'][f], preds['data']['p_val'][f]
            print(f'{fname} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')
            
            fnames.append(fname)
            stats.append(stat)
            stat_vals.append(stat_val)
            p_vals.append(p_val)
        
        mtt = pd.DataFrame({
            'feature_name': fnames,
            'statistics': stats,
            'stat_value': stat_vals,
            'p_value': p_vals
        })
            
    elif drift_type == 'feature':
        print('For Feature drift type:\n')
        
        drifts = []
        
        for f in range(cd.n_features):
            stat = 'Chi2' if f in list(categories_per_feature.keys()) else 'K-S'
            fname = feature_names[f]
            is_drift = fpreds['data']['is_drift'][f]
            stat_val, p_val = fpreds['data']['distance'][f], fpreds['data']['p_val'][f]
            print(f'{fname} -- Drift? {labels[is_drift]} -- {stat} {stat_val:.3f} -- p-value {p_val:.3f}')
            
            fnames.append(fname)
            stats.append(stat)
            drifts.append(labels[is_drift])
            stat_vals.append(stat_val)
            p_vals.append(p_val)
        
        mtt = pd.DataFrame({
            'feature_name': fnames,
            'drift_detected': drifts,
            'statistics': stats,
            'stat_value': stat_vals,
            'p_value': p_vals
        })
            
    else:
        print('Please specify a valid drift type: either `batch` or `feature`. ')
        return _
        
    return mtt

def chi_sq(X_reference, X_current, p_val = 0.05, labels = ['No!', 'Yes!']):
    """
    For categorical variables, Chi-Squared data drift detector with Bonferroni or False Discovery Rate (FDR) correction for multivariate data.
    
    Parameters
    ------------
    X_reference: DataFrame, reference data used in initial model training.
    X_current: DataFrame, current data being for comparison.
    p_val: Threshold to determine whether data has drifted. Defaults to 0.05.
    
    Returns
    ------------
    cs: DataFrame, summarizes Chi-square statistics and drift detections for each categorical variable.
    """
    
    cd = ChiSquareDrift(X_reference.select_dtypes(exclude = np.number).to_numpy(), p_val = p_val)
    preds = cd.predict(X_current.select_dtypes(exclude = np.number).to_numpy())
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    
    cat_names = X_reference.select_dtypes(exclude = np.number).columns.tolist()
    
    print(f"Threshold: {preds['data']['threshold']}")
    
    fnames = []
    drifts = []
    stats = []
    stat_vals = []
    p_vals = []
    
    for f in range(cd.n_features):
        fname = cat_names[f]
        is_drift = (preds['data']['p_val'][f] < preds['data']['threshold']).astype(int)
        stat_val, p_val = preds['data']['distance'][f], preds['data']['p_val'][f]
        print(f'{fname} -- Drift? {labels[is_drift]} -- Chi2 {stat_val:.3f} -- p-value {p_val:.3f}')
        
        fnames.append(fname)
        drifts.append(labels[is_drift])
        stats.append('Chi2')
        stat_vals.append(stat_val)
        p_vals.append(p_val)

    cs = pd.DataFrame({
        'feature_name': fnames,
        'drift_detected': drifts,
        'statistics': stats,
        'stat_value': stat_vals,
        'p_value': p_vals
    })
    
    return cs