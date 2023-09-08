import pandas as pd
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go
from aif360.metrics import ClassificationMetric
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.sklearn.metrics.metrics import class_imbalance

def compute_metrics(dataset_true, dataset_pred, unprivileged_groups, privileged_groups, disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average absolute odds difference"] = classified_metric_pred.average_abs_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    #metrics["Smoothed EDF"] = classified_metric_pred.smoothed_empirical_differential_fairness()
    metrics["Class Imbalance"] = class_imbalance(pd.DataFrame(dataset_true.labels, index=dataset_true.protected_attributes.ravel()), pd.DataFrame(dataset_pred.labels, index=dataset_pred.protected_attributes.ravel()))
    
    if disp:
        with pd.option_context('display.max_rows', None,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
            print(pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Score']))
        #display(pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Score']))
#         for k in metrics:
#             print("%s = %.4f" % (k, metrics[k]))
    
    return metrics

def metrics_plot(metrics1, threshold, metric_name, protected, metrics2=None):
    """
    Plot the score of a specific fairness metric in a horizontal bar plot. Green region signifies the acceptable values 
    where the model is fair, while the red region signifies that model is not fair. The area of these region is set on 
    based on the threshold value. If a second set of metrics is inputted, the plot will be a grouped horizontal bar plot.
    
    Args:
    metrics1 (OrderedDict): contains the fairness scores of the model
    threshold (float): how far from the ideal value is the acceptable range for the fairness metric
    metric_name (str): the fairness metric to analyze, must be a key from metrics1
    protected (str): the protected group to analyze fairness with
    metrics2 (OrderedDict, optional): a second set of metric scores for comparison. Default is None.
    """
    if metrics2 == None:
        upper_bound = metrics1[metric_name] + 0.5
        upper_bound = upper_bound if upper_bound < 1.2 else 1.3
        lower_bound = metrics1[metric_name] - 0.5
        lower_bound = lower_bound if lower_bound < -0.2 else -0.3
        fig = px.bar(x=[metrics1[metric_name]],y=[""], labels={'x':'Score','y':''}, title=metric_name, color_discrete_sequence=px.colors.qualitative.D3)
        fig['layout'].update(height=250)
        if metric_name == 'Disparate impact' or metric_name == 'Balanced accuracy':
            line = {'type': 'line', 'x0': 0, 'x1': 0, 'y0': -0.5, 'y1': 0.5, 'line': {'color': "#371ea3", 'width': 1.5}}
            left_red = {'type': "rect", 'x0': lower_bound, 'y0': -0.5, 'x1': 1-threshold, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
            middle_green = {'type': "rect", 'x0': 1-threshold, 'y0': -0.5, 'x1': 1+threshold if metric_name=='Disparate impact' else 1, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#008000', 'layer': 'below', 'opacity': 0.1}
            right_red = {'type': "rect", 'x0': 1+threshold if upper_bound > 1+threshold else 1, 'y0': -0.5, 'x1': upper_bound, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
        else:
            line = {'type': 'line', 'x0': 0, 'x1': 0, 'y0': -0.5, 'y1': 0.5, 'line': {'color': "#371ea3", 'width': 1.5}}
            left_red = {'type': "rect", 'x0': lower_bound, 'y0': -0.5, 'x1': -threshold, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
            middle_green = {'type': "rect", 'x0': -threshold, 'y0': -0.5, 'x1': threshold, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#008000', 'layer': 'below', 'opacity': 0.1}
            right_red = {'type': "rect", 'x0': threshold, 'y0': -0.5, 'x1': upper_bound, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
        fig = fig.add_shape(line)
        fig = fig.add_shape(middle_green)
        fig = fig.add_shape(left_red)
        fig = fig.add_shape(right_red)
        return fig
    else:
        upper_bound = max([metrics1[metric_name],metrics2[metric_name]]) + 0.5
        upper_bound = upper_bound if upper_bound < 1.2 else 1.3
        lower_bound = min([metrics1[metric_name],metrics2[metric_name]]) - 0.5
        lower_bound = lower_bound if lower_bound < -0.2 else -0.3
        fig = px.bar(x=[metrics1[metric_name], metrics2[metric_name]],y=["", ""], color=["Before", "After"], barmode='group', labels={'x':'Score','y':''}, title=metric_name, color_discrete_sequence=px.colors.qualitative.D3)
        fig['layout'].update(height=250)
        if metric_name == 'Disparate impact' or metric_name == 'Balanced accuracy':
            line = {'type': 'line', 'x0': 0, 'x1': 0, 'y0': -0.5, 'y1': 0.5, 'line': {'color': "#371ea3", 'width': 1.5}}
            left_red = {'type': "rect", 'x0': lower_bound, 'y0': -0.5, 'x1': 1-threshold, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
            middle_green = {'type': "rect", 'x0': 1-threshold, 'y0': -0.5, 'x1': 1+threshold if metric_name=='Disparate impact' else 1, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#008000', 'layer': 'below', 'opacity': 0.1}
            right_red = {'type': "rect", 'x0': 1+threshold if upper_bound > 1+threshold else 1, 'y0': -0.5, 'x1': upper_bound, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
        else:
            line = {'type': 'line', 'x0': 0, 'x1': 0, 'y0': -0.5, 'y1': 0.5, 'line': {'color': "#371ea3", 'width': 1.5}}
            left_red = {'type': "rect", 'x0': lower_bound, 'y0': -0.5, 'x1': -threshold, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
            middle_green = {'type': "rect", 'x0': -threshold, 'y0': -0.5, 'x1': threshold, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#008000', 'layer': 'below', 'opacity': 0.1}
            right_red = {'type': "rect", 'x0': threshold, 'y0': -0.5, 'x1': upper_bound, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
        fig = fig.add_shape(line)
        fig = fig.add_shape(middle_green)
        fig = fig.add_shape(left_red)
        fig = fig.add_shape(right_red)
        return fig

def disparate_impact_remover(model, train_data, test_data, target_feature, protected, privileged_classes, favorable_classes=[1.0], repair_level=1.0):
    """
    Executes the AI Fairness 360's Disparate Impact Remover algorithm. Disparate impact remover is a preprocessing 
    technique that edits feature values to increase group fairness while preserving rank-ordering within groups. 
    The algorithm corrects for imbalanced selection rates between unprivileged andprivileged groups at various levels 
    of repair. 
    
    Args:
        model: the classifier model object
        train_data (pd.DataFrame): the train split from the dataset, must be preprocessed beforehand
        test_data (pd.DataFrame): the test split from the dataset, must be preprocessed beforehand
        target_feature (str): the target feature
        protected (list): list of protected features
        privileged_classes (list(list)): a 2d list containing the privileged classes for each protected feature
        favorable_classes (list): a list of favorable classes in the target feature, restricted to n-1 length, where n is 
        the number of classes
        repair_level (int): how much the DI remover will change the input data to get DI close to 1
        
    Returns:
        X_tr (np.array): Transformed X_train
        X_te (np.array): Transformed X_test
    """
    model_copy = deepcopy(model)
    
    sd_train = StandardDataset(train_data,target_feature,favorable_classes,protected,privileged_classes)
    sd_test = StandardDataset(test_data,target_feature,favorable_classes,protected,privileged_classes)
    p = []
    u = []
    for i, j in zip(protected,privileged_classes):
        p.append({i: j})
        u.append({i: [x for x in train_data[i].unique().tolist() if x not in j and not(np.isnan(x))]})
    
    print("Metrics Before DI Remover: ")
    sd_test_pred = sd_test.copy()
    sd_test_pred.labels = model.predict(sd_test.features)
    before = compute_metrics(sd_test, sd_test_pred, u, p)
    
    di = DisparateImpactRemover(repair_level=repair_level)
    train_repd = di.fit_transform(sd_train)
    test_repd = di.fit_transform(sd_test)
    X_tr = train_repd.features
    X_te = test_repd.features
    y_tr = train_repd.labels.ravel()
    model_copy.fit(X_tr, y_tr)
    test_repd_pred = test_repd.copy()
    test_repd_pred.labels = model_copy.predict(X_te)
    print("\nMetrics After DI Remover: ")
    after = compute_metrics(test_repd, test_repd_pred, u, p)
    
    for i in after.keys():
        metrics_plot(metrics1=before,metrics2=after,threshold=0.2,metric_name=i,protected=protected[0]).show()
    
    return X_tr, X_te, before, after
    
def reweighing(model, train_data, test_data, target_feature, protected, privileged_classes, favorable_classes=[1.0]):
    """
    Executes use of AI Fairness 360's Reweighing algorithm. Reweighing is a preprocessing technique that Weights 
    the examples in each (group, label) combination differently to ensure fairness before classification.
    
    Args:
        model: the classifier model object
        train_data (pd.DataFrame): the train split from the dataset, must be preprocessed beforehand
        test_data (pd.DataFrame): the test split from the dataset, must be preprocessed beforehand
        target_feature (str): the target feature
        protected (list): list of protected features
        privileged_classes (list(list)): a 2d list containing the privileged classes for each protected feature
        favorable_classes (list): a list of favorable classes in the target feature, restricted to n-1 length, where n is the number of classes
        
    Returns:
        X_tr (np.array): Transformed X_train
    """
    model_copy = deepcopy(model)
    
    sd_train = StandardDataset(train_data,target_feature,favorable_classes,protected,privileged_classes)
    sd_test = StandardDataset(test_data,target_feature,favorable_classes,protected,privileged_classes)
    p = []
    u = []
    for i, j in zip(protected,privileged_classes):
        p.append({i: j})
        u.append({i: [x for x in train_data[i].unique().tolist() if x not in j and not(np.isnan(x))]})
    
    print("Metrics Before Reweighing: ")
    sd_test_pred = sd_test.copy()
    sd_test_pred.labels = model.predict(sd_test.features)
    before = compute_metrics(sd_test, sd_test_pred, u, p)
    
    RW = Reweighing(unprivileged_groups=u,privileged_groups=p)
    RW.fit(sd_train)
    dataset_transf_train = RW.transform(sd_train)
    X_tr = dataset_transf_train.features
    y_tr = dataset_transf_train.labels.ravel()
    model.fit(X_tr,y_tr)
    X_te = sd_test.features
    sd_test_pred.labels = model.predict(X_te)
    print("\nMetrics After Reweighing: ")
    after = compute_metrics(sd_test, sd_test_pred, u, p)
    
    for i in after.keys():
        metrics_plot(metrics1=before,metrics2=after,threshold=0.2,metric_name=i,protected=protected[0]).show()
    
    return X_tr, before, after

def exponentiated_gradient_reduction(model, train_data, test_data, target_feature, protected, privileged_classes, favorable_classes=[1.0]):
    """
    Executes use of AI Fairness 360's Exponentiated Gradient Reduction algorithm. Exponentiated gradient reduction is 
    an in-processing technique that reduces fair classification to a sequence of cost-sensitive classification problems, 
    returning a randomized classifier with the lowest empirical error subject to fair classification constraints.
    
    Args:
        model: the classifier model object
        train_data (pd.DataFrame): the train split from the dataset, must be preprocessed beforehand
        test_data (pd.DataFrame): the test split from the dataset, must be preprocessed beforehand
        target_feature (str): the target feature
        protected (list): list of protected features
        privileged_classes (list(list)): a 2d list containing the privileged classes for each protected feature
        favorable_classes (list): a list of favorable classes in the target feature, restricted to n-1 length, where n is the number of classes
    """
    model_copy = deepcopy(model)
    
    sd_train = StandardDataset(train_data,target_feature,favorable_classes,protected,privileged_classes)
    sd_test = StandardDataset(test_data,target_feature,favorable_classes,protected,privileged_classes)
    p = []
    u = []
    for i, j in zip(protected,privileged_classes):
        p.append({i: j})
        u.append({i: [x for x in train_data[i].unique().tolist() if x not in j and not(np.isnan(x))]})
    
    print("Metrics Before Exponentiated Gradient Reduction: ")
    sd_test_pred = sd_test.copy()
    sd_test_pred.labels = model.predict(sd_test.features)
    before = compute_metrics(sd_test, sd_test_pred, u, p)
    
    np.random.seed(0)
    exp_grad_red = ExponentiatedGradientReduction(estimator=model, constraints="EqualizedOdds", drop_prot_attr=False)
    exp_grad_red.fit(sd_train)
    exp_grad_red_pred = exp_grad_red.predict(sd_test)
    print("\nMetrics After Exponentiated Gradient Reduction: ")
    after = compute_metrics(sd_test, exp_grad_red_pred, privileged_groups=p, unprivileged_groups=u)
    
    for i in after.keys():
        metrics_plot(metrics1=before,metrics2=after,threshold=0.2,metric_name=i,protected=protected[0]).show()
        
    return before, after
    
def meta_classifier(model, train_data, test_data, target_feature, protected, privileged_classes, favorable_classes=[1.0]):
    """
    Executes use of AI Fairness 360's Meta Classifier algorithm. The meta algorithm here takes the fairness metric as 
    part of the input and returns a classifier optimized w.r.t. that fairness metric.
    
    Args:
        model: the classifier model object
        train_data (pd.DataFrame): the train split from the dataset, must be preprocessed beforehand
        test_data (pd.DataFrame): the test split from the dataset, must be preprocessed beforehand
        target_feature (str): the target feature
        protected (list): list of protected features
        privileged_classes (list(list)): a 2d list containing the privileged classes for each protected feature
        favorable_classes (list): a list of favorable classes in the target feature, restricted to n-1 length, where n is the number of classes
    """
    
    model_copy = deepcopy(model)
    
    sd_train = StandardDataset(train_data,target_feature,favorable_classes,protected,privileged_classes)
    sd_test = StandardDataset(test_data,target_feature,favorable_classes,protected,privileged_classes)
    p = []
    u = []
    for i, j in zip(protected,privileged_classes):
        p.append({i: j})
        u.append({i: [x for x in train_data[i].unique().tolist() if x not in j and not(np.isnan(x))]})
    
    print("Metrics Before Meta Classifier: ")
    sd_test_pred = sd_test.copy()
    sd_test_pred.labels = model.predict(sd_test.features)
    before = compute_metrics(sd_test, sd_test_pred, u, p)
    
    debiased_model = MetaFairClassifier(tau=1, sensitive_attr=protected[0], type="sr").fit(sd_train)
    dataset_debiasing_test = debiased_model.predict(sd_test)
    print("\nMetrics After Meta Classifier: ")
    after = compute_metrics(sd_test, dataset_debiasing_test, u, p)
    
    for i in after.keys():
        metrics_plot(metrics1=before,metrics2=after,threshold=0.2,metric_name=i,protected=protected[0]).show()
    
    return before, after

def calibrated_eqodds(model, train_data, test_data, target_feature, protected, privileged_classes, favorable_classes=[1.0]):
    """
    Executes use of AI Fairness 360's Meta Classifier algorithm. The meta algorithm here takes the fairness metric as 
    part of the input and returns a classifier optimized w.r.t. that fairness metric.
    
    Args:
        model: the classifier model object
        train_data (pd.DataFrame): the train split from the dataset, must be preprocessed beforehand
        test_data (pd.DataFrame): the test split from the dataset, must be preprocessed beforehand
        target_feature (str): the target feature
        protected (list): list of protected features
        privileged_classes (list(list)): a 2d list containing the privileged classes for each protected feature
        favorable_classes (list): a list of favorable classes in the target feature, restricted to n-1 length, where n is the number of classes
    """
    
    model_copy = deepcopy(model)
    
    sd_train = StandardDataset(train_data,target_feature,favorable_classes,protected,privileged_classes)
    sd_test = StandardDataset(test_data,target_feature,favorable_classes,protected,privileged_classes)
    p = []
    u = []
    for i, j in zip(protected,privileged_classes):
        p.append({i: j})
        u.append({i: [x for x in train_data[i].unique().tolist() if x not in j and not(np.isnan(x))]})
    
    print("Metrics Before Calibrated Equalized Odds: ")
    sd_test_pred = sd_test.copy()
    sd_test_pred.labels = model_copy.predict(sd_test.features)
    before = compute_metrics(sd_test, sd_test_pred, u, p)
    
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = p,
                                     unprivileged_groups = u,
                                     cost_constraint="weighted",
                                     seed=1234)
    sd_train_pred = sd_train.copy()
    sd_train_pred.labels = model_copy.predict(sd_train.features)
    cpp = cpp.fit(sd_train, sd_train_pred)
    dataset_transf_test_pred = cpp.predict(sd_test)
    print("\nMetrics After Calibrated Equalized Odds: ")
    after = compute_metrics(sd_test_pred, dataset_transf_test_pred, u, p)
    
    for i in after.keys():
        metrics_plot(metrics1=before,metrics2=after,threshold=0.2,metric_name=i,protected=protected[0]).show()
    
    return before, after

def reject_option(model, train_data, test_data, target_feature, protected, privileged_classes, favorable_classes=[1.0]):
    """
    Executes use of AI Fairness 360's Meta Classifier algorithm. The meta algorithm here takes the fairness metric as 
    part of the input and returns a classifier optimized w.r.t. that fairness metric.
    
    Args:
        model: the classifier model object
        train_data (pd.DataFrame): the train split from the dataset, must be preprocessed beforehand
        test_data (pd.DataFrame): the test split from the dataset, must be preprocessed beforehand
        target_feature (str): the target feature
        protected (list): list of protected features
        privileged_classes (list(list)): a 2d list containing the privileged classes for each protected feature
        favorable_classes (list): a list of favorable classes in the target feature, restricted to n-1 length, where n is the number of classes
    """
    
    model_copy = deepcopy(model)
    
    sd_train = StandardDataset(train_data,target_feature,favorable_classes,protected,privileged_classes)
    sd_test = StandardDataset(test_data,target_feature,favorable_classes,protected,privileged_classes)
    p = []
    u = []
    for i, j in zip(protected,privileged_classes):
        p.append({i: j})
        u.append({i: [x for x in train_data[i].unique().tolist() if x not in j and not(np.isnan(x))]})
    
    print("Metrics Before Reject Option: ")
    sd_test_pred = sd_test.copy()
    sd_test_pred.labels = model_copy.predict(sd_test.features)
    before = compute_metrics(sd_test, sd_test_pred, u, p)
    
    np.random.seed(0)
    ROC = RejectOptionClassification(unprivileged_groups=u, 
                                 privileged_groups=p, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name="Statistical parity difference",
                                  metric_ub=0.05, metric_lb=-0.05)
    sd_train_pred = sd_train.copy()
    sd_train_pred.labels = model_copy.predict(sd_train.features)
    ROC = ROC.fit(sd_train, sd_train_pred)
    dataset_transf_test_pred = ROC.predict(sd_test)
    print("\nMetrics After Reject Option: ")
    after = compute_metrics(sd_test_pred, dataset_transf_test_pred, u, p)
    
    for i in after.keys():
        metrics_plot(metrics1=before,metrics2=after,threshold=0.2,metric_name=i,protected=protected[0]).show()
    
    return before, after

# def compare_algorithms(b, di, rw, egr, mc, ceo, ro, metric_name="Disparate impact"):
#     """
#     Produce a scatter plot to compare the effectiveness of each algorithm to a specific metric, and also observe 
#     the fairness vs. performance trade-off
    
#     Args:
#     di (OrderedDict): contains the fairness scores of the model after the disparate impact remover algorithm
#     rw (OrderedDict): contains the fairness scores of the model after the reweighing algorithm
#     egr (OrderedDict): contains the fairness scores of the model after the exponentiated gradient reduction algorithm
#     mc (OrderedDict): contains the fairness scores of the model after the meta-classifier algorithm
#     metric_name (str): the fairness metric to analyze
#     """
#     fig = px.scatter(x = [b[metric_name],di[metric_name], rw[metric_name], egr[metric_name], mc[metric_name], ceo[metric_name], ro[metric_name]], 
#                      y = [b['Balanced accuracy'], di['Balanced accuracy'], rw['Balanced accuracy'], egr['Balanced accuracy'], mc['Balanced accuracy'], ceo['Balanced accuracy'], ro['Balanced accuracy']],
#                      color = ["Original", "Disparate Impact Remover", "Reweighing", "Exponentiated Gradient Reduction", "Meta Classifier", "Calibrated Equalized Odds", "Reject Option"],
#                      #symbol = ["Disparate Impact Remover", "Reweighing", "Exponentiated Gradient Reduction", "Meta Classifier"],
#                      labels = dict(x=metric_name, y="Balanced accuracy"),
#                      title = f"Balanced accuracy vs. {metric_name}")
#     fig.add_vline(x= 0 if metric_name != "Disparate impact" else 1, line_width=1, line_color="black")
#     fig.update_traces(marker_size=10)
#     return fig

def compare_algorithms(b, di, rw, egr, mc, ceo, ro, threshold, metric_name="Disparate impact"):
    """
    Produce a scatter plot to compare the effectiveness of each algorithm to a specific metric, and also observe 
    the fairness vs. performance trade-off
    
    Args:
    di (OrderedDict): contains the fairness scores of the model after the disparate impact remover algorithm
    rw (OrderedDict): contains the fairness scores of the model after the reweighing algorithm
    egr (OrderedDict): contains the fairness scores of the model after the exponentiated gradient reduction algorithm
    mc (OrderedDict): contains the fairness scores of the model after the meta-classifier algorithm
    metric_name (str): the fairness metric to analyze
    """
    upper_bound = max([b[metric_name],di[metric_name], rw[metric_name], egr[metric_name], mc[metric_name], ceo[metric_name], ro[metric_name]]) + 0.5
    upper_bound = upper_bound if upper_bound > 1.2 else upper_bound
    lower_bound = min([b[metric_name],di[metric_name], rw[metric_name], egr[metric_name], mc[metric_name], ceo[metric_name], ro[metric_name]]) - 0.5
    lower_bound = lower_bound if lower_bound < -0.2 else -0.3
#     fig = px.bar(x = [b[metric_name],di[metric_name], rw[metric_name], egr[metric_name], mc[metric_name], ceo[metric_name], ro[metric_name]], 
#                  y = ["","","","","","",""],
#                  color = ["Original", "Disparate Impact Remover", "Reweighing", "Exponentiated Gradient Reduction", "Meta Classifier", "Calibrated Equalized Odds", "Reject Option"],
#                  #symbol = ["Original", "Disparate Impact Remover", "Reweighing", "Exponentiated Gradient Reduction", "Meta Classifier", "Calibrated Equalized Odds", "Reject Option"],
#                  labels = dict(x="Score", y=""),
#                  title = f"Comparison of Algorithms for {metric_name} metric",
#                  color_discrete_sequence=px.colors.qualitative.D3,
#                  barmode="group")
    fig = px.scatter(x = [b[metric_name],di[metric_name], rw[metric_name], egr[metric_name], mc[metric_name], ceo[metric_name], ro[metric_name]], 
                     y = [b['Balanced accuracy'], di['Balanced accuracy'], rw['Balanced accuracy'], egr['Balanced accuracy'], mc['Balanced accuracy'], ceo['Balanced accuracy'], ro['Balanced accuracy']],
                     color = ["Original", "Disparate Impact Remover", "Reweighing", "Exponentiated Gradient Reduction", "Meta Classifier", "Calibrated Equalized Odds", "Reject Option"],
                     #symbol = ["Disparate Impact Remover", "Reweighing", "Exponentiated Gradient Reduction", "Meta Classifier"],
                     labels = dict(x=metric_name, y="Balanced accuracy"),
                     title = f"Balanced accuracy vs. {metric_name}",
                     height=500,
                     )
    if metric_name == 'Disparate impact' or metric_name == 'Balanced accuracy':
        line = {'type': 'line', 'x0': 0, 'x1': 0, 'y0': 0, 'y1': 1, 'line': {'color': "#371ea3", 'width': 1.5}}
        left_red = {'type': "rect", 'x0': lower_bound, 'y0': 0, 'x1': 1-threshold, 'y1': 1, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
        middle_green = {'type': "rect", 'x0': 1-threshold, 'y0': 0, 'x1': 1+threshold if metric_name=='Disparate impact' else 1, 'y1': 1, 'line': {'width': 0}, 'fillcolor': '#008000', 'layer': 'below', 'opacity': 0.1}            
        right_red = {'type': "rect", 'x0': 1+threshold if upper_bound > 1+threshold else 1, 'y0': 0, 'x1': upper_bound, 'y1': 1, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
    else:
        line = {'type': 'line', 'x0': 0, 'x1': 0, 'y0': 0, 'y1': 1, 'line': {'color': "#371ea3", 'width': 1.5}}
        left_red = {'type': "rect", 'x0': lower_bound, 'y0': 0, 'x1': -threshold, 'y1': 1, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
        middle_green = {'type': "rect", 'x0': -threshold, 'y0': 0, 'x1': threshold, 'y1': 1, 'line': {'width': 0}, 'fillcolor': '#008000', 'layer': 'below', 'opacity': 0.1}            
        right_red = {'type': "rect", 'x0': threshold, 'y0': 0, 'x1': upper_bound, 'y1': 1, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
    #fig = fig.add_shape(line)
    fig.add_trace(
        go.Scatter(
            x=list(np.zeros(2000, dtype=int)) if metric_name != "Disparate impact" else list(np.ones(2000, dtype=int)),
            y=[i/2000 for i in range(0,2000,1)],
            mode="lines",
            line=go.scatter.Line(color="gray"),
            showlegend=True,
            hoverinfo='text',
            hovertext='Ideal Value',
            name='Ideal Value'
        )
    )
    fig = fig.add_shape(middle_green)
    fig = fig.add_shape(left_red)
    fig = fig.add_shape(right_red)
    #fig.add_vline(x= 0 if metric_name != "Disparate impact" else 1, line_width=1, line_color="black")
    fig.update_traces(marker_size=10)
    fig.update_layout(hovermode='closest',yaxis_range=[min([b['Balanced accuracy'], di['Balanced accuracy'], rw['Balanced accuracy'], egr['Balanced accuracy'], mc['Balanced accuracy'], ceo['Balanced accuracy'], ro['Balanced accuracy']])-0.001,max([b['Balanced accuracy'], di['Balanced accuracy'], rw['Balanced accuracy'], egr['Balanced accuracy'], mc['Balanced accuracy'], ceo['Balanced accuracy'], ro['Balanced accuracy']])+0.001])
    return fig

def algo_exp(method):
    if method == "Disparate Impact Remover":
        return  """
    Disparate impact remover is a preprocessing technique that edits feature values increase group fairness while preserving rank-ordering within groups.
                """
    elif method == "Reweighing":
        return  """
    Reweighing is a preprocessing technique that Weights the examples in each (group, label) combination differently to ensure fairness before classification.
                """
    elif method == "Exponentiated Gradient Reduction":
        return  """
    Exponentiated gradient reduction is an in-processing technique that reduces fair classification to a sequence of cost-sensitive classification problems, returning a randomized classifier with the lowest empirical error subject to fair classification constraints.
                """
    elif method == "Meta Classifier":
        return  '''
    The meta algorithm here takes the fairness metric as part of the input and returns a classifier optimized w.r.t. that fairness metric.
                '''
    elif method == "Calibrated Equalized Odds":
        return  """
    Calibrated equalized odds postprocessing is a post-processing technique that optimizes over calibrated classifier score outputs to find probabilities with which to change output labels with an equalized odds objective.
                """
    elif method == "Reject Option":
        return  """
    Reject option classification is a postprocessing technique that gives favorable outcomes to unpriviliged groups and unfavorable outcomes to priviliged groups in a confidence band around the decision boundary with the highest uncertainty.
                """
    else:
        return ""