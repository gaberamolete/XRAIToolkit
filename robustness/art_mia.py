#import libraries 
import numpy as np
from sklearn.linear_model import LinearRegression
from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
import plotly.express as px

def art_mia(x_train, y_train, x_test, y_test, art_extra_dict, key, attack_train_ratio=0.3):
    """
    Returns the inferred train and inferred test 

    Parameters
    ----------
    x_train : numpy array
    y_train : numpy array
    x_test : numpy array
    y_test : numpy array
    art_extra_dict : dictionary of ART and extra models
    """

    #train the MLP regression model 
    art_regressor = art_extra_dict[key][0]
    bb_attack = MembershipInferenceBlackBox(art_regressor, input_type='loss')
 
    #set the attack train ratio
    attack_train_size = int(X_train.shape[0] * attack_train_ratio)
    attack_test_size = int(X_test.shape[0] * attack_train_ratio)
    bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size], x_test[:attack_test_size], y_test[:attack_test_size])

    #generate inferences from model output 
    inferred_train = bb_attack.infer(x_train, y_train)
    inferred_test = bb_attack.infer(x_test, y_test)

    return inferred_train, inferred_test

def art_generate_predicted(inferred_train, inferred_test):
    predicted = np.concatenate((inferred_train, inferred_test))
    return predicted

def art_generate_actual(inferred_train, inferred_test):
    actual = np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))
    return actual

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

def mia_viz(precision,recall):
    fig = px.bar(x=[precision, recall],y=["",""],color=["Precision", "Recall"], labels={'x':'Score','y':'Metric'}, title="Membership Inference Attacker Precision and Recall", color_discrete_sequence=px.colors.qualitative.D3,  barmode='group')
    fig['layout'].update(height=300)
    green = {'type': "rect", 'x0': 0, 'y0': -0.5, 'x1': 0.5, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#008000', 'layer': 'below', 'opacity': 0.1}
    red = {'type': "rect", 'x0': 0.5, 'y0': -0.5, 'x1': 1, 'y1': 0.5, 'line': {'width': 0}, 'fillcolor': '#ff0000', 'layer': 'below', 'opacity': 0.1}
    fig = fig.add_shape(green)
    fig = fig.add_shape(red)
    fig.add_vline(x = 0.5, line_width=1, line_color="black")
    return fig