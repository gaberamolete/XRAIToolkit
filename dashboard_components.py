import numpy as np
import pandas as pd
import shap
import base64
import io

#Performance overview 
from fairness import model_performance

# Fairness
from fairness import fairness

# Local explanation
from local_exp import dice_exp, exp_cf
from local_exp import dalex_exp, break_down, interactive
from local_exp import Predictor, exp_qii, get_feature_names, cp_profile
from local_exp import initiate_shap_loc, shap_waterfall, shap_force_loc, shap_bar_loc

# Global explanation
from global_exp import pd_profile, var_imp
from global_exp import ld_profile, al_profile, compare_profiles
from global_exp import initiate_shap_glob, shap_bar_glob, shap_summary, shap_dependence, shap_force_glob

# Stability
from stability import psi_list, generate_psi_df
#from stability import PageHinkley
from stability import ks

#ExplainerDashboard
from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *
import dash as html
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, dash_table, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class BlankComponent(ExplainerComponent):
    """
    Return an empty component to the dashboard
    """
    def __init__(self, explainer, title="Blank", name=None):
        super().__init__(explainer, title=title)
        
    def layout(self):
        return None
    
class FairnessIntroComponent(ExplainerComponent):
    """
    A component class for the introduction of fairness in AI/ML, containing info on various fairness metrics for 
    classification/regression.
    """
    def __init__(self, explainer, title="Fairness", name=None):
        super().__init__(explainer, title=title)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3("Fairness Overview")
                ]),
                dbc.CardBody([
                    html.Div(dcc.Markdown('''
                        **For classification;**
                        Metrics are Disparity impact (DI), Equal opportunity (EOP), Equalized odds (EOD), It provides other indexes like `Accuracy`,`True_positive_rate`, `False_positive_rate`, `False_negative_rate`, `predicted_as_positive`, `Recall`, `Precision`, `F1 score` and `AUC`  of overall data and defined groups in a table for comparison. 

                        #### Equal opportunity (EOP)
                        Under equal opportunity we consider a model to be fair if the TPRs of the privileged and unprivileged groups are equal. 
                        In practice, we will give some leeway for statistic uncertainty. 
                        We can require the differences to be less than a certain cutoff.

                        #### Equalized odds (EOD)
                        This can be interpreted as the percentage of people who have wrongfully benefited from the model.

                        #### Disparity Index (DI)
                         That is the predicted as positive for the normal group must not be less than cutoff of that of the protected group

                        #### Cutoff
                        The question is what cutoff should we use? There is actually no good answer to that. It will depend on your industry and application. If your model has significant consequences, like for mortgage applications, you will need a stricter cutoff. The cutoff may even be defined by law. Either way, it is important to define the cutoffs before you measure fairness.
                         0.8 seems to be a good defualt value for that. 

                         **Regression;**
                         Having Explainers, we are able to assess models' fairness. To make sure that the models are fair, we will be checking three independence criteria. These are:

                        independence: R⊥A
                        separation: R⊥A ∣ Y
                        sufficiency: Y⊥A ∣ R
                        Where:

                        A - protected group
                        Y - target
                        R - model's prediction
                        In the approach described in Steinberg, D., et al. (2020), the authors propose a way of checking this independence.
                        ***More info about metrics of regression***
                        https://arxiv.org/pdf/2001.06089.pdf
                    '''))
                ])
            ])
        ])
    
class FairnessCheckRegComponent(ExplainerComponent):
    """
    A component class for displaying the output of the fairness function on the dashboard for regression cases.
    """
    def __init__(self, explainer, model, X_test, Y_test, title="Fairness", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            title (str, optional): title of the component. Defaults to "Fairness".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3("Fairness Check")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                This component utilizes the `fairness()` function, taking the model, data, protected groups, fairness metric and treshold as an input, that will then calculates whether the model is fair to the defined protected groups or not. 
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Select the Protected Group Feature", style={"padding-left":"30px"})
                        ]),
                        dbc.Col([
                            html.H5("Select the Protected Group Value")
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="protected-group-feature",
                                options=sorted(self.X_test.columns),
                                placeholder="Select the Protected Group Feature",
                            ),  style={"padding-left":"30px","width":"100%"})
                        ]),
                        dbc.Col([
                            html.Div(id="protected-group-value",  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Select the Metric", style={"padding-left":"30px"})
                        ]),
                        dbc.Col([
                            html.H5("Select the Threshold")
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="metrics",
                                options=["DI", "EOD", "EOP"],
                                placeholder="Select the Metric",
                                value="DI"
                            ),  style={"padding-left":"30px","width":"100%"})
                        ]),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="threshold",
                                type="number",
                                min=0,
                                max=1,
                                step=0.1,
                                value=0.8,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-fairchk", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(html.Pre(id="text"), style={"padding-left":"50px"})
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(dbc.CardBody(html.Div(dcc.Loading(dcc.Graph(id="graph1")))))
                        ]),
                        dbc.Col([
                            dbc.Card(dbc.CardBody(html.Div(dcc.Loading(dcc.Graph(id="graph2")))))
                        ])
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("protected-group-value","children"),
            Input("protected-group-feature","value")
        )
        def update_dropdown(feature):
            if feature in self.X_test.select_dtypes(exclude = np.number).columns.tolist():
                return dcc.Dropdown(options= sorted(self.X_test[feature].unique()), placeholder="Select the Protected Group Value", id="value-frchk")
            elif feature in self.X_test.select_dtypes(include = ['int64']).columns.tolist():
                return dcc.RangeSlider(self.X_test[feature].min(), self.X_test[feature].max(), step=1, marks=None,tooltip={"placement": "bottom", "always_visible": True}, id="value-frchk")
            else:
                return dcc.RangeSlider(self.X_test[feature].min(), self.X_test[feature].max(), marks=None, tooltip={"placement": "bottom", "always_visible": True}, id="value-frchk")
                
            
        @app.callback(
            [
                Output("text","children"),
                Output("graph1","figure"),
                Output("graph2","figure"),
            ],
            Input("button-fairchk","n_clicks"),
            State("protected-group-feature","value"),
            State("value-frchk","value"),
            State("metrics","value"),
            State("threshold","value"), prevent_initial_call=True
        )
        def update_text_figure(n_clicks, feature, value, metric, threshold):
            protected_group = {}
            protected_group[feature] = value
            contents, fig1, fig2 = fairness(self.model, self.X_test, self.Y_test, protected_group, metric=metric, threshold=threshold,reg=True,xextra=False)
            return contents[0], fig1[0], fig2[0]
    
class FairnessCheckClfComponent(ExplainerComponent):
    """
    A component class for displaying the output of the fairness function on the dashboard for classification cases.
    """
    def __init__(self, explainer, model, X_test, Y_test, title="Fairness", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            title (str, optional): title of the component. Defaults to "Fairness".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3("Fairness Check")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                This component utilizes the `fairness()` function, taking the model, data, protected groups, fairness metric and treshold as an input, that will then calculates whether the model is fair to the defined protected groups or not. 
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Select the Protected Group Feature", style={"padding-left":"30px"})
                        ]),
                        dbc.Col([
                            html.H5("Select the Protected Group Value")
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="protected-group-feature",
                                options=sorted(self.X_test.columns),
                                placeholder="Select the Protected Group Feature",
                            ),  style={"padding-left":"30px","width":"100%"})
                        ]),
                        dbc.Col([
                            html.Div(id="protected-group-value",  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Select the Metric", style={"padding-left":"30px"})
                        ]),
                        dbc.Col([
                            html.H5("Select the Threshold")
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="metrics",
                                options=["DI", "EOD", "EOP"],
                                placeholder="Select the Metric",
                                value="DI"
                            ),  style={"padding-left":"30px","width":"100%"})
                        ]),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="threshold",
                                type="number",
                                min=0,
                                max=1,
                                step=0.1,
                                value=0.8,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-fairchk", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(html.Pre(id="text"), style={"padding-left":"50px"})
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Div(id="table-fair")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'auto'})
                        ]),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
#                     dbc.Row([
#                         dbc.Col([
#                             dbc.Card(dbc.CardBody(html.Div(dcc.Loading(dcc.Graph(id="graph-fair")))))
#                         ]),
#                     ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("protected-group-value","children"),
            Input("protected-group-feature","value")
        )
        def update_dropdown(feature):
            if feature in self.X_test.select_dtypes(exclude = np.number).columns.tolist():
                return dcc.Dropdown(options= sorted(self.X_test[feature].unique()), placeholder="Select the Protected Group Value", id="value-frchk")
            elif feature in self.X_test.select_dtypes(include = ['int64']).columns.tolist():
                return dcc.RangeSlider(self.X_test[feature].min(), self.X_test[feature].max(), step=1, marks=None,tooltip={"placement": "bottom", "always_visible": True}, id="value-frchk")
            else:
                return dcc.RangeSlider(self.X_test[feature].min(), self.X_test[feature].max(), marks=None, tooltip={"placement": "bottom", "always_visible": True}, id="value-frchk")
            
        @app.callback(
            [
                Output("text","children"),
                Output("table-fair","children"),
                #Output("graph-fair", "figure")
            ],
            Input("button-fairchk","n_clicks"),
            State("protected-group-feature","value"),
            State("value-frchk","value"),
            State("metrics","value"),
            State("threshold","value"), prevent_initial_call=True
        )
        def update_text_figure(n_clicks, feature, value, metric, threshold):
            protected_group = {}
            protected_group[feature] = value
            fairness_index, fairness_report, a = fairness(self.model, self.X_test, self.Y_test, protected_group, metric=metric, threshold=threshold,reg=False,xextra=False)
            contents = [f"For the protected feature {feature} of {value}, the {metric} is {fairness_index}\n"]
            d=1-threshold
            if ((fairness_index<=1+d) and (fairness_index>=1-d)):
                contents.append("The model is fair towards the {} in {}".format(feature,value))
            elif ((fairness_index>1+d) and (fairness_index<=1+2*d)):
                contents.append("The model is Partially Advantages towards the {} in {}".format(feature,value))
            elif ((fairness_index>1+2*d)):
                contents.append("The model is Totally Advantages towards the {} in {}".format(feature,value))
            elif ((fairness_index>=1-2*d) and (fairness_index<1-d)):
                 contents.append("The model is Partially Disdvantages towards the {} in {}".format(feature,value))
            elif ((fairness_index<1-2*d)):
                 contents.append("The model is Totally Disadvantages towards the {} in {}".format(feature,value))
            df = dash_table.DataTable(fairness_report.to_dict('records'), [{"name": i, "id": i} for i in fairness_report.columns],style_data={'whiteSpace':'normal','height':'auto'},fill_width=False)
            fig = px.scatter([fairness_report[fairness_report.Variable==feature]["Accuracy"][0]], [fairness_index], labels={"x":metric,"index":"Accuracy"})
            return contents, df, #fig

class ModelPerformanceRegComponent(ExplainerComponent):
    """
    A component class for displaying the output of the model_performance function in the fairness module on the dashboard for
    regression cases.
    """
    def __init__(self, explainer, model, X_test, Y_test, X_train, Y_train, test_data, train_data, target_feature, title="Performance", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            test_data (pd.DataFrame): concatenated version of X_test and Y_test
            train_data (pd.DataFrame): concatenated version of X_train and Y_train
            target_feature (str): target feature
            title (str, optional): title of the component. Defaults to "Performance".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train = X_train
        self.Y_train = Y_train
        self.test_data = test_data
        self.train_data = train_data
        self.target_feature = target_feature
        self.protected_group = {}
    
    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Overall Performance"),
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([   
                                    html.Div(dcc.Markdown('''
                                       This component used the `model_performance()` function to give an overview on model performance on test and train datasets. This also calculate performance for the protected group(s) vs all other data points.
                                   ''')) 
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Protected Group Feature: ", style={"padding-top":"5px"}),
                                ], width="auto"),
                                dbc.Col([
                                    html.Div(dcc.Dropdown(
                                        id="protected-group-feature1",
                                        options=sorted(self.X_test.columns),
                                        placeholder="Select the Protected Group Feature",
                                    ),  style={"padding-left":"30px","width":"100%"})
                                ]),
                                dbc.Col([
                                    html.Div("Protected Group Value: ", style={"padding-top":"5px"}),
                                ], width="auto"),
                                dbc.Col([
                                    html.Div(dcc.Dropdown(
                                        id="protected-group-value1",
                                        placeholder="Select the Protected Group Value",
                                    ),  style={"width":"100%","padding-right":"30px"})
                                ]),
                                dbc.Col([
                                    html.Div(dbc.Button("Add", id="button", n_clicks=0), style={"padding-left":"50px"})
                                ], width="auto")
                            ]),
                            html.Br(),
                            dbc.Row([
                                html.Div(id="alert-mp", style={"margin":"auto"})
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Train or Test: ", style={"padding-top":"5px"})
                                ], width="auto"),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="train-test",
                                        options=[
                                                 {
                                                  "label": "Train Data",
                                                  "value": "train",
                                                 },
                                                 {
                                                  "label": "Test Data",
                                                  "value": "test",
                                                 }, 
                                        ],
                                        placeholder="Train or Test",
                                    ),
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div()
                                ]),
                                dbc.Col([
                                    html.Div(dbc.Button("Compute", id="button-performance", n_clicks=0), style={"margin":"auto"})
                                ], width="auto"),
                                dbc.Col([
                                    html.Div()
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div()
                                ]),
                                dbc.Col([
                                    html.Div(dcc.Loading(html.Div(id="table")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px'})
                                ]),
                                dbc.Col([
                                    html.Div()
                                ])
                            ])
                        ])
                    ])
                ]),
            ]),
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("protected-group-value1","options"),
            Input("protected-group-feature1","value"), prevent_initial_call=True
        )
        def update_dropdown(feature):
            return sorted(self.X_test[feature].unique())
        
        @app.callback(
            Output("none","children"),
            Input("button","n_clicks"),
            State("protected-group-feature1","value"),
            State("protected-group-value1","value"), prevent_initial_call=True
        )
        def update_protected_groups(n_clicks, feature, value):
            self.protected_group[feature]=value
            
        @app.callback(
            Output("alert-mp","children"),
            Input("button","n_clicks")
        )
        def create_alert(n_clicks):
            return dbc.Alert("Successfully Added! Click Compute or Add More.", color="success", duration=3000) 
        
        @app.callback(
            Output("table","children"),
            Input("button-performance","n_clicks"),
            State("train-test","value"), prevent_initial_call=True
        )
        def update_table(n_clicks,isTrain):
            if None in self.protected_group.values():
                self.protected_group = {k: v for k, v in self.protected_group.items() if v is not None}
            df1, df2 = model_performance(self.model, self.X_test, self.Y_test, self.X_train, self.Y_train, self.test_data, self.train_data, self.target_feature, protected_groups=self.protected_group, reg=True)
            df1 = df1.round(5)
            df2 = df2.round(5)
            df1["Mean Absolute Percentage Error"] = df1["Mean Absolute Percentage Error"].apply(lambda x: '{:.2f}%'.format(x*100))
            df2["Mean Absolute Percentage Error"] = df2["Mean Absolute Percentage Error"].apply(lambda x: '{:.2f}%'.format(x*100))
            if isTrain == "train":
                return dash_table.DataTable(df2.to_dict('records'), [{"name": i, "id": i} for i in df2.columns],style_data={'whiteSpace':'normal','height':'auto'},fill_width=False)
            elif isTrain == "test":
                return dash_table.DataTable(df1.to_dict('records'), [{"name": i, "id": i} for i in df1.columns],style_data={'whiteSpace':'normal','height':'auto'},fill_width=False)
            
class ModelPerformanceClfComponent(ExplainerComponent):
    """
    A component class for displaying the output of the model_performance function in the fairness module on the dashboard for
    classification cases.
    """
    def __init__(self, explainer, model, X_test, Y_test, X_train, Y_train, test_data, train_data, target_feature, title="Performance", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            test_data (pd.DataFrame): concatenated version of X_test and Y_test
            train_data (pd.DataFrame): concatenated version of X_train and Y_train
            target_feature (str): target feature
            title (str, optional): title of the component. Defaults to "Performance".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train = X_train
        self.Y_train = Y_train
        self.test_data = test_data
        self.train_data = train_data
        self.target_feature = target_feature
        self.protected_group = {}
    
    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Overall Performance"),
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([   
                                    html.Div(dcc.Markdown('''
                                       This component used the `model_performance()` function to give an overview on model performance on test and train datasets. This also calculate performance for the protected group(s) vs all other data points.
                                   ''')) 
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Protected Group Feature: ", style={"padding-top":"5px"}),
                                ], width="auto"),
                                dbc.Col([
                                    html.Div(dcc.Dropdown(
                                        id="protected-group-feature1",
                                        options=sorted(self.X_test.columns),
                                        placeholder="Select the Protected Group Feature",
                                    ),  style={"padding-left":"30px","width":"100%"})
                                ]),
                                dbc.Col([
                                    html.Div("Protected Group Value: ", style={"padding-top":"5px"}),
                                ], width="auto"),
                                dbc.Col([
                                    html.Div(dcc.Dropdown(
                                        id="protected-group-value1",
                                        placeholder="Select the Protected Group Value",
                                    ),  style={"width":"100%","padding-right":"30px"})
                                ]),
                                dbc.Col([
                                    html.Div(dbc.Button("Add", id="button-add", n_clicks=0), style={"padding-left":"50px"})
                                ], width="auto")
                            ]),
                            html.Br(),
                            dbc.Row([
                                html.Div(id="alert-mp", style={"margin":"auto"})
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Train or Test: ", style={"padding-top":"5px"})
                                ], width="auto"),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="train-test",
                                        options=[
                                                 {
                                                  "label": "Train Data",
                                                  "value": "train",
                                                 },
                                                 {
                                                  "label": "Test Data",
                                                  "value": "test",
                                                 }, 
                                        ],
                                        placeholder="Train or Test",
                                    ),
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div()
                                ]),
                                dbc.Col([
                                    html.Div(dbc.Button("Compute", id="button-performance", n_clicks=0), style={"margin":"auto"})
                                ], width="auto"),
                                dbc.Col([
                                    html.Div()
                                ])
                            ]),
                            html.Br(),
                            dbc.Row([
                                html.Div(dcc.Loading(dcc.Graph(id="graph-cm")), style={"margin":"auto"})
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Div()
                                ]),
                                dbc.Col([
                                    html.Div(dcc.Loading(html.Div(id="table-clf")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px'})
                                ]),
                                dbc.Col([
                                    html.Div()
                                ])
                            ]),
                            html.Div(id='none0',children=[],style={'display': 'none'})
                        ])
                    ])
                ]),
            ]),
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("protected-group-value1","options"),
            Input("protected-group-feature1","value"), prevent_initial_call=True
        )
        def update_dropdown(feature):
            return sorted(self.X_test[feature].unique())
        
        @app.callback(
            Output("none0","children"),
            Input("button-add","n_clicks"),
            State("protected-group-feature1","value"),
            State("protected-group-value1","value"), prevent_initial_call=True
        )
        def update_protected_groups(n_clicks, feature, value):
            self.protected_group[feature]=value

        @app.callback(
            Output("alert-mp","children"),
            Input("button-add","n_clicks"), prevent_initial_call=True
        )
        def create_alert(n_clicks):
            return dbc.Alert("Successfully Added! Click Compute or Add More.", color="success", duration=3000)    
        
        @app.callback(
            Output("graph-cm","figure"),
            #Output("graph-bar","figure"),
            Output("table-clf","children"),
            Input("button-performance","n_clicks"),
            State("train-test","value"), prevent_initial_call=True
        )
        def update_table(n_clicks,isTrain):
            if None in self.protected_group.values():
                self.protected_group = {k: v for k, v in self.protected_group.items() if v is not None}
            df1, df2, df3 = model_performance(self.model, self.X_test, self.Y_test, self.X_train, self.Y_train, self.test_data, self.train_data, self.target_feature, protected_groups=self.protected_group, reg=False)
            df2 = df2.round(5)
            df3 = df3.round(5)
            if isTrain == 'train':
                fig1 = px.imshow(df1[0], text_auto='.1f', labels=dict(x="Predicted Label", y="True Label"),x=['[0]','[1]'],y=['[0]','[1]'], title="Model Results on Train Set")
                #fig2 = px.bar(df2["Split"][:limit_on_plot_number], df2["Accuracy"][:limit_on_plot_number])
                return fig1, dash_table.DataTable(df2.to_dict('records'), [{"name": i, "id": i} for i in df2.columns],style_data={'whiteSpace':'normal','height':'auto'},fill_width=False)
            else:
                fig1 = px.imshow(df1[1], text_auto='.1f', labels=dict(x="Predicted Label", y="True Label"),x=['[0]','[1]'],y=['[0]','[1]'], title="Model Results on Test Set")
                #fig2 = px.bar(df3["Split"][:limit_on_plot_number], df3["Accuracy"][:limit_on_plot_number])
                return fig1, dash_table.DataTable(df3.to_dict('records'), [{"name": i, "id": i} for i in df3.columns],style_data={'whiteSpace':'normal','height':'auto'},fill_width=False)
            
class EntryIndexComponent(ExplainerComponent):
    """
    A component class for selecting the local prediction to analyze in all local explanation components in the dashboard.
    """
    def __init__(self, explainer, X_test, title="Entry Index", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            X_test (pd.DataFrame): X_test
            title (str, optional): title of the component. Defaults to "Entry Index".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.X_test = X_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3("Prediction to Analyze")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.P("Entry Index: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="idx",
                                type="number",
                                min=0,
                                max=len(list(self.X_test.index))-1,
                                value=0,
                                step=1,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div("Selected Entry", style={"padding-left":"20px"}),
                        html.Div(dcc.Loading(html.Div(id="table-entry")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px'})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("table-entry","children"),
            Input("idx","value"),
        )
        def update_table(idx):
            X = self.X_test[idx:idx+1]
            return dash_table.DataTable(X.to_dict('records'), [{"name": i, "id": i} for i in X.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False)
    
class BreakDownComponent(ExplainerComponent):
    """
    A component class for displaying the output of the breakdown function from the local explanation module in the dashboard.
    """
    def __init__(self, explainer, model, X_train, Y_train, X_test, title="Break Down", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            title (str, optional): title of the component. Defaults to "Break Down".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Break Down")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **Break Down**: How can your model response be explained by the model's features? What are the most important features of the model? This function is best for why questions, or when you have a moderate number of features. Just be careful when features are correlated.
                                
                                Warning: This might take a while for large datasets
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-breakdown", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="break_down")), style={"margin":"auto"}),
                    ]),
                    html.Div(id='none',children=[],style={'display': 'none'})
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("break_down","figure"),
            Input("button-breakdown","n_clicks"),
            State("idx","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, idx):
            exp, obs = dalex_exp(list(self.model.values())[0], self.X_train, self.Y_train, self.X_test, idx)
            _, fig = break_down(exp, obs)
            return fig

class AdditiveComponent(ExplainerComponent):
    """
    A component class for displaying the output of the breakdown function w/ specific order specified in the dashboard.
    """
    def __init__(self, explainer, model, X_train, Y_train, X_test, title="Additive", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            title (str, optional): title of the component. Defaults to "Additive".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Additive")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **Additive**: How does the average model response change when new features are being fixed in the observation of interest? What if we force a specific order of variables?
                                
                                Warning: This might take a while for large datasets
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Select Order: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="order",
                                options=sorted(self.X_train.columns),
                                placeholder="Select Continuous Feature",
                                multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-additive", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="additive")), style={"margin":"auto"})
                    ]),
                    dbc.Row([
                        html.Br(),
                        html.Div(dcc.Loading(html.Div(id="table1")), style={"margin":"auto"})
                    ]),
                    
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("additive","figure"),
            Output("table1","children"),
            Input("button-additive","n_clicks"),
            State("order","value"),
            State("idx", "value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, order, idx):
            exp, obs = dalex_exp(list(self.model.values())[0], self.X_train, self.Y_train, self.X_test, idx)
            df, fig = break_down(exp, obs, order)
            return fig, dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False)

class CeterisParibusComponent(ExplainerComponent):
    """
    A component class for displaying the output of the cp profile function on the local explanation module to the dashboard.
    """
    def __init__(self, explainer, model, X_train, Y_train, X_test, cont, title="Ceteris Paribus", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            cont (pd.DataFrame): a subset of X_train containing only the continuous features
            title (str, optional): title of the component. Defaults to "Ceteris Paribus".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.cont = cont
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Ceteris Paribus")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **Ceteris Paribus**: How would the model response change for a particular observation if only a single feature is changed? This function is best for what if questions. Just be careful when features are correlated.
                                
                                Warning: This might take a while for large datasets
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Continuous Feature: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="prof-vars",
                                options=sorted(self.X_train.select_dtypes(include='number').columns),
                                placeholder="Select Continuous Feature",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-ceteris", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="ceteris-paribus")), style={"margin":"auto"})
                    ])
                    
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("ceteris-paribus","figure"),
            Input("button-ceteris","n_clicks"),
            State("prof-vars","value"),
            State("idx","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,prof_vars, idx):
            exp, obs = dalex_exp(list(self.model.values())[0], self.X_train, self.Y_train, self.X_test, idx)
            _, fig = cp_profile(exp, obs, [prof_vars])
            return fig

class InteractiveComponent(ExplainerComponent):
    """
    A component class for displaying the output of the interactive function on the local explanation module to the dashboard.
    """
    def __init__(self, explainer, model, X_train, Y_train, X_test, title="Interactive", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            title (str, optional): title of the component. Defaults to "Interactive".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Interactive")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **Interactions**: The effects of an explanatory variable depends on the values of other variables. How does that affect the model response? We focus on pairwise interactions.
                                        
                                Warning: This might take a while for large datasets, large amount of features, and the size of the model
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Interactive Count: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="count",
                                type="number",
                                min=1,
                                max=len(list(self.X_train.columns)),
                                step=1,
                                value=10,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-interactive", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="interactive")), style={"margin":"auto"})
                    ]),
                    html.Br(),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("interactive","figure"),
            Input("button-interactive","n_clicks"),
            State("count","value"),
            State("idx","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,count,idx):
            exp, obs = dalex_exp(list(self.model.values())[0], self.X_train, self.Y_train, self.X_test, idx)
            _, fig = interactive(exp,obs,count)
            return fig
        
class DiceExpComponent(ExplainerComponent):
    """
    A component class for displaying the output of the exp_cf function on the local explanation module to the dashboard
    """
    def __init__(self, explainer, X_train, Y_train, X_test, model, target_feature, model_type="regressor", title="DiceExp", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            model (dict): a dict containing a single model
            target_feature (str): target feature
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "DiceExp".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.model = list(model.values())[0]
        self.target_feature = target_feature
        self.model_type = model_type
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("DICE Explanations")
                ]),
                dbc.CardBody([
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Diverse Counterfactual Explanations (DiCE) is a tool developed by Microsoft that provides counterfactual explanations for machine learning models. Counterfactual explanations are a type of explanation that can help users understand why a machine learning model made a particular prediction or decision. They do this by showing what changes would need to be made to the input features of a model in order to change its output.

                                DiCE is designed to address a common problem with counterfactual explanations: they can often provide only a single, arbitrary solution for how to change the input features. This can be limiting, as it may not give the user a full understanding of how the model is working, or how changes to the input features would impact the output.


                                To overcome this limitation, DiCE generates multiple counterfactual explanations that are diverse and meaningful. Specifically, DiCE generates a set of counterfactual explanations that satisfy two criteria:
                                - Relevance: Each counterfactual explanation should be as close as possible to the original input while still changing the model's output. In other words, the changes made to the input should be minimal, to avoid making changes that would not be realistic or practical.
                                - Diversity: Each counterfactual explanation should be different from the others in the set, in order to provide a range of possible explanations for the model's output.

                                DiCE uses an optimization algorithm to generate these counterfactual explanations. The algorithm searches for the smallest possible change to the input features that would change the model's output, subject to the constraint that each counterfactual explanation should be diverse from the others in the set.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Total Counterfactuals: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="cf",
                                type="number",
                                min=1,
                                max=20,
                                step=1,
                                value=10,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Features to Vary: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="f2v",
                                options=sorted(self.X_train.columns),
                                placeholder="Features to Vary",
                                multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Desired Range/Class: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(id="desired-dice",  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-dice", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div("Counterfactuals", style={"padding-left":"20px"}),
                        html.Div(dcc.Loading(html.Div(id="table-dice")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px'})
                    ])
                ])
            ])
        ])

    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("desired-dice", "children"),
            Input("desired-dice", "children")
        )
        def update_desired(none):
            if self.model_type == "regressor":
                return dcc.RangeSlider(1, 5000000, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id="desired", value=[1, 5000000])
            else:
                return dcc.Dropdown(id="desired",options=[0,1,"opposite"],placeholder="Desired Class in Target", value="opposite")
                
        
        @app.callback(
            Output("table-dice","children"),
            Input("button-dice","n_clicks"),
            State("idx","value"),
            State("cf","value"),
            State("f2v","value"),
            State("desired","value"), prevent_initial_call=True
        )
        def update_table(n_clicks,idx,cf,f2v,desired):
            exp = dice_exp(self.X_train, self.Y_train, self.model, target = self.target_feature, model_type = self.model_type)
            X = self.X_test[idx:idx+1]
            if self.model_type == "regressor":
                e = exp_cf(X = X, exp = exp, total_CFs = cf, features_to_vary = f2v, reg=True, desired_range=desired)
            else:
                e = exp_cf(X = X, exp = exp, total_CFs = cf, features_to_vary = f2v, desired_class=desired)
            if isinstance(e, str):
                return e, e
            else:
                df = e.cf_examples_list[0].final_cfs_df
                return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False)
    
class QIIExpComponent(ExplainerComponent):
    """
    A component class for displaying the output of the exp_cf function from the local explanation module to the dashboard.
    """
    def __init__(self, explainer, model, pipe, X_test, cat, title="QIIExp", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            X_test (pd.DataFrame): X_test
            cat (pd.DataFrame): a subset of X_train with only categorical features
            title (str, optional): title of the component. Defaults to "QIIExp".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0][-1]
        self.pipe = pipe
        self.X_test = X_test
        self.cat = cat
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4('QII Explanation')
                ]),
                dbc.CardBody([
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Quantitative Input Influence (QII) is a method for quantifying the impact of each input feature on the model's output. QII can be used to identify which input features are most important to the model's decision, and how changes to those features would impact the output.

                                QII works by computing the partial derivatives of the model's output with respect to each input feature. These derivatives indicate how sensitive the model's output is to changes in each feature. By computing the absolute values of these derivatives, QII can rank the input features in order of importance, from most to least influential.

                                Once the input features have been ranked, QII can be used to generate counterfactual explanations that show how changes to specific input features would impact the model's output. These counterfactual explanations can be used to understand the logic behind the model's decision, and to identify potential biases or errors in the model.

                                For example, if a model is being used to predict loan approvals, QII could be used to identify which input features are most important to the model's decision, such as income, credit score, and employment history. By generating counterfactual explanations that show how changes to these features would impact the model's output, users can better understand how the model is making its decisions, and identify potential biases or errors in the model. 

                                More info in https://www.andrew.cmu.edu/user/danupam/datta-sen-zick-oakland16.pdf
                                
                                Warning: The pipeline may cause this component to not work.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Method: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="method-qii",
                                options=['banzhaf', 'shapley'],
                                placeholder="QII Method",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-qii", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Div(id="table-qii")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px'})
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="graph-qii")), style={"margin":"auto"})
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("table-qii","children"),
            Output("graph-qii","figure"),
            Input("button-qii","n_clicks"),
            State("method-qii","value"),
            State("idx","value"), prevent_initial_call=True
        )
        def update_table(n_clicks, method, idx):
            df, fig = exp_qii(self.model, self.X_test, idx, self.pipe, self.cat, method = method)
            return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False), fig

class ShapWaterfallComponent(ExplainerComponent):
    """
    A component class for displaying the output of the waterfall function from the local exp module to the dashboard
    """
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, title='Shap Waterfall', name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Shap Waterfall".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.pipe = pipe
        self.cat = cat
        if model_type == 'regressor':
            self.reg = True
        elif model_type == 'classifier':
            self.reg = False
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3('Shap Waterfall Plot (Local)')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                The waterfall plot explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.
                                
                                Warning: The pipeline may cause this component to not work.
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-waterfall", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Img(id="graph-waterfall")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-waterfall","src"),
            Input("button-waterfall","n_clicks"),
            State("idx","value"), prevent_initial_call=True
        )
        def update_figure(n_clicks, idx):
            #create some matplotlib graph
            exp, shap_value_loc, feature_names = initiate_shap_loc(self.X_train,self.model['DT'][-1],self.pipe,self.cat)
            if self.reg:
                fig = shap_waterfall(shap_value_loc,idx,feature_names=feature_names,reg=self.reg,show=False,class_ind=None,class_names=None)
            else:
                fig = shap_waterfall(shap_value_loc,idx,feature_names=feature_names,reg=self.reg,show=False,class_ind=0,class_names=['1','0'])
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)

class ShapForceComponent(ExplainerComponent):
    """
    A component class for displaying the output of the shap_force_loc function of the local exp module to the dashboard
    """
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, title='Shap Force', name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Shap Force".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.pipe = pipe
        self.cat = cat
        if model_type == 'regressor':
            self.reg = True
        elif model_type == 'classifier':
            self.reg = False
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3('Shap Force Plot (Local)')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                The local force plot attempts to summarize all the individual rows found in a waterfall plot in one continuous, additive "force". As with the previous plot, features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.
                                
                                Warning: The pipeline may cause this component to not work.
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-force", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Img(id="graph-force", style={"width":"900px"})), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-force","src"),
            Input("button-force","n_clicks"),
            State("idx","value"), prevent_initial_call=True
        )
        def update_figure(n_clicks, idx):
            #create some matplotlib graph
            exp, shap_value_loc, feature_names = initiate_shap_loc(self.X_train,self.model['DT'][-1],self.pipe,self.cat)
            fig = shap_force_loc(shap_value_loc,idx,feature_names=feature_names,reg=self.reg,show=False,class_ind=0,class_names=['1','0'])
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)

class ShapBarComponent(ExplainerComponent):
    """
    A component class for displaying the output of the shap_bar_loc function of the local exp module to the dashboard
    """
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, title='Shap Bar', name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Shap Bar".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.pipe = pipe
        self.cat = cat
        if model_type == 'regressor':
            self.reg = True
        elif model_type == 'classifier':
            self.reg = False
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3('Shap Bar Plot (Local)')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                This is another interpretation of the waterfall and local force plots, brought to you in a double-sided feature importance plot.

                                Compared to the first two plots, this does not give an `f(x)` nor `E[f(x)]` indicator. The picture above showcases the same observation, noting that the variables significantly contribute to a higher likelihood of the "Rejected" class. While it does not show the actual values of each variable for that observation, the local bar plot does further emphasize the individual SHAP value contributions.
                            
                                Warning: The pipeline may cause this component to not work.
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-bar", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Img(id="graph-bar")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-bar","src"),
            Input("button-bar","n_clicks"),
            State("idx","value"), prevent_initial_call=True
        )
        def update_figure(n_clicks, idx):
            #create some matplotlib graph
            exp, shap_value_loc, feature_names = initiate_shap_loc(self.X_train,self.model['DT'][-1],self.pipe,self.cat)
            fig = shap_bar_loc(shap_value_loc,idx,feature_names=feature_names,reg=self.reg,show=False,class_ind=0,class_names=['1','0'])
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)

class PartialDependenceProfileComponent(ExplainerComponent):
    """
    A component class for displaying the output of the pd_profile from the global exp module to the dashboard
    """
    def __init__(self, explainer, exp, X_train, title="PDP", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            exp: a Dalex explainer instance that will be use for generating the pd profile
            X_train (pd.DataFrame): X_train
            title (str, optional): title of the component. Defaults to "PDP".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.exp = exp
        self.X_train = X_train
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Partial Dependence Profile")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                The general idea underlying the construction of PD profiles is to show how does the expected value of model prediction behave as a function of a selected explanatory variable. For a single model, one can construct an overall PD profile by using all observations from a dataset, or several profiles for sub-groups of the observations. Comparison of sub-group-specific profiles may provide important insight into, for instance, the stability of the model’s predictions.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Variable: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="var-gpdp",
                                options=sorted(self.X_train.columns),
                                placeholder="Select Variable",
                                #multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Group (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="grp-gpdp",
                                options=sorted(self.X_train.select_dtypes(exclude=["number"]).columns.tolist()),
                                placeholder="Select Variable",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-gpdp", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="graph-gpdp")), style={"margin":"auto"})
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        # @app.callback(
        #     Output("grp-gpdp","options"),
        #     Input("var-gpdp","value"), prevent_initial_call=True
        # )
        # def update_dropdown(grp):
        #     cols = list(self.X_train.columns)
        #     cols.remove(grp)
        #     return sorted(cols)
        
        @app.callback(
            Output("graph-gpdp","figure"),
            Input("button-gpdp","n_clicks"),
            State("grp-gpdp","value"),
            State("var-gpdp","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,grp,var):
            if var in self.exp.data.select_dtypes(exclude = np.number).columns.tolist():
                var_type="categorical"
            else:
                var_type="numerical"
            _, fig = pd_profile(self.exp,var,var_type,grp)
            return fig

class VariableImportanceComponent(ExplainerComponent):
    """
    A component class for displaying the  output of the var_imp function from the global exp module to the dashboard
    """
    def __init__(self, explainer, exp, var_grp, title="VI", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            exp: a Dalex explainer instance that will be use for generating the VI
            var_grp (dict): grouping of variables to do VI with
            title (str, optional): title of the component. Defaults to "VI".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.exp = exp
        self.var_grp = var_grp
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Variable Importances")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Variable-importance: How important is an explanatory variable? We can use this for
                                - Model simplification: excluding variables that do not influence a model's predictions
                                - Model exploration: comparing variables' importance in different models may help in discovering interrelations between variables
                                - Domain-knowledge-based model validation: identification of most important variables may be helpful in assessing the validity of the model based on domain knowledge
                                - Knowledge generation: identification of important variables may lead to discovery of new factors involved in a particular mechanism
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Grouped or Not Grouped: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="grouped-gvi",
                                options=['Grouped','Not Grouped'],
                                placeholder="Groupings Mode",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-gvi", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="graph-gvi")), style={"margin":"auto"})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-gvi","figure"),
            Input("button-gvi","n_clicks"),
            State("grouped-gvi", "value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, grouped):
            if grouped == "Grouped": 
                _, fig = var_imp(self.exp, groups=self.var_grp)
            else:
                _, fig = var_imp(self.exp)
            return fig

class LocalDependenceComponent(ExplainerComponent):
    """
    A component class for displaying the output of the ld_profile function from the global exp module to the dashboard
    """
    def __init__(self, explainer, exp, X_train, title="LD", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            exp: a Dalex explainer instance that will be use for generating the profile
            X_train (pd.DataFrame): X_train
            title (str, optional): title of the component. Defaults to "LD".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.exp = exp
        self.X_train = X_train
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Local Dependence Profile")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Creates a local-dependence plot and outputs the equivalent table. User may specify the variables to showcase.
                                Note that this can only help explain variables that are of the same data type at the same time
                                i.e. you may not analyze a numerical and categorical variable in the same run.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Variable: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="var-ld",
                                options=sorted(self.X_train.columns),
                                placeholder="Select Variable",
                                value='land_size'
                                #multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Group (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="grp-ld",
                                options=sorted(self.X_train.select_dtypes(exclude=["number"]).columns.tolist()),
                                placeholder="Select Variable",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-ld", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="graph-ld")), style={"margin":"auto"})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-ld","figure"),
            Input("button-ld","n_clicks"),
            State("var-ld","value"),
            State("grp-ld","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, var, grp):
            if var in self.exp.data.select_dtypes(exclude = np.number).columns.tolist():
                var_type="categorical"
            else:
                var_type="numerical"
            _, fig = ld_profile(self.exp,var,var_type, groups=grp)
            return fig

class AccumulatedLocalComponent(ExplainerComponent):
    """
    A component class for displaying the output of the al_profile function from the global exp module to the dashboard
    """
    def __init__(self, explainer, exp, X_train, title="AL", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            exp: a Dalex explainer instance that will be use for generating the profile
            X_train (pd.DataFrame): X_train
            title (str, optional): title of the component. Defaults to "AL".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.exp = exp
        self.X_train = X_train
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Accumulated-Local Plot")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Creates a accumulated-local plot and outputs the equivalent table. User may specify the variables to showcase.
                                Note that this can only help explain variables that are of the same data type at the same time
                                i.e. you may not analyze a numerical and categorical variable in the same run.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Variable: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="var-al",
                                options=sorted(self.X_train.columns),
                                placeholder="Select Variable",
                                value='land_size'
                                #multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Group (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="grp-al",
                                options=sorted(self.X_train.select_dtypes(exclude=["number"]).columns.tolist()),
                                placeholder="Select Variable",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-al", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="graph-al")), style={"margin":"auto"})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-al","figure"),
            Input("button-al","n_clicks"),
            State("var-al","value"),
            State("grp-al","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, var, grp):
            if var in self.exp.data.select_dtypes(exclude = np.number).columns.tolist():
                var_type="categorical"
            else:
                var_type="numerical"
            _, fig = al_profile(self.exp,var,var_type,groups=grp)
            return fig

class CompareProfileComponent(ExplainerComponent):
    """
    A component class for displaying the output of the compare_profiles function from the global exp module to the dashboard
    """
    def __init__(self, explainer, exp, X_train, title="CP", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            exp: a Dalex explainer instance that will be use for generating the profile
            X_train (pd.DataFrame): X_train
            title (str, optional): title of the component. Defaults to "CP".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.exp = exp
        self.X_train = X_train
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Comparison of Profiles")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Compares partial-dependence, local-dependence, and accumulated-local profiles. User may specify the variables to showcase.
                                Note that this can only help explain variables that are of the same data type at the same time
                                i.e. you may not analyze a numerical and categorical variable in the same run.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Variable: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="var-cp",
                                options=sorted(self.X_train.columns),
                                placeholder="Select Variable",
                                value='land_size'
                                #multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Group (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="grp-cp",
                                options=sorted(self.X_train.select_dtypes(exclude=["number"]).columns.tolist()),
                                placeholder="Select Variable",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-cp", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(dcc.Graph(id="graph-cp")), style={"margin":"auto"})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-cp","figure"),
            Input("button-cp","n_clicks"),
            State("var-cp","value"),
            State("grp-cp","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, var, grp):
            if var in self.exp.data.select_dtypes(exclude = np.number).columns.tolist():
                var_type="categorical"
            else:
                var_type="numerical"
            fig = compare_profiles(self.exp,var,var_type,groups=grp)
            return fig

class ShapBarGlobalComponent(ExplainerComponent):
    """
    A component class for displaying the output of the shap_bar_glob function from the global exp module to the dashboard
    """
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, title='Shap Bar Global', name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Shap Bar Global".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.pipe = pipe
        self.cat = cat
        if model_type == 'regressor':
            self.reg = True
        elif model_type == 'classifier':
            self.reg = False
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3('Shap Bar Plot (Global)')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **Bar Plot**: This takes the average of the SHAP value magnitudes across the dataset and plots it as a simple bar chart.
                            
                                Warning: The pipeline may cause this component to not work.
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-bar-global", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Img(id="graph-bar-global")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-bar-global","src"),
            Input("button-bar-global","n_clicks"), prevent_initial_call=True
        )
        def update_figure(n_clicks):
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,self.model['DT'][-1],self.pipe,self.cat)
            X_train_proc = self.pipe.transform(self.X_train)
            fig = shap_bar_glob(shap_value_glob,X_proc=X_train_proc,feature_names=feature_names,reg=self.reg,class_ind=0,class_names=['1','0'])
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)

class ShapSummaryComponent(ExplainerComponent):
    """
    A component class for displaying the output of the shap_summary function from the global exp module to the dashboard
    """
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, title='Shap Summary', name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Shap Summary".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.pipe = pipe
        self.cat = cat
        if model_type == 'regressor':
            self.reg = True
        elif model_type == 'classifier':
            self.reg = False
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3('Shap Summary Plot (Global)')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **SHAP Summary Plot**: Rather than use a typical feature importance bar chart, we use a density scatter plot of SHAP values for each feature to identify how much impact each feature has on the model output for individuals in the validation dataset. Features are sorted by the sum of the SHAP value magnitudes across all samples. It is interesting to note that the relationship feature has more total model impact than the captial gain feature, but for those samples where capital gain matters it has more impact than age. In other words, capital gain effects a few predictions by a large amount, while age effects all predictions by a smaller amount.
                            
                                Warning: The pipeline may cause this component to not work.
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-summary-global", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Img(id="graph-summary-global")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-summary-global","src"),
            Input("button-summary-global","n_clicks"), prevent_initial_call=True
        )
        def update_figure(n_clicks):
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,self.model['DT'][-1],self.pipe,self.cat)
            X_train_proc = self.pipe.transform(self.X_train)
            fig = shap_summary(shap_value_glob,X_proc=X_train_proc,feature_names=feature_names,reg=self.reg,class_ind=0,class_names=['1','0'])
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)

class ShapDependenceComponent(ExplainerComponent):
    """
    A component class for displaying the output of the shap_dependence function from the global exp module to the dashboard
    """
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, title='Shap Dependence', name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Shap Dependence".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.pipe = pipe
        self.cat = cat
        if model_type == 'regressor':
            self.reg = True
        elif model_type == 'classifier':
            self.reg = False
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3('Shap Dependence Plot (Global)')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **Dependence Plot:** We can also run a plot for SHAP interaction values to observe its main effects and interaction effects with other variables. We can look at it in two ways: 1) by comparing the original variable to its SHAP values, and 2) by directly looking at another variable. Note that we may have to specify the class of the target variable if we are working with a classification model, as seen below.
                            
                                Warning: The pipeline may cause this component to not work.
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Variable: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="summary-var",
                                options=self.X_train.select_dtypes(include=['number']).columns.tolist(),
                                placeholder="Select Variable to Analyze",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Select Another Variable (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="summary-var-optional",
                                options=self.X_train.select_dtypes(include=['number']).columns.tolist(),
                                placeholder="Select Variable to Analyze",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-dependence-global", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Img(id="graph-dependence-global")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-dependence-global","src"),
            Input("button-dependence-global","n_clicks"),
            State("summary-var","value"),
            State("summary-var-optional","value"), prevent_initial_call=True
        )
        def update_figure(n_clicks,shap_ind,int_ind):
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,self.model['DT'][-1],self.pipe,self.cat)
            X_train_proc = self.pipe.transform(self.X_train)
            fig = shap_dependence(shap_value_glob,X_proc=X_train_proc,feature_names=feature_names,reg=self.reg,class_ind=0,class_names=['1','0'],shap_ind=shap_ind,int_ind=int_ind)
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)

class ShapForceGlobalComponent(ExplainerComponent):
    """
    A component class for displaying the output of the shap_force_glob from the global exp module to the dashboard
    """
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, title='Shap Force Global', name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Shap Force Global".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train = X_train
        self.pipe = pipe
        self.cat = cat
        if model_type == 'regressor':
            self.reg = True
        elif model_type == 'classifier':
            self.reg = False
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H3('Shap Force Plot (Global)')
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                **Force Plot**: Another way to visualize the same explanation is to use a force plot. If we take many local force plot explanations, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset.
                            
                                Warning: The pipeline may cause this component to not work.
                            '''),style={"padding":"30px"}) 
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-force-global", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(html.Img(id="graph-force-global")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ])
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("graph-force-global","src"),
            Input("button-force-global","n_clicks"), prevent_initial_call=True
        )
        def update_figure(n_clicks):
            #create some matplotlib graph
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,self.model['DT'][-1],self.pipe,self.cat)
            X_train_proc = self.pipe.transform(self.X_train)
            fig = shap_force_glob(exp,shap_value_glob,X_proc=X_train_proc,feature_names=feature_names,reg=self.reg,class_ind=0,class_names=['1','0'])
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)

class PSIComponent(ExplainerComponent):
    """
    A component class for displaying the psi_list function from the stability module to the dashboard
    """
    def __init__(self, explainer, X_train, X_test, cont, title="PSI", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            X_test (pd.DataFrame): X_test
            cont (pd.DataFrame): a subset of X_train with only continuous features
            title (str, optional): title of the component. Defaults to "PSI".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.X_train = X_train
        self.X_test = X_test
        self.cont = cont
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Population Stability Index")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Population Stability Index (PSI) compares the distribution of the target variable in the test dataset to a training data set that was used to develop the model.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Filter: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="filter-psi",
                                options=['No Shift','Large Shift','Slight Shift'],
                                placeholder="Filter",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-psi", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                       html.Div(html.Pre(id="text-psi"), style={"padding-left":"30px"}) 
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Div(id="table-psi")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px'})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("table-psi","children"),
            Output("text-psi","children"),
            Input("button-psi","n_clicks"),
            State("filter-psi","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, filter_df):
            df = generate_psi_df(self.X_train[self.cont], self.X_test[self.cont])
            if filter_df == None:
                df = df
            else:
                df = df[df.Shift == filter_df]
            if df[df.Shift == "Large Shift"].empty and df[df.Shift == "Slight Shift"].empty:
                text = "There is no change or shift in the distributions of both datasets for all columns."
            elif not(df[df.Shift == "Large Shift"].empty):
                text = "There is/are indications that a large shift has occurred between both datasets. Filter to Large Shift to see which columns."
            elif not(df[df.Shift == "Slight Shift"].empty):
                text = "There is/are indications that a slight shift has occurred between both datasets. Filter to Small Shift to see which columns."
            else:
                text = None
            return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False), text

# class PageHinkleyComponent(ExplainerComponent):
#     """
#     A component class for displaying the output of the PageHinkley function from the stability module to the dashboard
#     """
#     def __init__(self, explainer, X_train, X_test, cont, title="PH", name=None):
#         """
#         Args:
#             explainer: a dummy explainer, won't really be utilized to extract info from
#             model (dict): a dict containing a single model
#             X_train (pd.DataFrame): X_train
#             X_test (pd.DataFrame): X_test
#             cont (pd.DataFrame): a subset of X_train with only continuous features
#             title (str, optional): title of the component. Defaults to "PH".
#             name (optional): name of the component. Defaults to None.
#         """
#         super().__init__(explainer, title=title)
#         self.X_train = X_train
#         self.X_test = X_test
#         self.cont = cont
        
#     def layout(self):
#         return dbc.Container([
#             dbc.Card([
#                 dbc.CardHeader([
#                     html.H4("Page Hinkley")
#                 ]),
#                 dbc.CardBody([
#                     dbc.Row([
#                         dbc.Col([
#                             html.Div(dcc.Markdown('''
#                                 This change detection method works by computing the observed values and their mean up to the current moment. Page-Hinkley does not signal warning zones, only change detections. This detector implements the CUSUM control chart for detecting changes. This implementation also supports the two-sided Page-Hinkley test to detect increasing and decreasing changes in the mean of the input values.
#                             '''), style={"padding":"30px"})
#                         ])
#                     ]),
#                     dbc.Row([
#                         dbc.Col([
#                             html.P("Filter: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
#                         ], width="auto"),
#                         dbc.Col([
#                             html.Div(dcc.Dropdown(
#                                 id="filter-ph",
#                                 options=self.X_test[self.cont].columns.tolist(),
#                                 placeholder="Filter",
#                             ), style={"width":"100%", "padding-right":"30px"})
#                         ]),
#                     ]),
#                     html.Br(),
#                     dbc.Row([
#                         dbc.Col([
#                             html.Div()
#                         ]),
#                         dbc.Col([
#                             html.Div(dbc.Button("Compute", id="button-ph", n_clicks=0), style={"margin":"auto"})
#                         ], width="auto"),
#                         dbc.Col([
#                             html.Div()
#                         ])
#                     ]),
#                     html.Br(),
#                     dbc.Row([
#                        html.Div(html.Pre(id="text-ph"), style={"padding-left":"30px"}) 
#                     ]),
#                     html.Br(),
#                     dbc.Row([
#                         html.Div(dcc.Loading(html.Div(id="table-ph")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px'})
#                     ]),
#                 ])
#             ])
#         ])
    
#     def component_callbacks(self,app,**kwargs):
#         @app.callback(
#             Output("table-ph","children"),
#             Output("text-ph","children"),
#             Input("button-ph","n_clicks"),
#             State("filter-ph", "value"), prevent_initial_call=True
#         )
#         def update_graph(n_clicks, filter_df):
#             df = PageHinkley(self.X_train[self.cont], self.X_test[self.cont])
#             if filter_df != None:
#                 df = df[df.column==filter_df]
#             if df.empty and filter_df != None:
#                 text = "There is no data drift detected for the selected columns."
#             elif df.empty:
#                 text = "There is no data drift detected for all the columns."
#             else:
#                 text = "There is a data drift detected for the following index and columns with their corresponding values."
#             return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False), text

class KSTestComponent(ExplainerComponent):
    """
    A component class for displaying the output of the ks function in the stability module to the dashboard
    """
    def __init__(self, explainer, X_train, X_test, pipe, title="KS", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            X_test (pd.DataFrame): X_test
            pipe (sklearn.Pipeline): the preprocessing pipeline for the model
            title (str, optional): title of the component. Defaults to "KS".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.X_train = pd.DataFrame(pipe.transform(X_train))
        self.X_test = pd.DataFrame(pipe.transform(X_test))
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Kolgomorov-Smirnov (K-S) Test")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                The K-S test is a nonparametric test that compares the cumulative distributions of two data sets, in this case, the training data and the post-training data. The null hypothesis for this test states that the data distributions from both the datasets are same. If the null is rejected then we can conclude that there is a drift in the data.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("P-value: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='ks-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Filter: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="filter-ks",
                                options=['Filter with Indicated P-value'],
                                placeholder="Filter",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-ks", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                       html.Div(html.Pre(id="text-ks"), style={"padding-left":"30px"}) 
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Div(id="table-ks")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px'})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("table-ks","children"),
            Output("text-ks","children"),
            Input("button-ks","n_clicks"),
            State("ks-slider","value"),
            State("filter-ks","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, p_value, filter_df):
            df, rejected_cols = ks(self.X_train, self.X_test, p_value)
            text_list = []
            for col in rejected_cols:
                p_value = df[df["columns"]==col]["p_values"].values[0]
                text_list.append(f"{col} column is rejected, p-value at {p_value} \n")
            if filter_df == None:
                df = df
            else:
                df = df[df.p_values < p_value]
            return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False), text_list

class FairnessTab(ExplainerComponent):
    """
    A tab class for displaying the entirety of the fairness module on a single tab in the dashboard
    """
    def __init__(self, explainer, model, X_test, Y_test, X_train, Y_train, test_data, train_data, target_feature, model_type, title="Performance & Fairness", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            test_data (pd.DataFrame): concatenated version of X_test and Y_test
            train_data (pd.DataFrame): concatenated version of X_train and Y_train
            target_feature (str): target feature
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Performance & Fairness".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.overview = FairnessIntroComponent(explainer)
        if model_type == "regressor":
            self.fairness_check = FairnessCheckRegComponent(explainer,model,X_test,Y_test)
            self.model_performance = ModelPerformanceRegComponent(explainer,model,X_test,Y_test,X_train,Y_train,test_data,train_data,target_feature)
        elif model_type == "classifier":
            self.fairness_check = FairnessCheckClfComponent(explainer,model,X_test,Y_test)
            self.model_performance = ModelPerformanceClfComponent(explainer,model,X_test,Y_test,X_train,Y_train,test_data,train_data,target_feature)
            
            
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
               self.overview.layout() 
            ]),
            html.Br(),
            dbc.Row([
                self.fairness_check.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.model_performance.layout()
            ])
        ])

class LocalExpTab(ExplainerComponent):
    """
    A tab class for displaying the entirety of the local exp module to a single tab in the dashboard
    """
    def __init__(self, explainer, model, X_train, Y_train, X_test, cont, cat, model_type, target_feature, pipe=None, title="Local Explanations", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            cont (pd.DataFrame): a subset of the X_train containing only the continuous variables
            cat (pd.DataFrame): a subset of the X_train containing only the categorical variables
            model_type (str): classifier or regressor
            target_feature (str): target feature
            title (str, optional): title of the component. Defaults to "Local Explanations".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.idx = EntryIndexComponent(explainer,X_test)
        self.breakdown = BreakDownComponent(explainer,model,X_train,Y_train,X_test)
        self.interactive = InteractiveComponent(explainer,model,X_train,Y_train,X_test)
        self.additive = AdditiveComponent(explainer,model,X_train,Y_train,X_test)
        self.ceterisparibus = CeterisParibusComponent(explainer,model,X_train,Y_train,X_test,cont)
        self.dice = DiceExpComponent(explainer, X_train, Y_train, X_test, model, target_feature, model_type)
        if pipe == None:
            self.qii = BlankComponent(explainer)
        else:
            self.qii = QIIExpComponent(explainer, model, pipe, X_test, cat)
        self.waterfall = ShapWaterfallComponent(explainer, model, X_train, pipe, cat, model_type)
        self.force = ShapForceComponent(explainer, model, X_train, pipe, cat, model_type)
        self.bar = ShapBarComponent(explainer, model, X_train, pipe, cat, model_type)
        
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
                self.idx.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.dice.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.qii.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.breakdown.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.additive.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.ceterisparibus.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.interactive.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.waterfall.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.force.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.bar.layout()
            ]),
        ])

class GlobalExpTab(ExplainerComponent):
    """
    A tab class for displaying the entirety of the global exp module in a single tab in the dashboard
    """
    def __init__(self, explainer, exp, model, X_train, pipe, cat, model_type, var_grp, title="Global Explanations", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            exp: a Dalex explainer instance that will be use for generating the profiles
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of the X_train containing only the categorical variables
            model_type (str): classifier or regressor
            var_grp (dict): the groupings for the VI component
            title (str, optional): title of the component. Defaults to "Local Explanations".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.pdp = PartialDependenceProfileComponent(explainer,exp,X_train)
        self.vi = VariableImportanceComponent(explainer,exp,var_grp)
        self.ld = LocalDependenceComponent(explainer,exp,X_train)
        self.al = AccumulatedLocalComponent(explainer,exp,X_train)
        self.cp = CompareProfileComponent(explainer,exp,X_train)
        self.bar = ShapBarGlobalComponent(explainer,model,X_train,pipe,cat,model_type)
        self.summary = ShapSummaryComponent(explainer,model,X_train,pipe,cat,model_type)
        self.dependence = ShapDependenceComponent(explainer,model,X_train,pipe,cat,model_type)
        self.force = ShapForceGlobalComponent(explainer,model,X_train,pipe,cat,model_type)
        
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
                self.pdp.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.vi.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.ld.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.al.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.cp.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.bar.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.summary.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.dependence.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.force.layout()
            ])
        ])

class StabilityTab(ExplainerComponent):
    """
    A tab class for displaying the entirety of the stability module to a single tab in the dashboard
    """
    def __init__(self, explainer, X_train, X_test, cont, pipe, title="Stability", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            X_train (pd.DataFrame): X_train
            X_test (pd.DataFrame): X_test
            cont (pd.DataFrame): a subset of the X_train containing only the continuous variables
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            title (str, optional): title of the component. Defaults to "Local Explanations".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.psi = PSIComponent(explainer,X_train,X_test,cont)
        #self.ph = PageHinkleyComponent(explainer,X_train, X_test, cont)
        self.ks = KSTestComponent(explainer,X_train, X_test,pipe)
        
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
                self.psi.layout()
            ]),
            # html.Br(),
            # dbc.Row([
            #     self.ph.layout()
            # ]),
            html.Br(),
            dbc.Row([
                self.ks.layout()
            ]),
        ])
