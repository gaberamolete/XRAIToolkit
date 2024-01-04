from copy import deepcopy

# Performance Overview and Fairness
from .fairness import model_performance
from .fairness import fairness
from .fairness_algorithm import *

# Outliers
from .outlier import *

# Dashboard
from explainerdashboard import *
from explainerdashboard.custom import *
import dash as html
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, dash_table, State
import dash_mantine_components as dmc

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
                        ### Classification
                        Some terms and metrics we will be using:
                        - **Statistical parity difference:** Statistical parity difference measures the difference that the majority and protected classes receive a favorable outcome. This measure must be equal to 0  to be fair.
                        - **Equal opportunity difference:** This measures the deviation from the equality of opportunity, which means that the same proportion of each population receives the favorable outcome. This  measure must be equal to 0 to be fair.
                        - **Average absolute odds difference:** This measures bias by using the false positive rate and true positive rate. This measure must be equal to 0 to be fair.
                        - **Disparity impact:** This compares the proportion of individuals that receive a favorable outcome for two groups, a majority group and a minority group. This measure must be equal to 1 to be fair.
                        - **Theil Index:** This ranges between zero and ∞, with zero representing an equal distribution and higher values representing a higher level of inequality.
                        - **Smoothed Emperical Differential Fairness:** This calculates the differential in the probability of favorable and unfavorable outcomes between intersecting groups divided by features. All intersecting groups are equal, so there are no unprivileged or privileged groups. The calculation produces a value between 0 and 1 that is the minimum ratio of Dirichlet  smoothed probability for favorable and unfavorable outcomes between intersecting groups in the dataset.
                        - **Class Imbalance:** Bias occurs when a facet value has fewer training samples when compared with another facet in the dataset. CI values near either of the extremes values of -1 or 1 are very imbalanced and are at a substantial risk of making biased predictions. (REMOVED)
                        - **Threshold:** Threshold defines how far from the ideal value of the metric will be acceptable. The question is what threshold should we use? There is actually no good answer to that. It will depend on your industry and application. If your model has significant consequences, like for mortgage applications, you will need a stricter threshold. The threshold may even be defined by law. Either way, it is important to define the thresholds before you measure fairness. 0.2 seems to be a good default value for that. 

                        ### Regression
                        Given $R$ as the model's prediction, $Y$ as the model's target, and $A$ to be the protected group, we have three criteria:
                        - **Independence:** $R$ ⊥ $A$
                        - **Separation:** $R$ ⊥ $A$ ∣ $Y$
                        - **Sufficiency:** $Y$ ⊥ $A$ ∣ $R$ 

                        In the approach described in Steinberg, D., et al. (2020), the authors propose a way of checking this independence. ***More info about metrics of regression***: https://arxiv.org/pdf/2001.06089.pdf
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
    A component class for displaying the output of the functions in fairness algorithm on the dashboard for classification cases.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Fairness", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            train_data (pd.DataFrame): X_train and Y_train
            test_data (pd.DataFrame): X_test and Y_test
            target_feature (str): target feature to predict
            title (str, optional): title of the component. Defaults to "Fairness".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0][-1]
        self.train_data = train_data
        self.test_data = test_data
        self.target_feature = target_feature
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Fairness")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                This component evaluates the fairness of a classification datasets returning its performance in various fairness metrics covered in the fairness overview. Algorithms are also utilized in order to mitigate any biases found. The performance for all algorithms is mapped in the `All` option for a selected fairness metrics plotted against the balanced accuracy.
                            '''), style={"padding":"30px"})
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
                                options=sorted([i for i in self.train_data.columns.tolist() if i != self.target_feature]),
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
                                options=["Balanced accuracy", "Statistical parity difference", "Disparate impact", "Average absolute odds difference", "Equal opportunity difference", "Theil index", "Smooth EDF", "Class Imbalance"],
                                placeholder="Select the Metric",
                            ),  style={"padding-left":"30px","width":"100%"})
                        ]),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="threshold",
                                type="number",
                                min=0,
                                max=1,
                                step=0.1,
                                value=0.2,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Select Algorithm (Optional):", style={"padding-left":"30px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div([dcc.Dropdown(
                                id="algo",
                                options=["Disparate Impact Remover", "Reweighing", "Exponentiated Gradient Reduction", "Calibrated Equalized Odds", "Reject Option", "All"], # "Meta Classifier"
                                placeholder="Select the Algorithm",
                            ),
                            dbc.Tooltip(
                                id="fairness-drp-tooltip",
                                target='algo')], style={"padding-left":"30px","width":"100%"})
                        ])
                    ]),
                    dbc.Row([
                        html.Div(html.P(id="text-fairness"),style={"padding":"50px"})
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-fa"), style={"margin":"auto"})
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
                        html.Div(dcc.Loading(dcc.Graph(id="graph-fa")), style={"margin":"auto", "width":"1000px"})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("protected-group-value","children"),
            Input("protected-group-feature","value")
        )
        def update_dropdown(feature):
            return dcc.Dropdown(options=sorted(self.train_data[feature].unique()), placeholder="Select the Protected Group Value", multi=True, id="value-frchk")
        
        @app.callback(
            Output("fairness-drp-tooltip","children"),
            Input("algo","value")
        )
        def update_definition(algo):
            return algo_exp(algo)
        
        @app.callback(
            Output("graph-fa","figure"),
            Output("text","children"),
            Input("button-fa","n_clicks"),
            State("protected-group-feature","value"),
            State("value-frchk","value"),
            State("metrics", "value"),
            State("threshold", "value"),
            State("algo", "value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, protected_grp, protected_val, metrics, threshold, algo):
            train_data = self.train_data.copy()
            test_data = self.test_data.copy()
            sd_train = StandardDataset(train_data,self.target_feature,[1.0],[protected_grp],[protected_val])
            sd_test = StandardDataset(test_data,self.target_feature,[1.0],[protected_grp],[protected_val])
            p = []
            u = []
            for i, j in zip([protected_grp],[protected_val]):
                p.append({i: j})
                u.append({i: [x for x in train_data[i].unique().tolist() if x not in j and not(np.isnan(x))]})
            if algo == None:
                sd_test_pred = sd_test.copy()
                model = deepcopy(self.model)
                sd_test_pred.labels = self.model.predict(sd_test.features)
                before = compute_metrics(sd_test, sd_test_pred, u, p)
                fig = metrics_plot(metrics1=before,threshold=threshold,metric_name=metrics,protected=protected_grp)
                if metrics == "Disparate impact":
                    text = f"Bias detected in {protected_grp}: {protected_val} for {metrics} metric" if before[metrics] < 1-threshold or before[metrics] > 1+threshold else f"No bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                else:
                    text = f"Bias detected in {protected_grp}: {protected_val} for {metrics} metric" if before[metrics] < -threshold or before[metrics] > threshold else f"No bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                return fig, text
            elif algo == "Disparate Impact Remover":
                model = deepcopy(self.model)
                train_data = self.train_data.copy()
                test_data = self.test_data.copy()
                _, _, before, after = disparate_impact_remover(model,train_data,test_data,self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                fig = metrics_plot(metrics1=before, metrics2=after, threshold=threshold, metric_name=metrics, protected=protected_grp)
                if metrics == "Disparate impact":
                    text = f"After {algo}, bias is detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < 1-threshold or after[metrics] > 1+threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                else:
                    text = f"After {algo}, bias detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < -threshold or after[metrics] > threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                return fig, text
            elif algo == "Reweighing":
                model = deepcopy(self.model)
                _, before, after = reweighing(model,self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                fig = metrics_plot(metrics1=before, metrics2=after, threshold=threshold, metric_name=metrics, protected=protected_grp)
                if metrics == "Disparate impact":
                    text = f"After {algo}, bias is detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < 1-threshold or after[metrics] > 1+threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                else:
                    text = f"After {algo}, bias detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < -threshold or after[metrics] > threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                return fig, text
            elif algo == "Exponentiated Gradient Reduction":
                model = deepcopy(self.model)
                before, after = exponentiated_gradient_reduction(model,self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                fig = metrics_plot(metrics1=before, metrics2=after, threshold=threshold, metric_name=metrics, protected=protected_grp)
                if metrics == "Disparate impact":
                    text = f"After {algo}, bias is detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < 1-threshold or after[metrics] > 1+threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                else:
                    text = f"After {algo}, bias detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < -threshold or after[metrics] > threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                return fig, text
#             elif algo == "Meta Classifier":
#                 model = deepcopy(self.model)
#                 before, after = meta_classifier(model,self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
#                 fig = metrics_plot(metrics1=before, metrics2=after, threshold=threshold, metric_name=metrics, protected=protected_grp)
#                 if metrics == "Disparate impact":
#                     text = f"After {algo}, bias is detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < 1-threshold or after[metrics] > 1+threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
#                 else:
#                     text = f"After {algo}, bias detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < -threshold or after[metrics] > threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
#                 return fig, text
            elif algo == "Calibrated Equalized Odds":
                model = deepcopy(self.model)
                before, after = calibrated_eqodds(model,self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                fig = metrics_plot(metrics1=before, metrics2=after, threshold=threshold, metric_name=metrics, protected=protected_grp)
                if metrics == "Disparate impact":
                    text = f"After {algo}, bias is detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < 1-threshold or after[metrics] > 1+threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                else:
                    text = f"After {algo}, bias detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < -threshold or after[metrics] > threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                return fig, text
            elif algo == "Reject Option":
                model = deepcopy(self.model)
                before, after = reject_option(model,self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                fig = metrics_plot(metrics1=before, metrics2=after, threshold=threshold, metric_name=metrics, protected=protected_grp)
                if metrics == "Disparate impact":
                    text = f"After {algo}, bias is detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < 1-threshold or after[metrics] > 1+threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                else:
                    text = f"After {algo}, bias detected in {protected_grp}: {protected_val} for {metrics} metric" if after[metrics] < -threshold or after[metrics] > threshold else f"After {algo}, no bias detected in {protected_grp}: {protected_val} for {metrics} metric"
                return fig, text
            else:
                _, _, before1, after1 = disparate_impact_remover(deepcopy(self.model),self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                _, before2, after2 = reweighing(deepcopy(self.model),self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                before3, after3 = exponentiated_gradient_reduction(deepcopy(self.model),self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                before4, after4 = meta_classifier(deepcopy(self.model),self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                before5, after5 = calibrated_eqodds(deepcopy(self.model),self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                before6, after6 = reject_option(deepcopy(self.model),self.train_data.copy(),self.test_data.copy(),self.target_feature,protected=[protected_grp],privileged_classes=[protected_val])
                fig = compare_algorithms(before1, after1, after2, after3, after4, after5, after6,float(threshold), metrics)
                text = ""
                return fig, text
            
class ModelPerformanceRegComponent(ExplainerComponent):
    """
    A component class for displaying the output of the model_performance function in the fairness module on the dashboard for
    regression cases.
    """
    def __init__(self, explainer, model, X_test, Y_test, X_train, Y_train, test_data, train_data, target_feature, title="Fairness", name=None):
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
                                    html.Div(dcc.Loading(html.Div(id="table")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px', 'width':'auto'})
                                ]),
                                dbc.Col([
                                    html.Div(id='no-return')
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
            Output("no-return","children"),
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
    def __init__(self, explainer, model, X_test, Y_test, X_train, Y_train, test_data, train_data, target_feature, title="Fairness", name=None):
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
                                    html.Div(dcc.Loading(html.Div(id="table-clf")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px', 'width':'auto'})
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
            
class OutlierComponent(ExplainerComponent):
    """
    A dashboard component class for displaying the output of the outlier.py methods to the dashboard.
    """
    def __init__(self, explainer, train_data, test_data, title="Outlier", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            train_data (pd.DataFrame): concatenated version of X_train and Y_train
            test_data (pd.DataFrame): concatenated version of X_test and Y_test
            title (str, optional): title of the component. Defaults to "Performance".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.train_data = train_data
        self.test_data = test_data
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Outliers")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Types of Outlier Detection methods:
                                - **ABOD: Angle-based Outlier Detector** | ABOD performs well on multi-dimensional data. For an observation, the variance of its weighted cosine scores to all neighbors could be viewed as the outlying score.    
                                - **IForest: Isolation Forest Outlier Detector** | Isolation Forest performs well on multi-dimensional data. The IsolationForest 'isolates' observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. 
                                - **KPCA: Kernel Principal Component Analysis (KPCA) Outlier Detector** | PCA is performed on the feature space uniquely determined by the kernel, and the reconstruction  error on the feature space is used as the anomaly score.
                                - **PCA: Principal Component Analysis (PCA) Outlier Detector** | Principal component analysis (PCA) can be used in detecting outliers. PCA is a linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. In this procedure, covariance matrix of the data can be decomposed to orthogonal vectors, called eigenvectors, associated with eigenvalues. The eigenvectors with high eigenvalues capture most of the variance in the data. Therefore, a low dimensional hyperplane constructed by k eigenvectors can capture most of the variance in the data. However, outliers are different from normal data points, which is more obvious on the hyperplane constructed by the eigenvectors with small eigenvalues. Therefore, outlier scores can be obtained as the sum of the projected distance of a sample on all eigenvectors. 
                                - **AnoGAN: Anomaly Detection with Generative Adversarial Networks** | A deep convolutional generative adversarial network to learn a manifold of normal anatomical variability, accompanying a novel anomaly scoring scheme based on the mapping from image space to a latent space.
                                - **KNN: k-Nearest Neighbors Detector** | For an observation, its distance to its k-th nearest neighbor could be viewed as the outlying score. It could be viewed as a way to measure the density.kNN class for outlier detection. For an observation, its distance to its kth nearest neighbor could be viewed as the outlying score. It could be viewed as a way to measure the density.
                                - **CBLOF: Clustering Based Local Outlier Factor** | CBLOF takes as an input the data set and the cluster model that was generated by a clustering algorithm. It classifies the clusters into small clusters and large clusters using the parameters alpha and beta. The anomaly score is then calculated based on the size of the cluster the point belongs to as well as the distance to the nearest large cluster.
                                - **ALAD: Adversarially Learned Anomaly Detection** | Adversarially Learned Anomaly Detection (ALAD) based on bi-directional GANs, that derives adversarially learned features for the anomaly detection task.
                                - **ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions (ECOD)** | ECOD is a parameter-free, highly interpretable outlier detection algorithm based on empirical CDF functions.

                                More information on the methods can be found here: https://pyod.readthedocs.io/en/latest/install.html
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Methods: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div([dcc.Dropdown(
                                id="method-outlier",
                                options=["ABOD","CBLOF","ALAD","ECOD","IForest","AnoGAN","KNN","KPCA","XGBOD","PCA"],
                                placeholder="Choose a Statistics Test",
                            ),
                            dbc.Tooltip(
                                id="outlier-tooltip",
                                target='method-outlier')], style={"padding-left":"30px","width":"100%"})
                        ]),
                        dbc.Col([
                            html.Div("Contamination: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='outlier-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                         dbc.Col([
                            html.P("Features: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dmc.MultiSelect(
                                id="feature-outlier",
                                data=self.train_data.columns.tolist(),
                                placeholder="Select 2",
                                maxSelectedValues=2,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-outlier"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.H6("Train Data: ", style={"padding":"50px"})
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(dcc.Graph(id="graph-train-outlier")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.H6("Test Data: ", style={"padding":"50px"})
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(dcc.Graph(id="graph-test-outlier")), style={"margin":"auto"})
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
            Output("outlier-tooltip","children"),
            Input("method-outlier","value")
        )
        def update_definition(method):
            return method_exp(method)
        
        @app.callback(
            Output("graph-train-outlier","figure"),
            Output("graph-test-outlier","figure"),
            Input("button-outlier","n_clicks"),
            State("method-outlier","value"),
            State("outlier-slider","value"),
            State("feature-outlier","value"), prevent_initial_call=True
        )
        def update_graphs(n_clicks,method,contamination,feature):
            label_train, label_test = outlier(self.train_data, self.test_data, methods=[method], contamination=contamination)
            fig1 = visualize(self.train_data, label_train, show=feature)
            fig2 = visualize(self.test_data, label_test, show=feature)
            return fig1[0], fig2[0]

class ErrorAnalysisComponent(ExplainerComponent):
    """
    A dashboard component class for linking the dashboard from the Responsible AI library for error analysis to XRAI dashboard.
    """
    def __init__(self, explainer, title="Error Analysis", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            title (str, optional): title of the component. Defaults to "Performance".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Error Analysis")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                    Click the button to go to the aforementioned dashboard. It will lead you to another page.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Click Here", id="link-centered-error", className="ml-auto", href='http://localhost:5000'), style={'margin':'auto'})
                        ], width='auto'),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                ])
            ])
        ])

class FairnessTab(ExplainerComponent):
    """
    A tab class for displaying all the components of the fairness module to a singular tab in dashboard.
    """
    def __init__(self, explainer, model, X_test, Y_test, X_train, Y_train, test_data, train_data, test_data_proc, train_data_proc, target_feature, model_type, title="Performance & Fairness", name=None):
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
            test_data_proc (pd.DataFrame): concatenated version of preprocessed X_test and Y_test
            train_data_proc (pd.DataFrame): concatenated version of preprocessed X_train and Y_train
            target_feature (str): target feature
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Performance".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.overview = FairnessIntroComponent(explainer)
        if model_type == "regressor":
            self.fairness_check = FairnessCheckRegComponent(explainer,model,X_test,Y_test)
            self.model_performance = ModelPerformanceRegComponent(explainer,model,X_test,Y_test,X_train,Y_train,test_data,train_data,target_feature)
        elif model_type == "classifier":
            self.fairness_check = FairnessCheckClfComponent(explainer, model, train_data_proc, test_data_proc, target_feature)
            self.model_performance = ModelPerformanceClfComponent(explainer,model,X_test,Y_test,X_train,Y_train,test_data,train_data,target_feature)
        self.outlier = OutlierComponent(explainer,train_data,test_data) 
        self.error = ErrorAnalysisComponent(explainer)
            
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
            ]),
            html.Br(),
            dbc.Row([
                self.outlier.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.error.layout()
            ])
        ])