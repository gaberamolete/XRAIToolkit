# ART
from .art_extra_models import *
from .art_metrics import *
from .art_mia import *

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
    
class ARTPrivacyComponent(ExplainerComponent):
    """
    A component class for displaying the results on the ART privacy metrics to the dashboard.
    """
    def __init__(self, explainer, model, X_train_proc, Y_train, X_test_proc, Y_test, title="Privacy Metric", name=None):
        """
        Args:
            explainer: explainer instance from explainerdashboard
            model (dict): a container for the model
            X_train_proc (pd.DataFrame): a preprocessed X_train
            Y_train (pd.DataFrame): Y_train
            X_test_proc (pd.DataFrame): a preprocess X_test
            Y_test (pd.DataFrame): Y_test
            title (str, optional): title of the component. Defaults to "Privacy Metric".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train_proc = X_train_proc
        self.Y_train = Y_train
        self.X_test_proc = X_test_proc
        self.Y_test = Y_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("PDTP and SHAPr Privacy Metric")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("PDTP Number of Samples: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(1, 100, step=1, value=10, id='pdtp-nos-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.Div("PDTP Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 100, value=0, id='pdtp-t-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div("SHAPr Number of Samples: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(1, 100, step=1, value=10, id='shapr-nos-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.Div("SHAPr Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 100, value=0, id='shapr-t-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-pm"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.H6("PDTP:"),
                        dcc.Markdown(id="text-pdtp")
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(dcc.Graph(id="graph-pdtp")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.H6("SHAPr:"),
                        dcc.Markdown(id="text-shapr")
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(dcc.Graph(id="graph-shapr")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("text-pdtp","children"),
            Output("graph-pdtp","figure"),
            Output("text-shapr","children"),
            Output("graph-shapr","figure"),
            Input("button-pm","n_clicks"), 
            State("pdtp-nos-slider","value"),
            State("pdtp-t-slider","value"),
            State("pdtp-nos-slider","value"),
            State("pdtp-t-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,pdtp_samples, pdtp_threshold, shapr_samples, shapr_threshold):
            art_extra_classifiers_dict = art_extra_classifiers(self.model)
            sample_indexes = pdtp_generate_samples(pdtp_samples, self.X_train_proc)
            leakage, _, _ = pdtp_metric(self.X_train_proc, self.Y_train, art_extra_classifiers_dict, list(self.model.keys())[0], pdtp_threshold, sample_indexes=sample_indexes, num_iter=1)
            text1 = f'''
                    Average PDTP leakage: {np.average(leakage)} \n
                    Max PDTP leakage: {np.max(leakage)}
                    '''
            SHAPr_leakage, _ = SHAPr_metric(self.X_train_proc.sample(shapr_samples), self.Y_train.sample(shapr_samples), self.X_test_proc.sample(shapr_samples), self.Y_test.sample(shapr_samples), art_extra_classifiers_dict, list(self.model.keys())[0],threshold_value=shapr_threshold)
            text2 = f'''
                    Average SHAPr leakage: {np.average(SHAPr_leakage)} \n
                    Max SHAPr leakage: {np.max(SHAPr_leakage)}
                    '''
            fig1, fig2 = visualisation(leakage, SHAPr_leakage,pdtp_threshold,shapr_threshold)
            return text1, fig1, text2, fig2
        
class ARTInferenceAttackComponent(ExplainerComponent):
    """
    A component class for displaying the results of a membership inference attack of ART to the dashboard.
    """
    def __init__(self, explainer, model, X_train_proc, Y_train, X_test_proc, Y_test, title="MIA", name=None):
        """
        Args:
            explainer: explainer instance from explainerdashboard
            model (dict): a container for the model
            X_train_proc (pd.DataFrame): a preprocessed X_train
            Y_train (pd.DataFrame): Y_train
            X_test_proc (pd.DataFrame): a preprocess X_test
            Y_test (pd.DataFrame): Y_test
            title (str, optional): title of the component. Defaults to "MIA".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.X_train_proc = X_train_proc
        self.Y_train = Y_train
        self.X_test_proc = X_test_proc
        self.Y_test = Y_test
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Membership Inference Attack")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                This black-box attack basically trains an additional regressor (called the attack model) to predict the membership status of a sample. It can use as input to the learning process probabilities/logits or losses, depending on the type of model and provided configuration. The higher the performance of the attack the greater the security risk is.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Attack Train Ratio: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, step=0.01, value=0.3, id='mia-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-mia"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(dcc.Graph(id="graph-mia")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("graph-mia","figure"),
            Input("button-mia","n_clicks"),
            State("mia-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,train_ratio):
            art_extra_classifiers_dict = art_extra_classifiers(self.model)
            inferred_train, inferred_test = art_mia(self.X_train_proc.to_numpy(), self.Y_train.to_numpy(), self.X_test_proc.to_numpy(), self.Y_test.to_numpy(), art_extra_classifiers_dict, list(self.model.keys())[0], attack_train_ratio=train_ratio)
            predicted = art_generate_predicted(inferred_train, inferred_test)
            actual = art_generate_actual(inferred_train, inferred_test)
            precision, recall = calc_precision_recall(predicted, actual)
            fig = mia_viz(precision,recall)
            return fig
        
class RobustnessTab(ExplainerComponent):
    """
    A tab class for displaying all robustness related components in single tab within the dashboard.
    """
    def __init__(self, explainer, model, X_train_proc, Y_train, X_test_proc, Y_test, model_type, title="Robustness", name=None):
        """
        Args:
            explainer: explainer instance from explainerdashboard
            model (dict): a container for the model
            X_train_proc (pd.DataFrame): a preprocessed X_train
            Y_train (pd.DataFrame): Y_train
            X_test_proc (pd.DataFrame): a preprocess X_test
            Y_test (pd.DataFrame): Y_test
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "MIA".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.security = BlankComponent(explainer) if model_type == 'classifier' else ARTInferenceAttackComponent(explainer, model, X_train_proc, Y_train, X_test_proc, Y_test) # ARTPrivacyComponent(explainer, model, X_train_proc, Y_train, X_test_proc, Y_test)
        
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
                self.security.layout()
            ]),
        ])
