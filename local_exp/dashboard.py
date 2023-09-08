import shap
import base64
import io

# Local explanation
from .local_exp import dice_exp, exp_cf
from .local_exp import dalex_exp, break_down, interactive
from .local_exp import Predictor, exp_qii, get_feature_names, cp_profile
from .local_exp import initiate_shap_loc, shap_waterfall, shap_force_loc, shap_bar_loc

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
    def __init__(self, explainer, model, pipe, X_test, title="QIIExp", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            X_test (pd.DataFrame): X_test
            title (str, optional): title of the component. Defaults to "DiceExp".
            name (optional): name of the component. Defaults to None. 
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0][-1]
        self.pipe = pipe
        self.X_test = X_test
        
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
                        html.Div(dcc.Loading(html.Div(id="table-qii")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px','width':'auto'})
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
            df, fig = exp_qii(self.model, self.X_test, idx, self.pipe, method = method)
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
            exp, shap_value_loc, feature_names = initiate_shap_loc(self.X_train,list(self.model.values())[0][-1],self.pipe,self.cat)
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
            exp, shap_value_loc, feature_names = initiate_shap_loc(self.X_train,list(self.model.values())[0][-1],self.pipe,self.cat)
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
            exp, shap_value_loc, feature_names = initiate_shap_loc(self.X_train,list(self.model.values())[0][-1],self.pipe,self.cat)
            fig = shap_bar_loc(shap_value_loc,idx,feature_names=feature_names,reg=self.reg,show=False,class_ind=0,class_names=['1','0'])
            buf = io.BytesIO() # in-memory files
            fig.savefig(buf, format='png', bbox_inches = 'tight')
            data = base64.b64encode(buf.getbuffer()).decode("ascii") # encode to html elements
            buf.close()
            return "data:image/png;base64,{}".format(data)
        
class LocalExpTab(ExplainerComponent):
    """
    A tab component to display all the output of the local explanation methods in the dashboard
    """
    def __init__(self, explainer, model, X_train, Y_train, X_test, cont, cat, model_type, target_feature, pipe=None, title="Local Explanations", name=None):
        """
        Args:
            explainer: a dummy explainer, won't really be utilized to extract info from
            model (dict): a dict containing a single model
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            cont (pd.DataFrame): a subset of X_train with only continuous features
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            title (str, optional): title of the component. Defaults to "Local Explanation".
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
            self.qii = QIIExpComponent(explainer, model, pipe, X_test)
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
