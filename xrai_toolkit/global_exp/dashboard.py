import shap
import base64
import io
import numpy as np

# Global explanation
from .global_exp import pd_profile, var_imp
from .global_exp import ld_profile, al_profile, compare_profiles
from .global_exp import initiate_shap_glob, shap_bar_glob, shap_summary, shap_dependence, shap_force_glob

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
        @app.callback(
            Output("grp-gpdp","options"),
            Input("var-gpdp","value"), prevent_initial_call=True
        )
        def update_dropdown(grp):
            cols = sorted(self.X_train.select_dtypes(exclude=["number"]).columns.tolist())
            cols.remove(grp)
            return cols
        
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
                                                On a model, the LD is defined as an expected value of predictions (or CP profiles) over a conditional distribution (the distribution of a certain explanatory variable). This conditional, or marginal, distribution is essentially some smaller part of the entire distribution, as if we had a regression tree dividing our distribution into defined parts.
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
                                                  This method averages the changes in the predictions, instead of the predictions themselves, and accumulates them over a grid. It does this by describing the local change of the model due to an explanatory variable, and averaging it over the explanatory variable's distribution. This ensures that the method is stable even with models containing highly correlated variables.
                                                  
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
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,list(self.model.values())[0][-1],self.pipe,self.cat)
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
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,list(self.model.values())[0][-1],self.pipe,self.cat)
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
    def __init__(self, explainer, model, X_train, pipe, cat, model_type, feature_names, title='Shap Dependence', name=None):
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
        self.feature_names = feature_names
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
                                options=self.feature_names,
                                placeholder="Select Variable to Analyze",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Select Another Variable (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="summary-var-optional",
                                options=self.feature_names,
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
            #create some matplotlib graph
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,list(self.model.values())[0][-1],self.pipe,self.cat)
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
                            html.Div(dcc.Loading(html.Iframe(id="graph-force-global", style={"width":"750px","height":"400px"})), style={"margin":"auto"})
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
            Output("graph-force-global","srcDoc"),
            Input("button-force-global","n_clicks"), prevent_initial_call=True
        )
        def update_figure(n_clicks):
            #create some matplotlib graph
            exp, shap_value_glob, feature_names = initiate_shap_glob(self.X_train,list(self.model.values())[0][-1],self.pipe,self.cat)
            X_train_proc = self.pipe.transform(self.X_train)
            html_file = shap_force_glob(exp,shap_value_glob,X_proc=X_train_proc,feature_names=feature_names,reg=self.reg,class_ind=0,class_names=['1','0'])
            return html_file

class GlobalExpTab(ExplainerComponent):
    """
    A tab class for displaying all the outputs of global explanation methods of XRAI
    """
    def __init__(self, explainer, exp, model, X_train, pipe, cat, model_type, var_grp, feature_names, title="Global Explanations", name=None):
        """
        Args:
            explainer: explainer instance from explainerdashboard
            exp: explainer from Dalex
            model (dict): a container for the model with the key as the model name
            X_train (pd.DataFrame): X_train
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            cat (pd.DataFrame): a subset of X_train with only categorical features
            model_type (str): regressor or classifier
            var_grp (dict): a custom made categories of feature for variable importance
            feature_names (list): list of features after the preprocessing pipeline
            title (str, optional): title of the component. Defaults to "Global Explanations".
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
        self.dependence = ShapDependenceComponent(explainer,model,X_train,pipe,cat,model_type,feature_names)
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