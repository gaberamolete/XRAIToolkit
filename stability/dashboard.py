from contextlib import redirect_stdout

# Stability
from .stability import *
from .decile import *

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
                        html.Div(dcc.Loading(html.Div(id="table-psi")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px', 'width':'auto'})
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
        features = pipe.get_feature_names_out()
        self.X_train = pd.DataFrame(pipe.transform(X_train), columns=features)
        self.X_test = pd.DataFrame(pipe.transform(X_test), columns=features)
        
        
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
                        html.Div(dcc.Loading(html.Div(id="table-ks")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px', 'width':'auto'})
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

class DataDriftComponent(ExplainerComponent):
    """
    A component class for displaying the output of data drift reports generated through EvidentlyAI in the dashboard
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Data Drift", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Data Drift".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Data Drift")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
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
                                
                                Note: All Interactive Inputs are Optional.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="drift-stattest",
                                options=['ks','chisquare','z','wasserstein','kl_div','psi','jensenshannon','anderson','fisher_exact','cramer_von_mises','g-test','hellinger','mannw','ed','es','t_test','emperical_mmd','TVD'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='ddr-st-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.P("Column: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="drift-col",
                                placeholder="Select a Column",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Categorical Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="ddr-cat-stattest",
                                options=['chisquare','z','fisher-exact','g-test','TVD'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        
                        dbc.Col([
                            html.P("Numerical Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="ddr-num-stattest",
                                options=['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Categorical Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='ddr-cst-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.Div("Numerical Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='ddr-nst-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.Div("Drift Share: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.5, id='ddr-ds-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-ddr"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-ddr", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("drift-col","options"),
            Input("drift-stattest","value"), prevent_initial_call=True
        )
        def update_dropdown(stattest):
            if stattest in ['chisquare','z','fisher-exact','g-test','TVD']:
                return sorted(self.current_data.select_dtypes(exclude=['number']).columns.tolist())
            elif stattest in ['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd']:
                return sorted(self.current_data.select_dtypes(include=['number']).columns.tolist())
            else:
                return sorted(self.current_data.columns.tolist())
            
        @app.callback(
            Output("iframe-ddr","srcDoc"),
            Input("button-ddr","n_clicks"),
            State("drift-stattest","value"),
            State("drift-col","value"),
            State("ddr-st-slider","value"),
            State("ddr-ds-slider","value"),
            State("ddr-cat-stattest","value"),
            State("ddr-cst-slider","value"),
            State("ddr-num-stattest","value"),
            State("ddr-nst-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, stattest, col, st_threshold, drift_share, cat_stattest, cat_threshold, num_stattest, num_threshold):
            if stattest in ['chisquare','z','fisher-exact','g-test','TVD']:
                cur = self.current_data.select_dtypes(exclude=['number'])
                ref = self.reference_data.select_dtypes(exclude=['number'])
            elif stattest in ['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd']:
                cur = self.current_data.select_dtypes(include=['number'])
                ref = self.reference_data.select_dtypes(include=['number'])
            else:
                cur = self.current_data
                ref = self.reference_data
            if col==None:
                data_drift_report, _ = data_drift_dataset_report(cur,ref,column_mapping=self.column_mapping,drift_share=drift_share,stattest=stattest,stattest_threshold=st_threshold,cat_stattest=cat_stattest,cat_stattest_threshold=cat_threshold,num_stattest=num_stattest,num_stattest_threshold=num_threshold)
                data_drift_report.save_html('assets/data_drift_report.html')
                with open("assets/data_drift_report.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content
            else:
                data_drift_report_col, _ = data_drift_column_report(cur,ref,col,stattest=stattest,stattest_threshold=st_threshold)
                data_drift_report_col.save_html('assets/data_drift_report_col.html')
                with open("assets/data_drift_report_col.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content

class DataDriftTestComponent(ExplainerComponent):
    """
    A component class for displaying the output the results of data drift tests to the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Data Drift Test", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Data Drift Test".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Data Drift Test")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
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
                                
                                Note: All Interactive Inputs are Optional.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Statistic Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="drifttest-stattest",
                                options=['ks','chisquare','z','wasserstein','kl_div','psi','jensenshannon','anderson','fisher_exact','cramer_von_mises','g-test','hellinger','mannw','ed','es','t_test','emperical_mmd','TVD'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='ddt-st-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.P("Column: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="drifttest-col",
                                placeholder="Select a Column",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Categorical Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="ddt-cat-stattest",
                                options=['chisquare','z','fisher-exact','g-test','TVD'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        
                        dbc.Col([
                            html.P("Numerical Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="ddt-num-stattest",
                                options=['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Categorical Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='ddt-cst-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.Div("Numerical Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='ddt-nst-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-ddt"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-ddt", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("drifttest-col","options"),
            Input("drifttest-stattest","value"), prevent_initial_call=True
        )
        def update_dropdown(stattest):
            if stattest in ['chisquare','z','fisher-exact','g-test','TVD']:
                return sorted(self.current_data.select_dtypes(exclude=['number']).columns.tolist())
            elif stattest in ['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd']:
                return sorted(self.current_data.select_dtypes(include=['number']).columns.tolist())
            else:
                return sorted(self.current_data.columns.tolist())
            
        @app.callback(
            Output("iframe-ddt","srcDoc"),
            Input("button-ddt","n_clicks"),
            State("drifttest-stattest","value"),
            State("drifttest-col","value"),
            State("ddt-st-slider","value"),
            State("ddt-cat-stattest","value"),
            State("ddt-cst-slider","value"),
            State("ddt-num-stattest","value"),
            State("ddt-nst-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, stattest, col, st_threshold, cat_stattest, cat_threshold, num_stattest, num_threshold):
            if stattest in ['chisquare','z','fisher-exact','g-test','TVD']:
                cur = self.current_data.select_dtypes(exclude=['number'])
                ref = self.reference_data.select_dtypes(exclude=['number'])
            elif stattest in ['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd']:
                cur = self.current_data.select_dtypes(include=['number'])
                ref = self.reference_data.select_dtypes(include=['number'])
            else:
                cur = self.current_data
                ref = self.reference_data
            if col==None:
                data_drift_test, _ = data_drift_dataset_test(cur,ref,column_mapping=self.column_mapping,stattest=stattest,stattest_threshold=st_threshold,cat_stattest=cat_stattest,cat_stattest_threshold=cat_threshold,num_stattest=num_stattest,num_stattest_threshold=num_threshold)
                data_drift_test.save_html('assets/data_drift_test.html')
                with open("assets/data_drift_test.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content
            else:
                data_drift_test_col, _ = data_drift_column_test(cur,ref,col,stattest=stattest,stattest_threshold=st_threshold)
                data_drift_test_col.save_html('assets/data_drift_test_col.html')
                with open("assets/data_drift_test_col.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content

class DataQualityComponent(ExplainerComponent):
    """
    A component class for displaying the data quality report of the dataset using EvidentlyAI to the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Data Quality", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Data Drift Test".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Data Quality")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Calculate various descriptive statistics, the number and share of missing values per column, and correlations between columns in the dataset.
                                
                                For an identified column:
                                - Calculates various descriptive statistics,
                                - Calculates number and share of missing values
                                - Plots distribution histogram,
                                - Calculates quantile value and plots distribution,
                                - Calculates correlation between defined column and all other columns
                                - If categorical, calculates number of values in list / out of the list / not found in defined column
                                - If numerical, calculates number and share of values in specified range / out of range in defined column, and plots distributions
                                
                                Note: All interactive inputs are optional
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Almost Duplicated Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.95, id='dqr-adt-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.Div("Almost Constant Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.95, id='dqr-act-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Column: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="quality-col",
                                options=sorted(self.current_data.columns.tolist()),
                                placeholder="Select a Column",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Quantile: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.75, id='dqr-q-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-ddq"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-ddq", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("iframe-ddq","srcDoc"),
            Input("button-ddq","n_clicks"),
            State("quality-col","value"),
            State("dqr-adt-slider","value"),
            State("dqr-act-slider","value"),
            State("dqr-q-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, col, adt, act, quantile):
            if col==None:
                data_quality_report, _ = data_quality_dataset_report(self.current_data,self.reference_data,column_mapping=self.column_mapping,adt=adt,act=act)
                data_quality_report.save_html('assets/data_quality_report.html')
                with open("assets/data_quality_report.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content
            else:
                data_quality_report_col, _ = data_quality_column_report(self.current_data,self.reference_data,col,quantile=quantile)
                data_quality_report_col.save_html('assets/data_quality_report_col.html')
                with open("assets/data_quality_report_col.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content

class DataQualityTestComponent(ExplainerComponent):
    """
    A component class for displaying the data quality test results using EvidentlyAI to the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Data Quality Test", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Data Quality Test".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Data Quality Test")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                For all columns in a dataset:
                                - Tests number of rows and columns against reference or defined condition
                                - Tests number and share of missing values in the dataset against reference or defined condition
                                - Tests number and share of columns and rows with missing values against reference or defined condition
                                - Tests number of differently encoded missing values in the dataset against reference or defined condition
                                - Tests number of columns with all constant values against reference or defined condition
                                - Tests number of empty rows (expects 10% or none) and columns (expects none) against reference or defined condition
                                - Tests number of duplicated rows (expects 10% or none) and columns (expects none) against reference or defined condition
                                - Tests types of all columns against the reference, expecting types to match
                                
                                If column is selected, the tests will be performed for that specific column.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Column: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="qualitytest-col",
                                options=sorted(self.current_data.columns.tolist()),
                                placeholder="Select a Column",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Number of Sigma: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 6, value=2, step=1, id='dqt-nos-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.Div("Quantile: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.25, id='dqt-q-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-dqt"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-dqt", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("iframe-dqt","srcDoc"),
            Input("button-dqt","n_clicks"),
            State("qualitytest-col","value"),
            State("dqt-nos-slider","value"),
            State("dqt-q-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, col, n_sigmas, quantile):
            if col==None:
                data_quality_test, _ = data_quality_dataset_test(self.current_data,self.reference_data,column_mapping=self.column_mapping)
                data_quality_test.save_html('assets/data_quality_test.html')
                with open("assets/data_quality_test.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content
            else:
                data_quality_test_col, _ = data_quality_column_test(self.current_data,self.reference_data,col,n_sigmas=n_sigmas,quantile=quantile)
                data_quality_test_col.save_html('assets/data_quality_test_col.html')
                with open("assets/data_quality_test_col.html", "r", encoding='utf8') as f:
                    html_content =  f.read()
                return html_content

class TargetDriftComponent(ExplainerComponent):
    """
    A component class for displaying the target drift reporth through EvidentlyAI to the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Target Drift", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Target Drift".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Target Drift")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
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
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="tdrift-stattest",
                                options=['ks','chisquare','z','wasserstein','kl_div','psi','jensenshannon','anderson','fisher_exact','cramer_von_mises','g-test','hellinger','mannw','ed','es','t_test','emperical_mmd','TVD'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='tdr-st-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Categorical Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="tdr-cat-stattest",
                                options=['chisquare','z','fisher-exact','g-test','TVD'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Categorical Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='tdr-cst-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Numerical Stat Test: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="tdr-num-stattest",
                                options=['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd'],
                                placeholder="Choose a Statistics Test",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Numerical Stat Test Threshold: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='tdr-nst-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-tdr"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-tdr", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):   
        @app.callback(
            Output("iframe-tdr","srcDoc"),
            Input("button-tdr","n_clicks"),
            State("tdrift-stattest","value"),
            State("tdr-st-slider","value"),
            State("tdr-cat-stattest","value"),
            State("tdr-cst-slider","value"),
            State("tdr-num-stattest","value"),
            State("tdr-nst-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, stattest, st_threshold, cat_stattest, cat_threshold, num_stattest, num_threshold):
            if stattest in ['chisquare','z','fisher-exact','g-test','TVD']:
                cur = self.current_data.select_dtypes(exclude=['number'])
                ref = self.reference_data.select_dtypes(exclude=['number'])
            elif stattest in ['ks','wasserstein','anderson','cramer_von_mises','mannw','ed','es','t_test','emperical_mmd']:
                cur = self.current_data.select_dtypes(include=['number'])
                ref = self.reference_data.select_dtypes(include=['number'])
            else:
                cur = self.current_data
                ref = self.reference_data
            td_report, _ = target_drift_report(cur,ref,column_mapping=self.column_mapping,stattest=stattest,stattest_threshold=st_threshold,cat_stattest=cat_stattest,cat_stattest_threshold=cat_threshold,num_stattest=num_stattest,num_stattest_threshold=num_threshold)
            td_report.save_html('assets/target_drift_report.html')
            with open("assets/target_drift_report.html", "r", encoding='utf8') as f:
                html_content =  f.read()
            return html_content

class RegressionPerformanceComponent(ExplainerComponent):
    """
    A component class for displaying the regression performance report using EvidentlyAI to the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Regression Performance", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Regression Performance".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Regression Performance")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
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
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Columns (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="rpr-col",
                                options=sorted(self.current_data.columns.tolist()),
                                placeholder="Select Multiple Columns",
                                multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Top Error: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='rpr-te-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-rpr"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-rpr", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("iframe-rpr","srcDoc"),
            Input("button-rpr","n_clicks"),
            State("rpr-col","value"), 
            State("rpr-te-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,columns,top_error):
            rp_report, _ = regression_performance_report(self.current_data,self.reference_data,column_mapping=self.column_mapping,top_error=top_error,columns=columns)
            rp_report.save_html('assets/regression_performance_report.html')
            with open("assets/regression_performance_report.html", "r", encoding='utf8') as f:
                html_content =  f.read()
            return html_content
        
class RegressionPerformanceTestComponent(ExplainerComponent):
    """
    A component class for displaying the test results on regression performance using EvidentlyAI to the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Regression Performance Test", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Regression Performance Test".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Regression Performance Test")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Computes the following tests on regression data, failing if +/- a percentage (%) of scores over reference data is achieved:
                                - Mean Absolute Error (MAE)
                                - Root Mean Squared Error (RMSE)
                                - Mean Error (ME) and tests if it is near zero
                                - Mean Absolute Percentage Error (MAPE)
                                - Absolute Maximum Error
                                - R2 Score (coefficient of determination)
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Relative Percentage: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.1, id='rpt-rp-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-rpt"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-rpt", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("iframe-rpt","srcDoc"),
            Input("button-rpt","n_clicks"),
            State("rpt-rp-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,rel_val):
            rp_test, _ = regression_performance_test(self.current_data,self.reference_data,column_mapping=self.column_mapping,rel_val=rel_val)
            rp_test.save_html('assets/regression_performance_test.html')
            with open("assets/regression_performance_test.html", "r", encoding='utf8') as f:
                html_content =  f.read()
            return html_content

class ClassificationPerformanceComponent(ExplainerComponent):
    """
    A component class for displaying the classification performance report using EvidentlyAI to the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Classification Performance", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Classification Performance".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Classification Performance")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
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
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Probabilistic: ", style={"padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id="cpr-is-prob",                            
                                options=[{"label": "Yes","value": True},{"label": "No","value": False}],
                                placeholder="Is the Target Probabilistic?",
                                style={"width":"100%", "padding-right":"30px"},
                            ),
                        ]),
                        dbc.Col([
                            html.Div("Probas Threshold (If Probabilistic): ", style={"padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="cpr-probas-threshold",
                                type="number",
                                min=0,
                                max=1,
                                step=0.01,
                                value=None,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.P("Columns (Optional): ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="cpr-col",
                                options=sorted(self.current_data.columns.tolist()),
                                placeholder="Select Multiple Columns",
                                multi=True,
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-cpr"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-cpr", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("iframe-cpr","srcDoc"),
            Input("button-cpr","n_clicks"),
            State("cpr-is-prob","value"),
            State("cpr-probas-threshold","value"),
            State("cpr-col","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,is_prob,probas_threshold,columns):
            cp_report, _ = classification_performance_report(self.current_data,self.reference_data,column_mapping=self.column_mapping,is_prob=is_prob,probas_threshold=probas_threshold,columns=columns)
            cp_report.save_html('assets/classification_performance_report.html')
            with open("assets/classification_performance_report.html", "r", encoding='utf8') as f:
                html_content =  f.read()
            return html_content

class ClassificationPerformanceTestComponent(ExplainerComponent):
    """
    A component class for displaying the test results on classification performance using EvidentlyAI to the dashboard
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, title="Classification Performance Test", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Classification Performance Test".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        self.current_data, self.reference_data, self.column_mapping = mapping_columns(test_data,train_data,self.model,target_feature)
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Classification Performance Test")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Computes the following tests on classification data, failing if +/- a percentage (%) of scores over reference data is achieved:
                                - Accuracy, Precision, Recall, F1 on the whole dataset
                                - Precision, Recall, F1 on each class
                                - Computes the True Positive Rate (TPR), True Negative Rate (TNR), False Positive Rate (FPR), False Negative Rate (FNR)
                                - For probabilistic classification, computes the ROC AUC and LogLoss
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Probabilistic: ", style={"padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id="cpt-is-prob",                            
                                options=[{"label": "Yes","value": True},{"label": "No","value": False}],
                                placeholder="Is the Target Probabilistic?",
                                style={"width":"100%", "padding-right":"30px"},
                            ),
                        ]),
                        dbc.Col([
                            html.Div("Probas Threshold (If Probabilistic): ", style={"padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dbc.Input(
                                id="cpt-probas-threshold",
                                type="number",
                                min=0,
                                max=1,
                                step=0.01,
                                value=None,
                            ),  style={"width":"100%","padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Relative Percentage: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.2, id='cpt-rp-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-cpt"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Iframe(id="iframe-cpt", style={"height":"500px","width":"98%"})))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("iframe-cpt","srcDoc"),
            Input("button-cpt","n_clicks"), 
            State("cpt-is-prob","value"),
            State("cpt-probas-threshold","value"),
            State("cpt-rp-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,is_prob,probas_threshold,rel_val):
            cp_test, _ = classification_performance_test(self.current_data,self.reference_data,column_mapping=self.column_mapping,is_prob=is_prob,probas_threshold=probas_threshold,rel_val=rel_val)
            cp_test.save_html('assets/classification_performance_test.html')
            with open("assets/classification_performance_test.html", "r", encoding='utf8') as f:
                html_content =  f.read()
            return html_content

class AlibiCVMComponent(ExplainerComponent):
    """
    A component class for displaying the results on cramer von mises test using AlibiDetect to the dashboard.
    """
    def __init__(self, explainer, model, pipe, X_test, Y_test, X_train, Y_train, model_type, title="CVM", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "CVM".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        if self.model_type == 'regressor':
            # Shift data by adding Gaussian noise
            X_concept = X_test.copy()
            Y_concept = self.model.predict(X_concept) * 1.5 + np.random.normal(1, 10, size = len(Y_test))
            
            # Supervised drift detection
            self.loss_ref = (self.model.predict(X_train) - Y_train) ** 2
            lossr_test = (self.model.predict(X_test) - Y_test) ** 2
            lossr_concept = (self.model.predict(X_concept) - Y_concept) ** 2

            self.losses = {'No drift': lossr_test.to_numpy(), 'Concept drift': lossr_concept}
        else:
            # Shift data by adding artificially adding drift to X_test
            X_covar, y_covar = X_test.copy(), Y_test.copy()
            X_concept, y_concept = X_test.copy(), Y_test.copy()

            # Apply covariate drift by altering some data in x (manual altering)
            idx1 = Y_test[Y_test == 1].index
            X_covar.loc[idx1, 'last_pymnt_amnt'] += 10000

            # Apply concept drift by switching two species
            idx2 = Y_test[Y_test == 0].index
            y_concept[idx1] = 0
            y_concept[idx2] = 1

            Xs = {'No drift': pipe.transform(X_test), 'Covariate drift': pipe.transform(X_covar), 'Concept drift': pipe.transform(X_concept)}
            # Xs

            self.loss_ref = (self.model.predict(X_train) == Y_train).astype(int)
            loss_test = (self.model.predict(X_test) == Y_test).astype(int)
            loss_covar = (self.model.predict(X_covar) == y_covar).astype(int)
            loss_concept = (self.model.predict(X_concept) == y_concept).astype(int)
            self.losses = {'No drift': loss_test.to_numpy(), 'Covariate drift': loss_covar.to_numpy(), 'Concept drift': loss_concept.to_numpy()}
            
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("AlibiDetect: Cramer von Mises")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Cramer-von Mises (CVM) data drift detector, which tests for any change in the distribution of continuous univariate data. 
                                Works for both regression and classification use cases. For multivariate data, a separate CVM test is applied to each 
                                feature, and the obtained p-values are aggregated via the Bonferroni or False Discovery Rate (FDR) corrections.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("P-Value: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='cramer-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-cramer"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Pre(id="text-cramer")))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("text-cramer","children"),
            Input("button-cramer","n_clicks"),
            State("cramer-slider",'value'), prevent_initial_call=True
        )
        def update_graph(n_clicks,p_val):
            with open("temp.log", "w") as f:
                with redirect_stdout(f):
                    cvm_dict = cramer_von_mises(self.loss_ref, self.losses, p_val = p_val)
            with open("temp.log") as f:
                contents = f.readlines()
            return contents

class AlibiFETComponent(ExplainerComponent):
    """
    A component class for displaying the result on fisher exact test using AlibiDetect to the dashboard.
    """
    def __init__(self, explainer, model, pipe, X_test, Y_test, X_train, Y_train, title="FET", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            title (str, optional): title of the component. Defaults to "FET".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = list(model.values())[0]
        # Shift data by adding artificially adding drift to X_test
        X_covar, y_covar = X_test.copy(), Y_test.copy()
        X_concept, y_concept = X_test.copy(), Y_test.copy()

        # Apply covariate drift by altering some data in x (manual altering)
        idx1 = Y_test[Y_test == 1].index
        X_covar.loc[idx1, 'last_pymnt_amnt'] += 10000

        # Apply concept drift by switching two species
        idx2 = Y_test[Y_test == 0].index
        y_concept[idx1] = 0
        y_concept[idx2] = 1

        Xs = {'No drift': pipe.transform(X_test), 'Covariate drift': pipe.transform(X_covar), 'Concept drift': pipe.transform(X_concept)}

        self.loss_ref = (self.model.predict(X_train) == Y_train).astype(int)
        loss_test = (self.model.predict(X_test) == Y_test).astype(int)
        loss_covar = (self.model.predict(X_covar) == y_covar).astype(int)
        loss_concept = (self.model.predict(X_concept) == y_concept).astype(int)
        self.losses = {'No drift': loss_test, 'Covariate drift': loss_covar, 'Concept drift': loss_concept}
            
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("AlibiDetect: Fishers Exact Test")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Fisher exact test (FET) data drift detector, which tests for a change in the mean of binary univariate data. Works for classification use cases only.
                                For multivariate data, a separate FET test is applied to each feature, and the obtained p-values are aggregated via the Bonferroni or False Discovery Rate (FDR) corrections.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("P-Value: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='fet-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-fet"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Pre(id="text-fet")))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("text-fet","children"),
            Input("button-fet","n_clicks"),
            State("fet-slider",'value'), prevent_initial_call=True
        )
        def update_graph(n_clicks,p_val):
            with open("temp.log", "w") as f:
                with redirect_stdout(f):
                    fet_dict = fishers_exact_test(self.loss_ref, self.losses, p_val = p_val)
            with open("temp.log") as f:
                contents = f.readlines()
            return contents
        
class TabularDriftComponent(ExplainerComponent):
    """
    A component class for displaying the tabular drift table using AlibiDetect to the dashboard
    """
    def __init__(self, explainer, pipe, train_data, test_data, title="Tabular Drift", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            train_data (pd.DataFrame): training data
            test_data (pd.DataFrame): test data
            title (str, optional): title of the component. Defaults to "Tabular Drift".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.train_data = train_data
        self.test_data = test_data
        self.pipe = pipe
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("AlibiDetect: Tabular Drift")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Mixed-type tabular data drift detector with Bonferroni or False Discovery Rate (FDR) correction for multivariate data.
                                Kolmogorov-Smirnov (K-S) univariate tests are applied to continuous numerical data and Chi-Squared (Chi2) univariate tests to categorical data.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Drift Type: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="td-drop",
                                options=['batch','feature'],
                                placeholder="Drift Type",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("P-Value: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='td-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-td", n_clicks=0), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Div(id="table-td")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px', 'width':'auto'})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("table-td","children"),
            Input("button-td","n_clicks"),
            State("td-drop","value"),
            State("td-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks, drift, p_val):
            df = tabular_drift(self.train_data, self.test_data, self.pipe, p_val=p_val, drift_type=drift)
            print(df)
            return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False)

class AlibiCSComponent(ExplainerComponent):
    """
    A component class for displaying the chi-square test using AlibiDetect to the dashboard 
    """
    def __init__(self, explainer, X_train, X_test, title="Chi Square", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            X_train (pd.DataFrame): X_train
            X_test (pd.DataFrame): X_test
            title (str, optional): title of the component. Defaults to "Chi Square".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.X_train = X_train
        self.X_test = X_test
            
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("AlibiDetect: Chi-Square")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                For categorical variables, Chi-Squared data drift detector with Bonferroni or False Discovery Rate (FDR) correction for multivariate data.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("P-Value: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(0, 1, value=0.05, id='cs-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-cs"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Pre(id="text-cs")))
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("text-cs","children"),
            Input("button-cs","n_clicks"),
            State("cs-slider",'value'), prevent_initial_call=True
        )
        def update_graph(n_clicks,p_val):
            with open("temp.log", "w") as f:
                with redirect_stdout(f):
                    cs = chi_sq(self.X_train, self.X_test, p_val=p_val)
            with open("temp.log") as f:
                contents = f.readlines()
            return contents

class DecileComponent(ExplainerComponent):
    """
    A component class for displaying the results of the decile analysis functions to the dashboard.
    """
    def __init__(self, explainer, model, X_train, X_test, train_data, test_data, target_feature, title="Decile", name=None):
        """
        Args:
            explainer: explainer instance generated through explainerdashboard
            model (dict): a container for the model
            X_train (pd.DataFrame): X_train
            X_test (pd.DataFrame): X_test
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): the column name of the target feature
            title (str, optional): title of the component. Defaults to "Decile".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.target_feature = target_feature
        self.train_data['prob'] = pd.Series(list(model.values())[0].predict_proba(X_train)[:, 1], index = train_data.index)
        self.test_data['prob'] = pd.Series(list(model.values())[0].predict_proba(X_test)[:, 1], index = test_data.index)
            
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Decile Analysis")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                The Decile Table is creared by first sorting the rows by their predicted probabilities, in decreasing order from highest (closest to one) to lowest (closest to zero). Splitting the rows into equally sized segments, we create groups containing the same numbers of rows, for example, 10 decile groups each containing 10% of the row base.
                                
                                LABELS INFO:
                                - prob_min         : Minimum probability in a particular decile 
                                - prob_max         : Minimum probability in a particular decile
                                - prob_avg         : Average probability in a particular decile
                                - cnt_events       : Count of events in a particular decile
                                - cnt_resp         : Count of responders in a particular decile
                                - cnt_non_resp     : Count of non-responders in a particular decile
                                - cnt_resp_rndm    : Count of responders if events assigned randomly in a particular decile
                                - cnt_resp_wiz     : Count of best possible responders in a particular decile
                                - resp_rate        : Response Rate in a particular decile \[(cnt_resp/cnt_cust)*100\]
                                - cum_events       : Cumulative sum of events decile-wise
                                - cum_resp         : Cumulative sum of responders decile-wise 
                                - cum_resp_wiz     : Cumulative sum of best possible responders decile-wise 
                                - cum_non_resp     : Cumulative sum of non-responders decile-wise 
                                - cum_events_pct   : Cumulative sum of percentages of events decile-wise 
                                - cum_resp_pct     : Cumulative sum of percentages of responders decile-wise
                                - cum_resp_pct_wiz : Cumulative sum of percentages of best possible responders decile-wise 
                                - cum_non_resp_pct : Cumulative sum of percentages of non-responders decile-wise 
                                - KS               : KS Statistic decile-wise
                                - lift             : Cumuative Lift Value decile-wise
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("Decile Groups: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(1, 100, step=1, value=10, id='dec-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                        dbc.Col([
                            html.P("Model Selection Method: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Dropdown(
                                id="dec-drop",
                                options=['Gain Chart','Lift Chart','Lift Decile Chart','KS Statistic'],
                                placeholder="Model Selection by ...",
                            ), style={"width":"100%", "padding-right":"30px"})
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-dec"), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Pre(id="text-dec")))
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dcc.Loading(dcc.Graph(id="graph-dec")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Div(id="table-dec")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px', 'width':'auto'})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):    
        @app.callback(
            Output("text-dec","children"),
            Output("graph-dec","figure"),
            Output("table-dec","children"),
            Input("button-dec","n_clicks"),
            State("dec-slider",'value'),
            State("dec-drop","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,change_deciles,method):
            df, text = decile_table(self.train_data[self.target_feature], self.train_data['prob'],change_deciles)
            if method == "Gain Chart":
                fig = model_selection_by_gain_chart({list(self.model.keys())[0]: df})
            elif method == "Lift Chart":
                fig = model_selection_by_lift_chart({list(self.model.keys())[0]: df})
            elif method == "Lift Decile Chart":
                fig = model_selection_by_lift_decile_chart({list(self.model.keys())[0]: df})
            else:
                fig = model_selection_by_ks_statistic({list(self.model.keys())[0]: df})
            return text, fig, dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False)

class StabilityTab(ExplainerComponent):
    """
    A tab class for displaying the all the stability components in a single tab within the dashboard
    """
    def __init__(self, explainer, X_train, Y_train, X_test, Y_test, cont, pipe, model, train_data, test_data, target_feature, model_type, title="Stability (Metrics)", name=None):
        """
        Args:
            explainer: explainer instance from explainerdashboard
            X_train (pd.DataFrame): X_train
            Y_train (pd.DataFrame): Y_train
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            cont (pd.DataFrame): a subset of X_train containing only continuous features
            pipe (sklearn.Pipeline): the preprocessing pipeline of the model
            model (dict): a container for the model selected
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): target column
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Stability (Metrics)".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.psi = PSIComponent(explainer,X_train,X_test,cont)
#        self.ph = PageHinkleyComponent(explainer,X_train, X_test, cont)
        self.ks = KSTestComponent(explainer,X_train, X_test,pipe)
        self.dd = DataDriftComponent(explainer,model,train_data,test_data,target_feature)
        self.dq = DataQualityComponent(explainer,model,train_data,test_data,target_feature)
        self.td = TargetDriftComponent(explainer,model,train_data,test_data,target_feature)
        self.pr = RegressionPerformanceComponent(explainer,model,train_data,test_data,target_feature) if model_type=='regressor' else ClassificationPerformanceComponent(explainer,model,train_data,test_data,target_feature)
        self.cvm = AlibiCVMComponent(explainer,model,pipe,X_test,Y_test,X_train,Y_train,model_type)
        self.fet = AlibiFETComponent(explainer,model,pipe,X_test,Y_test,X_train,Y_train) if model_type == 'classifier' else BlankComponent(explainer)
        self.atd = TabularDriftComponent(explainer,pipe,train_data,test_data)
        self.cs = AlibiCSComponent(explainer,X_train,X_test)
        self.dec = DecileComponent(explainer, model, X_train, X_test, train_data, test_data, target_feature) if model_type == 'classifier' else BlankComponent(explainer) 
        
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
                self.psi.layout()
            ]),
            html.Br(),
#             dbc.Row([
#                 self.ph.layout()
#             ]),
#             html.Br(),
            dbc.Row([
                self.ks.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.dec.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.dd.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.dq.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.td.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.pr.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.cvm.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.fet.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.atd.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.cs.layout()
            ]),
        ])
    
class StabilityTestTab(ExplainerComponent):
    """
    A tab class for displaying the stability test class components in a single tab within the dashboard.
    """
    def __init__(self, explainer, model, train_data, test_data, target_feature, model_type, title="Stability (Tests)", name=None):
        """
        Args:
            explainer: explainer instance from explainerdashboard
            model (dict): a container for the model selected
            train_data (pd.DataFrame): training set
            test_data (pd.DataFrame): test set
            target_feature (str): target column
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Stability (Tests)".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.dd = DataDriftTestComponent(explainer,model,train_data,test_data,target_feature)
        self.dq = DataQualityTestComponent(explainer,model,train_data,test_data,target_feature)
        self.pt = RegressionPerformanceTestComponent(explainer,model,train_data,test_data,target_feature) if model_type == 'regressor' else ClassificationPerformanceTestComponent(explainer,model,train_data,test_data,target_feature)
        
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
                self.dd.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.dq.layout()
            ]),
            html.Br(),
            dbc.Row([
                self.pt.layout()
            ]),
        ])
