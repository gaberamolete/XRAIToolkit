# Uncertainty
from .calibration import *

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
    
class CalibrationComponent(ExplainerComponent):
    """
    A component class for displaying all the calibration functions to the dashboard.
    """
    def __init__(self, explainer, model, X_test, Y_test, model_type, title="Calibration", name=None):
        """
        Args:
            explainer: explainer instance from the explainerdashboard
            model (dict): container for the model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Calibration".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.Y_test = Y_test
        self.reg = True if model_type == "regressor" else False
        if self.reg:
            self.y_pred_test = list(model.values())[0].predict(X_test)
        else:
            self.y_pred_test = list(model.values())[0].predict_proba(X_test)[:, 1]
        
        
    def layout(self):
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Calibration")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(dcc.Markdown('''
                                Through the toolkit, users should be able to use different post-hoc calibration methods like Logistic calibration, beta calibration, temperature scaling, and Isotonic regression to perform calibration.
                                
                                Calibration Metrics:
                                - Expected Calibration Error (ECE), which divides the confidence space into several bins and measures the observed accuracy in each bin. The bin gaps between observed accuracy and bin confidence are summed up and weighted by the amount of samples in each bin.
                                - Maximum Calibration Error (MCE), denotes the highest gap over all bins.
                                - Average Calibration Error (ACE), denotes the average miscalibration where each bin gets weighted equally.
                                - Negative Log Likelihood (NLL), measures the quality of a predicted probability distribution with respect to the ground truth.
                                - Pinball loss (PL) is a quantile-based calibration, and is a synonym for Quantile Loss. Tests for quantile calibration of a probabilistic regression model. This is an asymmetric loss that measures the quality of the predicted quantiles.
                                - Prediction Interval Coverage Probability (PICP), a quantile-based calibration. The is used for Bayesian models to determine quality of the uncertainty estimates. In Bayesian mode, an uncertainty estimate is attached to each sample. The PICP measures the probability that the true (observed) accuracy falls into the  𝑝% prediction interval. Returns the PICP and the Mean Prediction Interval Width (MPIW).
                                - Quantile Calibration Error (QCE), a quantile-based calibration. Returns the Marginal Quantile Calibration Error (M-QCE), which measures the gap between predicted quantiles and observed quantile coverage for multivariate distributions. This is based on the Normalized Estimation Error Squared (NEES), known from object tracking.
                                - Expected Normalized Calibration Error (ENCE), a variance-based calibration. Used for normal distributions, where we measure the quality of the predicted variance/stddev estimates. We require that the predicted variance matches the observed error variance, which is equivalent to the Mean Squared Error. ENCE applies a binning scheme with 𝐵 bins over the predicted standard deviation 𝜎𝑦(𝑋) and measures the absolute (normalized) difference between RMSE and RMV.
                                - Uncertainty Calibration Error (UCE), a variance-based calibration. Used for normal distributions, where we measure the quality of the predicted variance/stddev estimates. We require that the predicted variance matches the observed error variance, which is equivalent to the Mean Squared Error. UCE applies a binning scheme with 𝐵 bins over the predicted variance 𝜎2𝑦(𝑋) and measures the absolute difference between MSE and MV.
                            '''), style={"padding":"30px"})
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.P("Post-hoc Calibration: ", style={"padding-left":"30px","padding-right":"30px", "padding-top":"5px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div([dcc.Dropdown(
                                id="method-calib",
                                options= ['Logistic Calibration', 'Beta Calibration', 'Temperature Scaling', 'Isotonic Regression'] if self.reg==True else ['Logistic Calibration', 'Beta Calibration', 'Temperature Scaling', 'Isotonic Regression', 'Histogram Binning', 'Bayesian Binning into Quantiles', 'Ensemble of Near Isotonic Regression'], 
                                placeholder="Choose a Calibration Method",
                            ), dbc.Tooltip(id="tooltip-calib",target="method-calib")], style={"width":"100%", "padding-right":"30px"})
                        ]),
                        dbc.Col([
                            html.Div("Number of Bins: ", style={"padding-left":"20px"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div(dcc.Slider(1, 100, step=1, value=50, id='bins-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Div()
                        ]),
                        dbc.Col([
                            html.Div(dbc.Button("Compute", id="button-calib"), style={"margin":"auto"})
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
                            html.Div(dcc.Loading(dcc.Graph(id="graph-calib")), style={"margin":"auto"})
                        ], width="auto"),
                        dbc.Col([
                            html.Div()
                        ]),
                    ]),
                    html.Br(),
                    dbc.Row([
                        html.Div(dcc.Loading(html.Div(id="table-calib")), style={"margin":"auto", 'overflow':'scroll', 'padding':'20px', 'height':'500px', 'width':'auto'})
                    ]),
                ])
            ])
        ])
    
    def component_callbacks(self,app,**kwargs):
        @app.callback(
            Output("tooltip-calib","children"),
            Input("method-calib","value")
        )
        def update_tooltip(method):
            if method == "Logistic Calibration":
                return "Logistic Calibration, also known as Platt scaling, trains an SVM and then trains the parameters of an additional sigmoid function to map the SVM outputs into probabilities."
            elif method == "Beta Calibration":
                return "Beta Calibration, a well-founded and easily implemented improvement on Platt scaling for binary classifiers. Assumes that per-class scores of classifier each follow a beta distribution."
            elif method == "Temperature Scaling":
                return "Temperature Scaling, a single-parameter variant of Platt Scaling."
            elif method == "Isotonic Regression":
                return "Isotonic Regression, similar to Histogram Binning but with dynamic bin sizes and boundaries. A piecewise constant function gets for to ground truth labels sorted by given confidence estimates."
            elif method == "Histogram Binning":
                return "Histogram binning, where each prediction is sorted into a bin and assigned a calibrated confidence estimate."
            elif method == "Bayesian Binning into Quantiles":
                return "Bayesian Binning into Quantiles (BBQ). Utilizes multiple Histogram Binning instances with different amounts of bins, and computes a weighted sum of all methods to obtain a well-calibrated confidence estimate. Not recommended for regression outputs."
            elif method == "Ensemble of Near Isotonic Regression":
                return "Ensemble of Near Isotonic Regression models (ENIR). Allows a violation of monotony restrictions. Using the modified Pool-Adjacent-Violaters Algorithm (mPAVA), this method builds multiple Near Isotonic Regression Models and weights them by a certain score function. Not recommended for regression outputs."
            else:
                return ""
        
        @app.callback(
            Output("graph-calib","figure"),
            Output("table-calib","children"),
            Input("button-calib","n_clicks"),
            State("method-calib","value"),
            State("bins-slider","value"), prevent_initial_call=True
        )
        def update_graph(n_clicks,calib,n_bins):
            if self.reg:
                lc, lc_calibrated = calib_lc(self.y_pred_test, self.Y_test, True)
                bc, bc_calibrated = calib_bc(self.y_pred_test, self.Y_test, True)
                temp, temp_calibrated = calib_temp(self.y_pred_test, self.Y_test, True)
                hb, hb_calibrated = calib_hb(self.y_pred_test, self.Y_test, reg = True)
                ir, ir_calibrated = calib_ir(self.y_pred_test, self.Y_test, reg = True)
                #bbq, bbq_calibrated = calib_bbq(self.y_pred_test, self.Y_test, score = 'BIC', reg = True)
                calibs = {
                    # 'Uncalibrated': [y_pred_test, model],
                    'Logistic Calibration': [lc_calibrated, lc],
                    'Beta Calibration': [bc_calibrated, bc],
                    'Temperature Scaling': [temp_calibrated, bc],
                    #'Histogram Binning': [hb_calibrated, hb],
                    'Isotonic Regression': [ir_calibrated, ir],
                    #'Bayesian Binning into Quantiles': [bbq_calibrated, bbq],
                    # 'Ensemble of Near Isotonic Regression': [enir_calibrated, enir]
                }
                df = calib_metrics(self.Y_test, calibs, n_bins=n_bins, reg = True)
                fig, _ = plot_reliability_diagram(self.Y_test, calibs[calib][0], calibs[calib][1], title = calib, error_bars = True, n_bins = n_bins, reg = True)
                return fig, dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False)
            else:
                lc, lc_calibrated = calib_lc(self.y_pred_test, self.Y_test)
                bc, bc_calibrated = calib_bc(self.y_pred_test, self.Y_test)
                temp, temp_calibrated = calib_temp(self.y_pred_test, self.Y_test)
                hb, hb_calibrated = calib_hb(self.y_pred_test, self.Y_test)
                ir, ir_calibrated = calib_ir(self.y_pred_test, self.Y_test)
                bbq, bbq_calibrated = calib_bbq(self.y_pred_test, self.Y_test)
                enir, enir_calibrated = calib_enir(self.y_pred_test, self.Y_test)
                calibs = {
                    # 'Uncalibrated': [y_pred_test, model],
                    'Logistic Calibration': [lc_calibrated, lc],
                    'Beta Calibration': [bc_calibrated, bc],
                    'Temperature Scaling': [temp_calibrated, temp],
                    'Histogram Binning': [hb_calibrated, hb],
                    'Isotonic Regression': [ir_calibrated, ir],
                    'Bayesian Binning into Quantiles': [bbq_calibrated, bbq],
                    'Ensemble of Near Isotonic Regression': [enir_calibrated, enir]
                }
                df = calib_metrics(self.Y_test, calibs, n_bins=n_bins)
                fig, _ = plot_reliability_diagram(self.Y_test, calibs[calib][0], calibs[calib][1], title = calib, error_bars = True, n_bins = n_bins)
                return fig, dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],style_data={'whiteSpace':'normal','height':'auto',},fill_width=False)
            
class UncertaintyTab(ExplainerComponent):
    """
    A tab class for displaying all uncertainty components in a single tab within the dashboard.
    """
    def __init__(self, explainer, model, X_test, Y_test, model_type, title="Uncertainty", name=None):
        """
        Args:
            explainer: explainer instance from the explainerdashboard
            model (dict): container for the model
            X_test (pd.DataFrame): X_test
            Y_test (pd.DataFrame): Y_test
            model_type (str): regressor or classifier
            title (str, optional): title of the component. Defaults to "Uncertainty".
            name (optional): name of the component. Defaults to None.
        """
        super().__init__(explainer, title=title)
        self.calib = CalibrationComponent(explainer, model, X_test, Y_test, model_type)
        
    def layout(self):
        return dbc.Container([
            html.Br(),
            dbc.Row([
                self.calib.layout()
            ]),
        ])
