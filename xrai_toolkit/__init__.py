
# load model and data
from .model_ingestion.data_model import load_data_model

# EDA
# from XRAIDashboard.eda.auto_eda import *
from .eda.auto_eda import ydata_profiling_eda2, autoviz_eda2

# Performance overview 
# from XRAIDashboard.fairness.fairness import model_performance
from .fairness.fairness import gini_coefficient, model_performance, fairness

# # Fairness
# # from XRAIDashboard.fairness.fairness import fairness
# # from XRAIDashboard.fairness.fairness_algorithm import *
from .fairness.fairness_algorithm import disparate_impact_remover, algo_exp, metrics_plot, exponentiated_gradient_reduction, meta_classifier, compute_metrics, compare_algorithms, reweighing,   compute_metrics, calibrated_eqodds, reject_option
# # from XRAIDashboard.fairness.cluster_metrics import *
from .fairness.cluster_metrics import silhouette_score_visualiser, rand_index, adjusted_rand_index, mutual_info, CH_index, db_index
# # from XRAIDashboard.fairness.XRAI_features import *
from .fairness.XRAI_features import xrai_features

# Local explanation
# from XRAIDashboard.local_exp.local_exp import *
from .local_exp.local_exp import dice_exp, exp_cf, Predictor, get_feature_names, exp_qii, dalex_exp, break_down, interactive, cp_profile, initiate_shap_loc, shap_waterfall, shap_force_loc, shap_bar_loc

# Global explanation
# from XRAIDashboard.global_exp.global_exp import *
from .global_exp.global_exp import dalex_exp, pd_profile, var_imp, ld_profile, al_profile, compare_profiles, get_feature_names, initiate_shap_glob, shap_bar_glob, shap_summary, shap_dependence, shap_force_glob

# Stability
# from XRAIDashboard.stability.stability import *
from .stability.stability import classification_performance_report, data_quality_column_report, data_drift_dataset_test, data_quality_dataset_report, maximum_mean_discrepancy, data_drift_column_test, data_quality_dataset_test, data_drift_dataset_report, categs, mapping_columns, cramer_von_mises, psi_list, target_drift_report, calculate_psi, generate_psi_df, regression_performance_test, fishers_exact_test, data_quality_column_test, classification_performance_test, ks, data_drift_column_report, regression_performance_report, get_feature_names
# from XRAIDashboard.stability.decile import *
from .stability.decile import print_labels, decile_table, model_selection_by_gain_chart, model_selection_by_lift_chart, model_selection_by_lift_decile_chart, model_selection_by_ks_statistic, decile_report

#Outlier
# from XRAIDashboard.fairness.outlier import *
from .fairness.outlier import outlier, removal, visualize, method_exp

#Uncertainty
# from XRAIDashboard.uncertainty.calibration import *
from .uncertainty.calibration import calib_lc, calib_bc, calib_temp, calib_hb, calib_ir, calib_bbq, calib_enir, calib_ece, calib_mce, calib_ace, calib_nll, calib_pl, calib_picp, calib_qce, calib_ence, calib_uce, calib_metrics, plot_reliability_diagram, my_logit, my_logistic
# from XRAIDashboard.uncertainty.uct import *
from .uncertainty.uct import uct_manipulate_data, uct_get_all_metrics, uct_plot_adversarial_group_calibration, uct_plot_average_calibration, uct_plot_ordered_intervals, uct_plot_XY

# Robustness
from .robustness.art_mia import art_mia, art_generate_predicted, art_generate_actual, calc_precision_recall, mia_viz
from .robustness.art_metrics import pdtp_generate_samples, pdtp_metric, SHAPr_metric, visualisation
from .robustness.art_extra_models import art_extra_classifiers

#ExplainerDashboard
# from XRAIDashboard.eda.dashboard import *
from .eda.dashboard import BlankComponent, AutoVizComponent, EDATab
# from XRAIDashboard.fairness.dashboard import *
from .fairness.dashboard import BlankComponent, FairnessIntroComponent, FairnessCheckRegComponent, FairnessCheckClfComponent, ModelPerformanceRegComponent, ModelPerformanceClfComponent, OutlierComponent, ErrorAnalysisComponent, FairnessTab
# from XRAIDashboard.local_exp.dashboard import *
from .local_exp.dashboard import BlankComponent, EntryIndexComponent, BreakDownComponent, AdditiveComponent, CeterisParibusComponent, InteractiveComponent, DiceExpComponent, QIIExpComponent, ShapWaterfallComponent, ShapForceComponent, ShapBarComponent, LocalExpTab
# from XRAIDashboard.global_exp.dashboard import *
from .global_exp.dashboard import BlankComponent, PartialDependenceProfileComponent, VariableImportanceComponent, LocalDependenceComponent, AccumulatedLocalComponent, CompareProfileComponent, ShapBarGlobalComponent, ShapSummaryComponent, ShapDependenceComponent, ShapForceGlobalComponent, GlobalExpTab
# from XRAIDashboard.stability.dashboard import *
from .stability.dashboard import DataDriftComponent, PSIComponent, AlibiFETComponent, RegressionPerformanceTestComponent, StabilityTab, ClassificationPerformanceComponent, AlibiCVMComponent, KSTestComponent, DataDriftTestComponent, DataQualityComponent, DataQualityTestComponent, TargetDriftComponent, ClassificationPerformanceTestComponent, RegressionPerformanceComponent, BlankComponent, DecileComponent
# from XRAIDashboard.robustness.dashboard import *
from .robustness.dashboard import BlankComponent, ARTPrivacyComponent, ARTInferenceAttackComponent, RobustnessTab
# from XRAIDashboard.uncertainty.dashboard import *
from .uncertainty.dashboard import BlankComponent, CalibrationComponent, AdversarialCalibrationComponent, AverageCalibrationComponent, OrderedIntervalsComponent, XYComponent, UncertaintyTab
# # EDA
# from .eda.auto_eda import *

# # Performance overview 
# from .fairness.fairness import model_performance

# # Fairness
# from .fairness.fairness import fairness
# from .fairness.fairness_algorithm import *
# from .fairness.cluster_metrics import *

# # Local explanation
# from .local_exp.local_exp import *

# # Global explanation
# from .global_exp.global_exp import *

# # Stability
# # from .stability.stability import *
# # from .stability.decile import *

# #Outlier
# from .fairness.outlier import *

# #Uncertainty
# from .uncertainty.calibration import *
# from .uncertainty.uct import *

# #ExplainerDashboard
# # from explainerdashboard import *
# from .eda.dashboard import *
# from .fairness.dashboard import *
# from .local_exp.dashboard import *
# from .global_exp.dashboard import *
# # from .stability.dashboard import *
# from .robustness.dashboard import *
# from .uncertainty.dashboard import *
# # import dash_bootstrap_components as dbc


# from raiwidgets.responsibleai_dashboard import ResponsibleAIDashboard
# from .fairness.XRAI_features import *
# from raiwidgets import ErrorAnalysisDashboard
