import pandas as pd
import seaborn as sns
import dtale
import autoviz
from ydata_profiling import ProfileReport ## COMMENTED OUT
from autoviz.AutoViz_Class import AutoViz_Class

#output in notebook
def dtale_eda(dataframe):
    d = dtale.show(dataframe)
    return d

# #output in browser
# def dtale_eda2(dataframe):
#     d = dtale.show(dataframe)
#     d.open_browser()

# #output in noteboook
# def ydata_profiling_eda(dataframe):
#     profile = ProfileReport(dataframe, title='Pandas Profiling Report')
#     return profile.to_widgets()

# output as html
def ydata_profiling_eda2(dataframe): ## COMMENTED OUT
    profile = ProfileReport(dataframe, title='Pandas Profiling Report')
    return profile.to_file("assets/your_report.html")

# #output in notebook
# def autoviz_eda(dataframe):
#     AV = AutoViz_Class()
#     AV.AutoViz(dataframe, chart_format='bokeh')

#output in browser
def autoviz_eda2(dataframe):
    AV = AutoViz_Class()
    AV.AutoViz("",dfte=dataframe, chart_format='html')
        