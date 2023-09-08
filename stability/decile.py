import numpy as np
import pandas as pd
import plotly.graph_objects as go

from colorama import Fore
from typing import Union, Iterable, List, Dict

def print_labels():
    """
    Labels for decile table.
    """
    
    print(
        "LABELS INFO:\n\n",
        "prob_min         : Minimum probability in a particular decile\n", 
        "prob_max         : Minimum probability in a particular decile\n",
        "prob_avg         : Average probability in a particular decile\n",
        "cnt_events       : Count of events in a particular decile\n",
        "cnt_resp         : Count of responders in a particular decile\n",
        "cnt_non_resp     : Count of non-responders in a particular decile\n",
        "cnt_resp_rndm    : Count of responders if events assigned randomly in a particular decile\n",
        "cnt_resp_wiz     : Count of best possible responders in a particular decile\n",
        "resp_rate        : Response Rate in a particular decile [(cnt_resp/cnt_cust)*100]\n",
        "cum_events       : Cumulative sum of events decile-wise \n",
        "cum_resp         : Cumulative sum of responders decile-wise \n",
        "cum_resp_wiz     : Cumulative sum of best possible responders decile-wise \n",
        "cum_non_resp     : Cumulative sum of non-responders decile-wise \n",
        "cum_events_pct   : Cumulative sum of percentages of events decile-wise \n",
        "cum_resp_pct     : Cumulative sum of percentages of responders decile-wise \n",
        "cum_resp_pct_wiz : Cumulative sum of percentages of best possible responders decile-wise \n",
        "cum_non_resp_pct : Cumulative sum of percentages of non-responders decile-wise \n",
        "KS               : KS Statistic decile-wise \n",
        "lift             : Cumuative Lift Value decile-wise",
         )



def decile_table(y_true, y_prob, change_deciles = 10, labels = True, round_decimal = 5):
    """
    Generates the Decile Table from labels and probabilities.
    
    The Decile Table is creared by first sorting the rows by their predicted 
    probabilities, in decreasing order from highest (closest to one) to 
    lowest (closest to zero). Splitting the rows into equally sized segments, 
    we create groups containing the same numbers of rows, for example, 10 decile 
    groups each containing 10% of the row base.
    
    Parameters
    ------------
    y_true: Array or Series with ground truth labels.
    y_prob: Array or series of predicted values.
    change_deciles: int, the number of partitions for creating the table can be changed. Defaults to 10.
    labels: bool. If True, prints a legend for the abbreviations of decile table column names. Defaults to True.
    round_decimal (int, optional): The decimal precision till which the result is needed. Defaults to 5.

    Returns
    ------------
    dt: The dataframe dt (decile-table) with the deciles and related information.
    """
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_prob'] = y_prob
    # df['decile']=pd.qcut(df['y_prob'], 10, labels=list(np.arange(10,0,-1))) 
    # ValueError: Bin edges must be unique

    df.sort_values('y_prob', ascending=False, inplace=True)
    df['decile'] = np.linspace(1, change_deciles+1, len(df), False, dtype=int)

    # dt abbreviation for decile_table
    dt = df.groupby('decile').apply(lambda x: pd.Series([
        np.min(x['y_prob']),
        np.max(x['y_prob']),
        np.mean(x['y_prob']),
        np.size(x['y_prob']),
        np.sum(x['y_true']),
        np.size(x['y_true'][x['y_true'] == 0]),
    ],
        index=(["prob_min", "prob_max", "prob_avg",
                "cnt_cust", "cnt_resp", "cnt_non_resp"])
    )).reset_index()

    dt['prob_min']=dt['prob_min'].round(round_decimal)
    dt['prob_max']=dt['prob_max'].round(round_decimal)
    dt['prob_avg']=round(dt['prob_avg'],round_decimal)
    # dt=dt.sort_values(by='decile',ascending=False).reset_index(drop=True)

    tmp = df[['y_true']].sort_values('y_true', ascending=False)
    tmp['decile'] = np.linspace(1, change_deciles+1, len(tmp), False, dtype=int)

    dt['cnt_resp_rndm'] = np.sum(df['y_true']) / change_deciles
    dt['cnt_resp_wiz'] = tmp.groupby('decile', as_index=False)['y_true'].sum()['y_true']

    dt['resp_rate'] = round(dt['cnt_resp'] * 100 / dt['cnt_cust'], round_decimal)
    dt['cum_cust'] = np.cumsum(dt['cnt_cust'])
    dt['cum_resp'] = np.cumsum(dt['cnt_resp'])
    dt['cum_resp_wiz'] = np.cumsum(dt['cnt_resp_wiz'])
    dt['cum_non_resp'] = np.cumsum(dt['cnt_non_resp'])
    dt['cum_cust_pct'] = round(dt['cum_cust'] * 100 / np.sum(dt['cnt_cust']), round_decimal)
    dt['cum_resp_pct'] = round(dt['cum_resp'] * 100 / np.sum(dt['cnt_resp']), round_decimal)
    dt['cum_resp_pct_wiz'] = round(dt['cum_resp_wiz'] * 100 / np.sum(dt['cnt_resp_wiz']), round_decimal)
    dt['cum_non_resp_pct'] = round(
        dt['cum_non_resp'] * 100 / np.sum(dt['cnt_non_resp']), round_decimal)
    dt['KS'] = round(dt['cum_resp_pct'] - dt['cum_non_resp_pct'], round_decimal)
    dt['lift'] = round(dt['cum_resp_pct'] / dt['cum_cust_pct'], round_decimal)

    if labels is True:
        print_labels()
        
    # Display KS
    print(Fore.RED + "KS is " + str(max(dt['KS']))+"%"+ " at decile " + str((dt.index[dt['KS']==max(dt['KS'])][0] + 1)))
    ks = "KS is " + str(max(dt['KS']))+"%"+ " at decile " + str((dt.index[dt['KS']==max(dt['KS'])][0] + 1))

    return dt, ks

def model_selection_by_gain_chart(model_dict):
    """
    Gives a Gain Plotly figure.
    
    Parameters
    ------------
    model_dict: Dict, dictionary with key-pair values of model name and Array or Series of predicted values from said model.
    """
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(0,100+10,10)), y=list(range(0,100+10,10)),
                    mode='lines+markers',name='Random Model'))
    for model_name,dc in model_dict.items():
        # dc.insert(0,0)
        fig.add_trace(go.Scatter(x=list(range(0,100+10,10)), y=pd.concat([pd.Series(0), dc['cum_resp_pct']]),
                    mode='lines+markers',name=model_name))
    fig.update_xaxes(
        title_text = "% of Data Set",)

    fig.update_yaxes(title_text = "% of Gain",)
    fig.update_layout(title='Gain Charts',)
    fig.show()

    return fig
    
def model_selection_by_lift_chart(model_dict):
    """
    Gives a Lift Plotly figure.
    
    Parameters
    ------------
    model_dict: Dict, dictionary with key-pair values of model name and Array or Series of predicted values from said model.
    """
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(10,100+10,10)), y=np.repeat(1,10),
                    mode='lines+markers',name='Random Lift'))
    for model_name,dc in model_dict.items():
        fig.add_trace(go.Scatter(x=list(range(10,100+10,10)), y=dc['lift'],
                    mode='lines+markers',name=model_name))
    fig.update_xaxes(
        title_text = "% of Data Set",)

    fig.update_yaxes(title_text = "Lift",)
    fig.update_layout(title='Lift Charts',)
    fig.show()

    return fig
    
def model_selection_by_lift_decile_chart(model_dict):
    """
    Gives a Lift Plotly figure specific to decile.
    
    Parameters
    ------------
    model_dict: Dict, dictionary with key-pair values of model name and Array or Series of predicted values from said model.
    """
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(10,100+10,10)), y=np.repeat(1,10),
                    mode='lines+markers',name='Random Lift'))
    for model_name, dc in model_dict.items():
        fig.add_trace(go.Scatter(x=list(range(10,100+10,10)), y= dc['cnt_resp'] / dc['cnt_resp_rndm'],
                    mode='lines+markers',name=model_name))
    fig.update_xaxes(
        title_text = "% of Data Set",)

    fig.update_yaxes(title_text = "Lift",)
    fig.update_layout(title='Lift Decile Charts',)
    fig.show()

    return fig
    
def model_selection_by_ks_statistic(model_dict):
    """
    Gives a KS-statistic Plotly figure.
    
    Parameters
    ------------
    model_dict: Dict, dictionary with key-pair values of model name and Array or Series of predicted values from said model.
    """
    
    fig = go.Figure()
    for model_name, dc in model_dict.items():
        fig.add_trace(go.Scatter(x=list(range(10,100+10,10)), y= dc['cum_resp_pct'],
                    mode='lines+markers',name=f'{model_name} - Responders'))
        fig.add_trace(go.Scatter(x=list(range(10,100+10,10)), y= dc['cum_non_resp_pct'],
            mode='lines+markers',name=f'{model_name} - Non-Responders'))
        ksmx = dc['KS'].max()
        ksdc1 = dc[dc['KS'] == ksmx]['decile'].values
        fig.add_trace(go.Scatter(x = [ksdc1, ksdc1], y = [dc[dc['KS'] == ksmx]['cum_resp_pct'],
                                                         dc[dc['KS'] == ksmx]['cum_non_resp_pct']],
                                name = f'{model_name} - KS Statistic: {ksmx} at decile {ksdc1[0]}', mode = 'text'))
    fig.update_xaxes(
        title_text = "% of Data Set",)

    fig.update_yaxes(title_text = "KS",)
    fig.update_layout(title='KS-Statistic Charts',)
    fig.show()

    return fig
    
def decile_report(y_true, prob_dict: Dict):
    """
    Gives the decile report and Plotly figures for multiple models.
    
    Parameters
    ------------
    y_true: Array or Series with ground truth labels.
    prob_dict: Dict, dictionary with key-pair values of model name and Array or Series of predicted values from said model.
    
    Returns
    ------------
    dcs: Dictionary of decile reports tagged to model name. 
    """
    
    dcs = {}
    for model_name, prob in prob_dict.items():
        dc = decile_table(y_true, prob)
        print(dc)
        
        dcs[model_name] = dc
    
    model_selection_by_gain_chart(dcs)
    model_selection_by_lift_chart(dcs)
    model_selection_by_lift_decile_chart(dcs)
    model_selection_by_ks_statistic(dcs)
    
    return dcs