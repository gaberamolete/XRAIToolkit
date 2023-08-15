import plotly.express as px
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.alad import ALAD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.anogan import AnoGAN
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.pca import PCA
from pyod.models.xgbod import XGBOD
from pyod.models.lof import LOF
import numpy as np
import pandas as pd
from pyod.models.suod import SUOD
from pyod.models.copod import COPOD


def outlier(train,test,methods=[KNN,IForest],contamination=0.05):
    
    '''
    Detects outliers and remove them if user wants.
    
    Parameters
    ----------
    train: dataframe, the data set that outlier analysis is going to be done on it. we train detectors on this data set,
    test: data set that you would like to do predection on it
    method: Outlier detection method name options to choose from:
        ABOD: Angle-based Outlier Detector ,
        CBLOF:Clustering Based Local Outlier Factor,
        ALAD: Adversarially Learned Anomaly Detection,
        ECOD:Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions (ECOD),
        IForest:IsolationForest Outlier Detector. Implemented on scikit-learn library,
        AnoGAN: Anomaly Detection with Generative Adversarial Networks,
        KNN:k-Nearest Neighbors Detector,
        KPCA: Kernel Principal Component Analysis (KPCA) Outlier Detector,
        XGBOD:Improving Supervised Outlier Detection with Unsupervised Representation Learning. A semi-supervised outlier detection framework.
        PCA: Principal Component Analysis (PCA) Outlier Detector
    
    
    --------------------------------------------------------------------------
    More info about the https://pyod.readthedocs.io/en/latest/pyod.models.html
    
    contamination: Amount of detected outlier by the method, range is (0-1), the higher the value the stricter the method
    
    
    '''
     
    
    # Clean train for the outlier detector model
    X=train.copy()
    X.dropna(axis='columns',inplace=True)
    X=X._get_numeric_data()
    
    
    # Clean test for the outlier detector model
    XX=test.copy()
    XX.dropna(axis='columns',inplace=True)
    XX=XX._get_numeric_data()
    
    
    #Placeholder for lables
    Labels_train=pd.DataFrame()
    Labels_test=pd.DataFrame()
    
    Methods_name={"ABOD":ABOD,"CBLOF":CBLOF,"ALAD":ALAD,"ECOD":ECOD,"IForest":IForest,"AnoGAN":AnoGAN,"KNN":KNN,"KPCA":KPCA,"XGBOD":XGBOD,"PCA":PCA}
    
    for method in methods:
        #print(method.__name__)
        method=Methods_name[method]
        
        outliers_fraction=contamination
        outlier_method = method(contamination=outliers_fraction)
        outlier_method.fit(X)
        outlier_method.labels_
        
        
       
        #Return the classified inlier/outlier for train
        boolean_lable_train=[bool(a) for a in outlier_method.labels_]
        Labels_train["Outlier_"+method.__name__]=boolean_lable_train
        
        
        #Return the classified inlier/outlier for test
        boolean_lable_test=[bool(a) for a in outlier_method.predict(XX)]
        Labels_test["Outlier_"+method.__name__]=boolean_lable_test
        
   
    return Labels_train, Labels_test




def removal(data,l,major_voting=51):
    
    
    '''
    Remove outliers based on the differnet methods results
    
    Parameters
    ----------
    l: this is a data frame generated by outlier function which contains different methods opinion on each data
    major_voting: a fload number between 0-100%, it decides to remove based on methods opinion. E.g if 50 if 50% or more of methods detect a data as
    outlier it will be removed   
    
    
    return:
    I will return outlier free dataframe.
    
    
    '''
    
    
    # function for calculating the major precent of detected outlier.
    def re(input_list,major_voting):
        p=100*np.sum(input_list==True)/len(input_list)
        
        if p>=major_voting:
            return True
        else:
            return False
    
    ll=l.copy()
    for i in range(len(l)):
        ll.loc[i,"result"]=re(l.loc[i,:],major_voting=major_voting)
    
    inlaier=[not s for s in ll["result"]]

    clean=data.loc[inlaier,:]
    return clean



def visualize(data,l,show=['preg','skin']):
        
    '''
    This function is used to visualize the outlier fro inlaier in each  
    
    Parameters
    ----------
    data= dataframe you need to do outlier analysis on it
    l: this is a dataframe generated by outlier function which contains different methods opinion on each data
    show is list of pairs you would like to see how  outliers are scattered in their 
  
    
    
    return:
    It will return visualizations of outliers.
    

    '''
    
    # list for storing data
    figs=[]
    
    for method in l.columns:
        # Visualization of data points
        X=data.copy()
        X["Outlier"]=l[method]
        
        
        fig = px.scatter(X, x=show[0], y=show[1], color="Outlier",title="Detected Outliers by "+method.split("_")[-1])
        fig.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
        figs.append(fig)
        fig.show()
    return figs
