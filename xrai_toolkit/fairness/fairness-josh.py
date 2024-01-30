

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

import sys
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from IPython.display import display, HTML
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore")

import scipy

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# Extra options
pd.options.display.max_rows = 30
pd.options.display.max_columns = 25

# Show all code cells output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

#Fairness for Regression
import dalex as dx




def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))





# Function for model performance overview
def model_performance(model,test_x,test_y,train_x,train_y,all_test,all_train, target_feature ,protected_groups, reg=False):
    '''
    Parameters
    ----------
    Model: list of models,model object, can be sklearn, tensorflow, or keras
    test_x: DataFrame,
    test_y: DataFrame or Series, contains target column and list of target variables
    target_feature: str, name of target variable
    protected_groups: dictionary of protected groups and protected category in that group, example: {"LGU" : 'pasay','income_class':"1st" }
    reg: Boolean for model type of Regression
    
    '''
    if not reg:
        
        # Define an empty DataFrame
        DF_test=pd.DataFrame()
        DF_train=pd.DataFrame()

        for pg in protected_groups.keys():
            # Test data
            DF_test[pg]=test_x[pg]
            DF_test["Ground_truth"]=pd.DataFrame(test_y)
            DF_test["Predicted"]=model.predict(test_x)
            # Train data
            DF_train[pg]=train_x[pg]
            DF_train["Ground_truth"]=pd.DataFrame(train_y)
            DF_train["Predicted"]=model.predict(train_x)
            
            
        df_cm_list = []

        ### Train 
        print("Performance on Train data :\n")
        print("Accuracy on test data: ", round(accuracy_score(DF_train["Ground_truth"], DF_train["Predicted"]),3))
        
        #p=sns.pairplot(all_train, hue =target_feature )
        conmat = confusion_matrix(DF_train["Ground_truth"], DF_train["Predicted"])
        val = np.mat(conmat) 
        classnames = list(set(DF_train["Ground_truth"]))
        df_cm = pd.DataFrame(val, index=classnames, columns=classnames,)
        df_cm_list.append(df_cm)

        plt.figure()
        heatmap = sns.heatmap(df_cm, annot=True, fmt='.1f',cmap="Blues")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Model Results')
        plt.show()               
            

        ### Test Data
        print("Performance on test data :\n")
        print("Accuracy on test data: ", round(accuracy_score(DF_test["Ground_truth"], DF_test["Predicted"]),3))
        
        #p=sns.pairplot(all_test, hue =target_feature )
        conmat = confusion_matrix(DF_test["Ground_truth"], DF_test["Predicted"])
        val = np.mat(conmat) 
        classnames = list(set(DF_test["Ground_truth"]))
        df_cm = pd.DataFrame(val, index=classnames, columns=classnames,)
        df_cm_list.append(df_cm)


        plt.figure()
        heatmap = sns.heatmap(df_cm, annot=True, fmt='.1f',cmap="Blues")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(' Model Results')
        plt.show()    

 


        
        for pg in protected_groups.keys():
            ## This function is designed to calculate metrics on different splits of defined protected groupes

            def Result_sum(DF_test,pg):
                datatest={'Split': ['Overall performance'],
                "protected_groups":[pg],
                'Accuracy': [metrics.accuracy_score(DF_test["Ground_truth"],DF_test["Predicted"])],
                 "F1 Score":[metrics.f1_score(DF_test["Ground_truth"],DF_test["Predicted"])],
                 "Precision Score":[metrics.precision_score(DF_test["Ground_truth"],DF_test["Predicted"])],
                 "Recall Score": [metrics.recall_score(DF_test["Ground_truth"],DF_test["Predicted"])] }
                result_sum_test=pd.DataFrame(datatest)


                for i in np.unique(DF_test[pg]):

                    DFI_test=DF_test.groupby(pg).get_group(i)
                    result_sum_test.loc[len(result_sum_test)]=[str(i),pg,metrics.accuracy_score(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.f1_score(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.precision_score(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.recall_score(DFI_test["Ground_truth"],DFI_test["Predicted"])]


                return result_sum_test



            result_sum_test=Result_sum(DF_test,pg)
            result_sum_train=Result_sum(DF_train,pg)
            limit_on_plot_number=5

            number_of_groups=len(np.unique(result_sum_test["Split"]))
            if number_of_groups>5:
                limit_on_plot_number=5
            else:
                limit_on_plot_number=number_of_groups



            # Plot metrics for different groups
            plt.bar(result_sum_test["Split"][:limit_on_plot_number], result_sum_test["Accuracy"][:limit_on_plot_number], color ='blue',width = 0.4,label="Test")
            plt.bar(result_sum_train["Split"][:limit_on_plot_number], result_sum_train["Accuracy"][:limit_on_plot_number], color ='red',width = 0.4, label="Train")
            plt.xlabel(pg)
            plt.ylabel("Accuracy")
            plt.xticks(rotation = 90)
            plt.legend()
            plt.show()

        return df_cm_list, result_sum_train, result_sum_test

    if reg:
        #https://dalex.drwhy.ai/python-dalex-fairness-regression.html
        # Define an empty DataFrame
        DF_test=pd.DataFrame()
        DF_train=pd.DataFrame()

        for pg in protected_groups.keys():
            # Test data
            DF_test[pg]=test_x[pg]
            DF_test["Ground_truth"]=pd.DataFrame(test_y)
            DF_test["Predicted"]=model.predict(test_x)
            # Train data
            DF_train[pg]=train_x[pg]
            DF_train["Ground_truth"]=pd.DataFrame(train_y)
            DF_train["Predicted"]=model.predict(train_x)
     

    
    
    
        result_sum_test=pd.DataFrame()
        result_sum_train=pd.DataFrame()
        
        for pg in protected_groups.keys():
            
            # This function is designed to calculate metrics on different splits of defined protected groupes
            def Result_sum(DF_test,pg):
                datatest={'Split': ['Overall performance'],
                "protected_groups":"-",
                'Mean Absolute Percentage Error': [metrics.mean_absolute_percentage_error(DF_test["Ground_truth"],DF_test["Predicted"])],
                 "Mean Absolute Error":[metrics.mean_absolute_error(DF_test["Ground_truth"],DF_test["Predicted"])],
                 "R2":[metrics.r2_score(DF_test["Ground_truth"],DF_test["Predicted"])],
                 "Max Error": [metrics.max_error(DF_test["Ground_truth"],DF_test["Predicted"])] }
                result_sum_test=pd.DataFrame(datatest)


                for i in np.unique(DF_test[pg]):

                    DFI_test=DF_test.groupby(pg).get_group(i)
                    result_sum_test.loc[len(result_sum_test)]=[str(i),pg,metrics.mean_absolute_percentage_error(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.mean_absolute_error(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.r2_score(DFI_test["Ground_truth"],DFI_test["Predicted"]),metrics.max_error(DFI_test["Ground_truth"],DFI_test["Predicted"])]


                return result_sum_test

            result_sum_test=pd.concat([result_sum_test, Result_sum(DF_test,pg)], axis=0)
            result_sum_train=pd.concat([result_sum_train,Result_sum(DF_train,pg)], axis=0)
            

        print("Overal performence for Train data is :")
        result_sum_train=result_sum_train.drop_duplicates(subset=['Split','protected_groups'], keep='first')
        display(result_sum_train)             

        print("Overal performence for Test data is :")
        result_sum_test=result_sum_test.drop_duplicates(subset=['Split','protected_groups'], keep='first')
        display(result_sum_test)

        return result_sum_test, result_sum_train



def fairness(models,x,y,protected_groups={},metric="DI", threshold=0.8, xextra=False,reg=False,dashboard=True):
    '''
    Parameters
    ----------
    Model: list of models,model object, can be sklearn, tensorflow, or keras
    test_x: DataFrame,
    test_y: DataFrame or Series, contains target column and list of target variables
    target_feature: str, name of target variable
    protected_groups: dictionary of protected groups and protected category in that group, example: {"LGU" : 'pasay','income_class':"1st" }
    reg: Boolean for model type of Regression
    dashboard: bool, if run thru the dashboard
    '''
    if reg==False:
      # function for calculating fainess metrics
        def fairness_metrics(y,y_prime):
            """Calculate fairness for subgroup of population"""

            #Confusion Matrix
            cm=confusion_matrix(list(y),list(y_prime))


            TN, FP, FN, TP = cm.ravel()[0], cm.ravel()[1], cm.ravel()[2], cm.ravel()[3]

            N = TP+FP+FN+TN #Total population
            ACC = (TP+TN)/N #Accuracy
            TPR = TP/(TP+FN) # True positive rate
            FPR = FP/(FP+TN) # False positive rate
            FNR = FN/(TP+FN) # False negative rate
            PPP = (TP + FP)/N # % predicted as positive

            #Recall = TruePositives / (TruePositives + FalseNegatives)
            recall=TP/(TP+FN)

            #Precision = TruePositives / (TruePositives + FalsePositives)
            precision=TP/(TP+FP)

            #F-Measure = (2 * Precision * Recall) / (Precision + Recall)
            F1=(2*recall*precision)/(precision+recall)

            #
            fpr, tpr, thresholds = metrics.roc_curve(y, y_prime)
            AUC=metrics.auc(fpr, tpr)


            #Gini score
            gini=gini_coefficient(np.array(y))


            return np.array([ACC, TPR, FPR, FNR, PPP, recall, precision, F1, AUC, gini ])


        xx=x.copy()
        yy=y.copy()
        for model_name in models.keys():


            print('**************************************Fairness analysis for model {} **************************************'.format(model_name))
            print("\n")
            model=models[model_name]

            x=xx.copy()
            y=yy.copy()
            x["y_prime"]=list(model.predict(x))
            x["y"]=list(y)

            if xextra is not False:
                x=pd.concat([xextra, x], axis=1)



            # Calculating overall Disparity  
            i=1
            fairness_report=pd.DataFrame(columns=["Variable","protected_group","Accuracy","True_positive_rate","False_positive_rate","False_negative_rate","predicted_as_positive","Recall", "Precision", "F1 score", "AUC","Gini"])    
            #Calculate fairness metrics 
            fm = fairness_metrics(x['y'],x["y_prime"])
            a=["Overall",""]
            b=list(fm)
            b=[round(item, 3) for item in b]
            a.extend(b)
            row=a
            fairness_report.loc[0,:]=row


            if metric=="DI":
                #print('''In the US there is a legal precedent to set the cutoff to 0.8. That is the predicted as positive for the normal group must not be less than 80% of that of the protected group.''')

                print('---------------------------------------------------------------------------------------------------------------------')



            for i in protected_groups.keys():

                i=str(i)
                if (x[i].dtypes=="float64") or (x[i].dtypes=="int64"):
                    # Find and mark protected and unprotected
                    x["Prev_"+i]=[1 if (r>=protected_groups[i][0]) and (r<=protected_groups[i][1]) else 0 for r in x[i]]
                if x[i].dtypes=="object":
                    x["Prev_"+i]=[1 if r==protected_groups[i] else 0 for r in x[i]]
                    
                #  make a list of protected features and fairness metrics for each
                a=[i,protected_groups[i]]
                b=list(fairness_metrics(x.loc[x["Prev_"+i]==1,'y'],x.loc[x["Prev_"+i]==1,"y_prime"]))
                b=[round(item, 3) for item in b]
                a.extend(b)
                row=a


                # Add metric lists to the fairness_report data frame to use it later in model artifacts
                fairness_report.loc[i,:]=row

                #Calculate fairness metrics for protected group
                fm_Prev_1 = fairness_metrics(x.loc[x["Prev_"+i]==1,'y'],x.loc[x["Prev_"+i]==1,"y_prime"])[4]
                fm_Prev_0 = fairness_metrics(x.loc[x["Prev_"+i]==0,'y'],x.loc[x["Prev_"+i]==0,"y_prime"])[4]

                #Get ratio of fairness metrics
                fm_ratio = fm_Prev_0/fm_Prev_1

                # Equal opportunity

                """Under equal opportunity we consider a model to be fair if the TPRs of the privileged and unprivileged groups are equal. 
                In practice, we will give some leeway for statistic uncertainty. We can require the differences to be more than a certain cutoff (here 0.8).
                This ensures that the TPR for the unprivileged group is not significantly smaller than for the privileged group."""

                fm_Prev_1 = fairness_metrics(x.loc[x["Prev_"+i]==1,'y'],x.loc[x["Prev_"+i]==1,"y_prime"])[1]
                fm_Prev_0 = fairness_metrics(x.loc[x["Prev_"+i]==0,'y'],x.loc[x["Prev_"+i]==0,"y_prime"])[1]
                EOP=fm_Prev_0/fm_Prev_1


                #Equalized odds

                fm_Prev_1 = fairness_metrics(x.loc[x["Prev_"+i]==1,'y'],x.loc[x["Prev_"+i]==1,"y_prime"])[1]
                fm_Prev_0 = fairness_metrics(x.loc[x["Prev_"+i]==0,'y'],x.loc[x["Prev_"+i]==0,"y_prime"])[1]
                EOD=fm_Prev_0-fm_Prev_1



                if metric=="DI":
                    print("----------------------------------------------------{}---------------------------------------------------------".format(i))
                    print("For the protected feature {} of {} the disparity index is {}".format(i,protected_groups[i],round(fm_ratio,3)))
                    print("\n")
                    d=1-threshold
                    fairness_index=fm_ratio

                    if ((fairness_index<=1+d) and (fairness_index>=1-d)):

                        print('\x1b[6;30;42m'+"The model is fair towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>1+d) and (fairness_index<=1+2*d)):

                        print('\x1b[6;30;42m'+"The model is Partially Advantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>1+2*d)):
                         print('\x1b[6;30;42m'+"The model is Totally Advantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>=1-2*d) and (fairness_index<1-d)):
                         print('\x1b[6;30;41m'+"The model is Partially Disdvantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index<1-2*d)):
                         print('\x1b[6;30;41m'+"The model is Totally Disadvantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')     



                elif metric=="EOP":
                    print("----------------------------------------------------{}---------------------------------------------------------".format(i))
                    print("For the protected feature {} of {} the Equal opportunity is {}".format(i,protected_groups[i],round(EOP,3)))
                    print("\n")
                    fairness_index=EOP

                    d=1-threshold

                    if ((fairness_index<=1+d) and (fairness_index>=1-d)):

                        print('\x1b[6;30;42m'+"The model is fair towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>1+d) and (fairness_index<=1+2*d)):

                        print('\x1b[6;30;42m'+"The model is Partially Advantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>1+2*d)):
                         print('\x1b[6;30;42m'+"The model is Totally Advantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>=1-2*d) and (fairness_index<1-d)):
                         print('\x1b[6;30;41m'+"The model is Partially Disdvantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index<1-2*d)):
                         print('\x1b[6;30;41m'+"The model is Totally Disadvantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')     

                elif metric=="EOD":
                    print("----------------------------------------------------{}---------------------------------------------------------".format(i))
                    print("For the protected feature {} of {} the Equalized odds is {}".format(i,protected_groups[i],round(EOD,3)))
                    print("\n")
                    fairness_index=EOD

                    d=1-threshold

                    if ((fairness_index<=1+d) and (fairness_index>=1-d)):

                        print('\x1b[6;30;42m'+"The model is fair towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>1+d) and (fairness_index<=1+2*d)):

                        print('\x1b[6;30;42m'+"The model is Partially Advantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>1+2*d)):
                         print('\x1b[6;30;42m'+"The model is Totally Advantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index>=1-2*d) and (fairness_index<1-d)):
                         print('\x1b[6;30;41m'+"The model is Partially Disdvantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')

                    elif ((fairness_index<1-2*d)):
                         print('\x1b[6;30;41m'+"The model is Totally Disadvantages towards the {} in {}".format(protected_groups[i],i)+ '\x1b[0m')     



                plt.scatter(a[2], fairness_index, label=model_name+'_'+str(i))
            display(fairness_report)


        print("\n\n")    
        plt.title('Fairness VS accuracy trade off')
        plt.xlabel('Accuracy')
        plt.ylabel('Fairness {}'.format(metric))
        plt.legend()
        plt.show()
        
        return fairness_index, fairness_report, a
        
    if reg!=False:
        #Data 
        xx=x.copy()
        yy=y.copy()
        
        if xextra is not False:
            x=pd.concat([xextra, x], axis=1)
        '''
        explainers={}
        Expname=[]
        for model_name in models.keys():
            #
            expname='exp_'+model_name
            Expname.append(expname)
            explainers[expname]=dx.Explainer(models[model_name], x, y, verbose=False)
            #print(explainers[expname])
            
            Fobjects={}
            Fnames=[]
            for pg in protected_groups.keys():
                
                protected = np.where(xx[pg] == protected_groups[pg], pg+"_"+protected_groups[pg], "else")
                privileged = 'else'
                fnames='fobject'+'_'+pg
                Fnames.append(fnames)
                
                Fobjects[fnames]=explainers[expname].model_fairness(protected, privileged)
                
                #Fobjects[fnames].fairness_check()
                #Fobjects[fnames].plot(type='density')
         '''
        
        Fobjects={}
        Fnames=[]
        fig1_list = []
        fig2_list = []
        contents_list = []  
        for pg in protected_groups.keys():  
                
            explainers={}
            Expname=[]
            for model_name in models.keys():
                
                expname='exp_'+str(model_name)
                Expname.append(expname)
                explainers[expname]=dx.Explainer(models[model_name], x, y, verbose=False)
                
                
                #Define protected group
                if (xx[pg].dtypes=="float64") or (xx[pg].dtypes=="int64"):
                    protected = np.where(np.logical_and((xx[pg] >= protected_groups[pg][0]), (xx[pg] <= protected_groups[pg][1])), str(pg)+"_"+str(protected_groups[pg]), "else")
                if xx[pg].dtypes=="object":
                    protected = np.where(xx[pg] == str(protected_groups[pg]), str(pg)+"_"+str(protected_groups[pg]), "else")
                
                privileged = 'else'
                fnames='fobject'+'_'+str(pg)
                Fnames.append(fnames)
                
                Fobjects[expname]=explainers[expname].model_fairness(protected, privileged)
            
            if len(Expname)==2:
                if dashboard == False:
                    print("-------------------"+"Model : "+Expname[0].split("_")[1]+"---- Protected group: "+pg+"--------------------------")
                    Fobjects[Expname[0]].fairness_check()
                    print("-------------------"+"Model : "+Expname[1].split("_")[1]+"---- Protected group: "+pg+"--------------------------")
                    Fobjects[Expname[1]].fairness_check()

                    Fobjects[Expname[0]].plot(Fobjects[Expname[1]],type='density')
                    Fobjects[Expname[0]].plot(Fobjects[Expname[1]])
                else: 
                    fig1 = Fobjects[Expname[0]].plot(Fobjects[Expname[1]],type='density', show=False)
                    fig2 = Fobjects[Expname[0]].plot(Fobjects[Expname[1]], show=False)
                    fig1_list.append(fig1)
                    fig2_list.append(fig2)
                    with open("temp.log", "w") as f:
                        with redirect_stdout(f):
                            print("-------------------"+"Model : "+Expname[0].split("_")[1]+"---- Protected group: "+pg+"--------------------------")
                            Fobjects[Expname[0]].fairness_check()
                            print("-------------------"+"Model : "+Expname[1].split("_")[1]+"---- Protected group: "+pg+"--------------------------")
                            Fobjects[Expname[1]].fairness_check()
                    with open("temp.log") as f:
                        contents = f.readlines()
                    contents_list.append(contents)

                
            elif len(Expname)==1:
                if dashboard == False:
                    print("-------------------"+"Model : "+Expname[0].split("_")[1]+"---- Protected group: "+pg+"--------------------------")
                    Fobjects[Expname[0]].fairness_check()
                    Fobjects[Expname[0]].plot(type='density')
                    Fobjects[Expname[0]].plot()
                else:
                    fig1 = Fobjects[Expname[0]].plot(type='density', show=False)
                    fig2 = Fobjects[Expname[0]].plot(show=False)
                    with open("temp.log", "w") as f:
                        with redirect_stdout(f):
                            print("-------------------"+"Model : "+Expname[0].split("_")[1]+"---- Protected group: "+pg+"--------------------------")
                            Fobjects[Expname[0]].fairness_check()
                    with open("temp.log") as f:
                        contents = f.readlines()
                    contents_list.append(contents)
                    fig1_list.append(fig1)
                    fig2_list.append(fig2)
                
        return contents_list, fig1_list, fig2_list,

            

            

