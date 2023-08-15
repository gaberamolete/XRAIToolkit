'''
Prepare dataset for Toolkit
Input: train_data path, test_data path, model_path and target_feature
Output: train_x, train_y, test_x, test_y, train_data, test_data and loaded model
'''

# import required libraries
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
import joblib

def load_data_model(train_data, test_data, model_path, target_feature):
    '''
    input: train_data path, test_data path, model_path and target_feature
    Output: train_x, train_y, test_x, test_y, train_data, test_data and loaded model
    '''
    # import train and test data
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    models={}
    for m_name in model_path.keys():
        # load the model from the disk
        input_file = open(model_path[m_name], "rb")
        models[m_name] = joblib.load(input_file)
    #model = pickle.load(open(model_path, 'rb'))
    
    # generate train_x, train_y, test_x, test_y, using train_data and test_data
    train_y=train_data[target_feature]
    test_y=test_data[target_feature]
    train_x=train_data.drop(target_feature ,axis=1)
    test_x=test_data.drop(target_feature ,axis=1)
    
    return train_x, train_y, test_x, test_y, train_data, test_data , models