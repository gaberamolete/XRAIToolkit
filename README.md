# Running in Local Machine using Conda Environment
To easily create the environment for XRAI in your local machine, you can run the following in a command prompt or terminal.
```
conda env create -f environment.yml
```
The `environment.yml` is the latest environment with no dependency issues, while the `latest_env.yml` has some due to the addition of the Responsible AI library. Some components in the dashboard may not be available from `environment.yml`.

After creating an environment, you can create a kernel to enable the notebooks to use the conda environment using the following lines.
```
python -m ipykernel install --user --name=XRAI
```

# Requirements
```
python = 3.9.0
```
Ensure that you have the correct libraries and dependencies installed by checking `requirements.txt`.


# Play with XRAI tool using Given Data
Assuming the base directories are set correctly, run this in your terminal.

`python download_pima_data_model.py`

Once downloaded, you may run this code block in your Jupyter Notebook.

```py
import build_data_model from download_pima_data_model
build_data_model()
```


# Play with XRAI tool using Custom Data
Run this code block in your Jupyter Notebook. `load_data_model` allows you to load data and models so that our other XRAI functions can intake these files properly.

```py
from data_model import load_data_model

# provide train_data, test_data, model_path and target_feature
train_data = 'train_data.csv'
test_data = 'test_data.csv'
model_path = 'finalized_model.pkl'
target_feature = 'class'

X_train, Y_train, X_test, Y_test, train_data, test_data , model = load_data_model(train_data, test_data, model_path, target_feature)
```
Consider the following points:
- In this version; we require `train_data` and `test_data` in csv format. 
- `train_data` and `test_data` should be in same format. For example, if you have applied label encoding on your categorical variable you should have done the same processing on your same variable in test data.
- The current version works for both categorical and numerical variables, although in future versions we are planning to add more existing features for categorical variables.

# XRAI Tool Features
For Version 1 of the XRAI Toolkit, we have following features incorporated in XRAI Dashboard and Jupyter Notebook:
1. Error Analysis: Identify model errors and discover cohorts of data for which the model underperforms.
2. Model overview: Understands model predictions using various matrices such as accuracy, recall etc. 
3. Data exploration: Find out error segments where model underperforms and then see data exploration/statistics for these segments and many more
4. Feature importance: We have capability to understand global feature importance, individual variable feature importance(group wise for ex. age more than 40 contributes more towards model predictions) or can visualize row wise (customer wise) importance.
5. Fairness analysis.

All the above features can be analyze on entire train or test data. Similarly, you may also define custom segments/groups/cohorts to undergo the same analysis. This may be helpful for analyzing underrepresented or protected groups. 


# Future releases
This is an initial release intended to collect feedback from DSAI/ADI users. We plan to release a 2nd version of the Toolkit (and Guidelines) on March 2023. Stay tuned for the following features and capabilties:
- We aim to give functions that enable better understanding of categorical features, such as __. 
- Regression analysis and Multi-label classification will be compatible for Version 2.
- Additional exciting features like data drift, segment analytics, outlier analysis, stability analysis, and many more! 
