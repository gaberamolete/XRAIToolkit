
'''
Sample data: pima indian diabetes
Datatype: all variables are numeric
Task: Classification
'''

# improt required libraries
import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split

def build_data_model():
	'''
	Input: None
	Output: train_data, test_data, target_variable_name and trained_model_file
	'''
	# download and load data
	url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
	names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
	data = pd.read_csv(url, names=names)

	# Create X and Y
	X = data.iloc[:,0:8]
	Y = data.iloc[:,8].tolist()

	# Split data into train and test
	target_feature = 'class'
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
	train_data = X_train.copy()
	test_data = X_test.copy()
	train_data['class'] = y_train
	test_data['class'] = y_test

	# dump datasets : train.csv and test.csv
	train_data.to_csv('train_data.csv', index=False)
	test_data.to_csv('test_data.csv', index=False)

	# build model
	clf = svm.SVC(gamma=0.001, C=100., probability=True)
	model = clf.fit(X_train, y_train)

	# save the model to disk
	filename = 'pima_model.pkl'
	pickle.dump(model, open(filename, 'wb'))

	print('train_data.csv, test_data.csv and pima_model.pkl dumped in the directory')
	return train_data, test_data, model, target_feature


if __name__ == '__main__':
	build_data_model()