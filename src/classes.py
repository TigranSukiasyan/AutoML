from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from func import *

class UserParams():

	""" Args:
        * dataset_path (str)
        * target_name (str)
        * train_test_path (str)
        ** test_size = 0.2 (int or float)
        ** problem_type = 'auto' (str)
    """
  
    # assign the parameter values
	def __init__(self, dataset_path = 'C:\\Users\\Tigran\\Desktop\\AutoML\\train.csv', target_name = 'price_doc', 
		train_test_path = 'C:\\Users\\Tigran\\Desktop\\AutoML\\tests\\', test_size = 0.2, problem_type='auto'):

		self.dataset_path = dataset_path
		self.target_name = target_name
		self.train_test_path = train_test_path
		self.test_size = test_size
		self.problem_type = problem_type

class LoadSplit(UserParams):

	def __init__(self):
		super().__init__()

	def ls(self):

		db = reduce_mem_usage(pd.read_csv(self.dataset_path))
		#db = pd.read_csv(self.dataset_path)
		# data confirmation

		# drop NAs from the target
		db.dropna(subset = [self.target_name], inplace = True)
		# define the problem type
		# find out whether the target is numeric or categorical
		
		if self.problem_type == 'auto':
			# n is the number of unique values to look for in the target for defining the problem type
			n = 10
			if is_string_dtype(db[self.target_name]) == False and len(db[self.target_name].unique()) > n:
				problem_type_inferred = 'regression'
				print("The inferred problem type is:", "", problem_type_inferred)
			else:
				problem_type_inferred = 'classification'
				print("The inferred problem type is:", "", problem_type_inferred)
		else:
			problem_type_inferred = self.problem_type

		print(db.info(verbose = False)) # can be set to True in the future
		print('***** Please confirm that the above information about the dataset is correct *****')

		val = input('Type "confirm" for confirmation')
		if val == 'confirm':
			y = db.pop(self.target_name)
			X_train, X_test, y_train, y_test = train_test_split(db, y, test_size = self.test_size, random_state = 42)
		else:
			print('please review the dataset')
			exit()

		# save the train and split datasets
		X_train.to_csv(self.train_test_path + 'X_train.csv', index = False)
		X_test.to_csv(self.train_test_path + 'X_test.csv', index = False)
		y_train.to_csv(self.train_test_path + 'y_train.csv', index = False)
		y_test.to_csv(self.train_test_path + 'y_test.csv', index = False)

		print("train/test split is done!")
		return X_train, X_test, y_train, y_test, problem_type_inferred

def DataPreprocessor(db, data_preprocessing = 'on'):

	if data_preprocessing == 'on':

		# fills the numeric features
		db.fillna(db.median(), inplace = True)

		# save the mean values
		db.median().to_pickle('preprocessors\\median.pkl')

		# encodes the categorical features
		encoder = ce.OrdinalEncoder()
		encoder = encoder.fit(db)
		db = encoder.transform(db)

		# a condition for feature scaling
		scaler = MinMaxScaler()
		scaler.fit(db)
		db = pd.DataFrame(scaler.transform(db))

		# save the preprocessors
		# encoder
		dump(encoder, open('preprocessors\\encoder.pkl', 'wb'))
		# scaler
		dump(scaler, open('preprocessors\\scaler.pkl', 'wb'))

		return db

	else:
		return db

