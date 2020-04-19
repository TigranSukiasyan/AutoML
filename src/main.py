from classes import *
from datetime import datetime
import os

# load and split the data
init = LoadSplit()
X_train, X_test, y_train, y_test, problem_type = init.ls()

# Data Preprocessing Block
X_train = DataPreprocessor(db = X_train, data_preprocessing = 'on')

# model training based on problem_type
if problem_type == 'regression':

	# call the regressor
    rgr = Regressor()
    
    
    # get the reports
    
else:
    # call the classifier class

    # get the reports


if __name__ == '__main__':

	today = datetime.now()

	if today.hour < 12:

		h = "00"

	else:
		h = "12"

path = "../mdir/" + today.strftime('%Y%m%d') + h
os.mkdir(path)
os.chdir(path)