from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import catboost as cb
import pickle
import os
import gc
gc.enable()

def reduce_mem_usage(df, verbose=True):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2    
	for col in df.columns:
		col_type = df[col].dtypes
		if col_type in numerics:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)    
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df



# logistic regression
def fit_logistic(X, y, lf_path):

	# set the hyperparameters
	param_grid = dict(C = [0.0001,0.001,0.01,0.1,10], penalty = ['l2', 'l1'])

	# call the model
	clf = LogisticRegression()

	# perform the randomized grid search
	grid_search = RandomizedSearchCV(clf, param_grid, cv = 3, random_state = 0, score = 'roc_auc', refit = True)
	grid_search.fit(X, y)

	# save the best model
	model = grid_search.best_estimator_

	prediction = model.predict_proba(X)[:,1]	
    
    # save logistic regression model
    save_to = '{}{}.pkl'.format(lf_path, 'lf')
	dump(model, open(save_to, 'wb'))

	# return the probability predictions of the model
	return prediction

# light gbm
def fit_lgb(X, y, lgb_path):
    
	# set the hyperparameters
	# Define the search space

	param_test ={'num_leaves': sp_randint(6, 50), 
	'min_child_samples': sp_randint(100, 500), 'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
	'subsample': sp_uniform(loc=0.2, scale=0.8), 
	'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
	'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
	'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

	n_HP_points_to_test = 100

	# call the model
	clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='auc', objective = 'binary', n_jobs=-1, n_estimators=5000)
	
	# perform the randomized grid search
	grid_search = RandomizedSearchCV(estimator=clf, param_distributions=param_test, n_iter=n_HP_points_to_test, scoring='roc_auc',
		cv=3, refit=True, random_state=314, verbose=True)
	grid_search.fit(X, y)

	# save the best model
	model = grid_search.best_estimator_
                  
    prediction = model.predict_proba(X)[:,1]
    
    #Save LightGBM Model
    save_to = '{}{}.txt'.format(lgb_path, 'lgb')
    model.booster_.save_model(save_to)
    
    return prediction

def fit_cb(X, y, cb_path):
    
    model = cb.CatBoostClassifier(iterations=999999,
                                  max_depth=2,
                                  learning_rate=0.02,
                                  colsample_bylevel=0.03,
                                  objective="Logloss")
                                  
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, early_stopping_rounds=1000)

    grid = {'learning_rate': [0.03, 0.001, 0.1, 0.5],
        'depth': [2, 4, 6, 10, 15, 25, 50, 70],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}

    grid_search_result = model.grid_search(grid, X=X, y=y)
    
    prediction = model.predict_proba(X)[:,1]
    
    #Save Catboost Model          
    save_to = "{}{}.mlmodel".format(cb_path, 'catboost')
    model.save_model(save_to, format="coreml", export_parameters={'prediction_type': 'probability'})
                     
    return prediction