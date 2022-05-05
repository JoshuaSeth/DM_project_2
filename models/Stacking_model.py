from json import load
from data.load import load_data
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import std
import time
import matplotlib.pyplot as plt
import copy

from helpers.ranking import to_ranking

# Load data
df = load_data(add_day_parts=True, same_value_operations=[('site_id', 'price_usd', 'avg'), ('srch_id', 'price_usd', 'avg')], fts_operations=[('prop_starrating', 'visitor_hist_starrating', 'diff')], add_seasons=True)




# Split into target and predictors
print('filling na')
y = df['booking_bool']
X = df.drop(['booking_bool','click_bool', 'position', 'gross_bookings_usd', 'date_time'], axis=1)
X = X.fillna(X.mean())
del df



# Set up stakcing regressor
print('traininng')
# rf = RandomForestRegressor()
rf= StackingRegressor([('rf',RandomForestRegressor()), ('br',BayesianRidge()), ('ab',AdaBoostRegressor())], verbose=3, n_jobs=7)
# rf=BayesianRidge()
rf.fit(X_train, y_train)


# Split 
test = load_data(test=True, add_day_parts=True, same_value_operations=[('site_id', 'price_usd', 'avg'), ('srch_id', 'price_usd', 'avg')], fts_operations=[('prop_starrating', 'visitor_hist_starrating', 'diff')], add_seasons=True)

# X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, y_train = X, y
X_test = test.drop('date_time', axis=1)
X_test = X_test.fillna(X.mean()) #Mean of x or mean of x_test?

# Evaluate``
print('testing')
preds = rf.predict(X_test)

# [print(a,b, c) for a,b, c in zip(test['srch_id'], test['prop_id'],preds)]

ranking = to_ranking(X_test, preds)

ranking.to_csv('test_ranking.csv', index=False)

# mse = mean_squared_error(y_test, preds)
# mae = mean_absolute_error(y_test, preds)
# print('\nerrors\n\n', mse, mae)