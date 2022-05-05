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

# Load data
# df = load_data(add_day_parts=True, same_value_operations=[('site_id', 'price_usd', 'avg'), ('srch_id', 'price_usd', 'avg')], fts_operations=[('prop_starrating', 'visitor_hist_starrating', 'diff')], add_seasons=True)

df = load_data(num_rows=10000)

# Split into target and predictors
y = df['booking_bool']
X = df.drop('booking_bool', axis=1)
X = X.fillna(X.mean())

# Split 
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Normalize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.drop('date_time', axis=1))
X_test = scaler.transform(X_test.drop('date_time', axis=1))

# Set up stakcing regressor
rf = StackingRegressor([('rf',RandomForestRegressor()), ('br',BayesianRidge()), ('ab',AdaBoostRegressor())])
# rf=BayesianRidge()
rf.fit(X_train, y_train)

# Evaluate
preds = rf.predict(X_test)

[print(a,b) for a,b in zip(y_test, preds)]

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print('\nerrors\n\n', mse, mae)