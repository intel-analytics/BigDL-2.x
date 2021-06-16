# Necessary imports
import numpy as np
import pandas as pd

  
 
# load and summarize the housing dataset
from pandas import read_csv
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
M, N = data.shape

# split dataset into input and output columns
X, y = data[:, :-1], data[:, -1]
# define model
# train_X, test_X, train_y, test_y = train_test_split(X, y,
                    #   test_size = 0.3, random_state = 123)
train_X = data[:(int)(0.8*N), :-1 ]
test_X = data[(int)(0.8*N):, :-1]
train_y = data[:(int)(0.8*N), -1]
test_y = data[(int)(0.8*N):, -1]
model = xgboost.XGBRegressor()
# fit model
model.fit(train_X, train_y)
# define new data
# row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
# new_data = np.asarray([row])
# # make a prediction
yhat = model.predict(test_X)
print(yhat, test_y)
# summarize prediction
# print('Predicted: %.3f' % yhat)