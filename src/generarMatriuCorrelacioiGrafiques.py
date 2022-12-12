import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error


import math

dataset = pd.read_csv("../BBDD/games.csv")

dataset = dataset.drop_duplicates()

dataset['winner'] = dataset['winner'].replace(['white', 'black', 'draw'], [0, 1, 2])
dataset = dataset[dataset['winner'] != 2]

#fem la matriu de correlació
corr_matrix = dataset.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

dataset = dataset[["turns","white_rating","black_rating","opening_ply","winner"]]
dataset = dataset.values
for i in range(1,5):
    ax = plt.subplot(2,3,i)
    ax.scatter(dataset[:,i-1], dataset[:,-1])
    