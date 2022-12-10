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

