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

dataset = dataset[["turns","white_rating","black_rating","opening_ply","winner"]]
dataset = dataset.values

scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

x = dataset[:,:4]
y = dataset[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#Regressió lineal multivariable
linear_reg = LinearRegression()

linear_reg.fit(x_train, y_train)

prediccions = linear_reg.predict(x_test)
print("Score: " + str(linear_reg.score(x_test, y_test)))
print("MSE: " + str(mean_squared_error(y_test, prediccions)))

plt.figure()
ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediccions, 'r')

#Regressió d'arbre de decisió
decision_tree = DecisionTreeRegressor()

decision_tree.fit(x_train, y_train)

print("Profunditat del arbre: " + str(decision_tree.get_depth()))
print("Numero de fulles del arbre: " + str(decision_tree.get_n_leaves()))

prediccions_arbre = decision_tree.predict(x_test)
print("Score: " + str(decision_tree.score(x_test, y_test)))
print("MSE: " + str(mean_squared_error(y_test, prediccions_arbre)))

ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediccions_arbre, 'r')

#Regressió d'arbre aleatori
random_tree = RandomForestRegressor()

random_tree.fit(x_train, y_train)

prediccions_randtree = random_tree.predict(x_test)
print("Score: " + str(random_tree.score(x_test, y_test)))
print("MSE: " + str(mean_squared_error(y_test, prediccions_randtree) ))

ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediccions_randtree, 'r')

#Regressió Logística
random_tree = LogisticRegression()

random_tree.fit(x_train, y_train)

prediccions_randtree = random_tree.predict(x_test)
print("Score: " + str(random_tree.score(x_test, y_test)))
print("MSE: " + str(mean_squared_error(y_test, prediccions_randtree) ))

ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediccions_randtree, 'r')

#SGD - Descens de Gradient Estocàstic
from sklearn.linear_model import SGDClassifier
random_tree = SGDClassifier()

random_tree.fit(x_train, y_train)

prediccions_randtree = random_tree.predict(x_test)
print("Score: " + str(random_tree.score(x_test, y_test)))
print("MSE: " + str(mean_squared_error(y_test, prediccions_randtree) ))

ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediccions_randtree, 'r')