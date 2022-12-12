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

#Regressió Lineal Multivariable
# Importar las librerias necesarias
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, mean_squared_error

# Crear el modelo de regresión lineal
model = LinearRegression()

# Crear el diccionario de hiperparámetros a probar
param_grid = {
    'fit_intercept': [True, False], # Probar si el modelo debe incluir un término intercepto
    'copy_X': [True, False], # Probar si el modelo debe copiar los datos de entrada
}

# Crear la búsqueda en cuadrícula
grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)

# Entrenar el modelo utilizando la búsqueda en cuadrícula
grid_search.fit(x_train, y_train)

# Imprimir los mejores hiperparámetros encontrados
print('Mejores hiperparámetros:', grid_search.best_params_)


# Calcular el f1-score y el MSE medio para los mejores hiperparámetros
print("La millor score: " + str(grid_search.best_score_))
print("Error quadratic mitja: " + str(math.sqrt(mean_squared_error(y_test,grid_search.predict(x_test)))))


#Regressió d'arbre de decisió
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Define los parámetros que quieres explorar
param_grid = {'max_depth': [1,3,5,10,15,25,30,50],
              'min_weight_fraction_leaf': [0.0,0.1,0.2,0.3,0.4,0.5],
              'max_features': ['auto','log2','sqrt',None],
              'max_leaf_nodes': [None,10,20,50,80,100,250,500]}

# Crea una instancia del modelo DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# Utiliza la búsqueda en cuadrícula (grid search) para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error')

# Entrena el modelo utilizando los hiperparámetros encontrados
grid_search.fit(x_train, y_train)

# Imprime los mejores hiperparámetros encontrados
print("Los mejores hiperparámetros son: {}".format(grid_search.best_params_))

# Calcular el f1-score y el MSE medio para los mejores hiperparámetros
print("La millor score: " + str(grid_search.best_score_))
print("Error quadratic mitja: " + str(math.sqrt(mean_squared_error(y_test,grid_search.predict(x_test)))))

#Regressió d'arbre aleatoria
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define los parámetros que quieres explorar
param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
              'max_features': ['auto', 'sqrt'],
              'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]}

# Crea una instancia del modelo RandomForestRegressor
regressor = RandomForestRegressor()

# Utiliza la búsqueda en cuadrícula (grid search) para encontrar los mejores hiperparámetros
grid_search = RandomizedSearchCV(regressor, param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Entrena el modelo utilizando los hiperparámetros encontrados
grid_search.fit(x_train, y_train)

# Imprime los mejores hiperparámetros encontrados
print("Los mejores hiperparámetros son: {}".format(grid_search.best_params_))

# Calcular el f1-score y el MSE medio para los mejores hiperparámetros
print("La millor score: " + str(grid_search.best_score_))
print("Error quadratic mitja: " + str(math.sqrt(mean_squared_error(y_test,grid_search.predict(x_test)))))

#Regressió Logística
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define los parámetros que quieres explorar
param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'penalty': ['l2'],
              'C': [100, 10, 1.0, 0.1, 0.01]}

# Crea una instancia del modelo LogisticRegression
regressor = LogisticRegression()

# Utiliza la búsqueda en cuadrícula (grid search) para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(regressor, param_grid, cv=5, verbose=2, n_jobs=-1)

# Entrena el modelo utilizando los hiperparámetros encontrados
grid_search.fit(x_train, y_train)

# Imprime los mejores hiperparámetros encontrados
print("Los mejores hiperparámetros son: {}".format(grid_search.best_params_))

# Calcular el f1-score y el MSE medio para los mejores hiperparámetros
print("La millor score: " + str(grid_search.best_score_))
print("Error quadratic mitja: " + str(math.sqrt(mean_squared_error(y_test,grid_search.predict(x_test)))))

#SGD - Descens de Gradient Estocàstic
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor

# Define los parámetros que quieres explorar
param_grid = {'loss': ['squared_error', 'huber'],
              'penalty': ['l2', 'l1', 'elasticnet'],
              'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
              'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

# Crea una instancia del modelo SGDRegressor
regressor = SGDRegressor()

# Utiliza la búsqueda en cuadrícula (grid search) para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error')

# Entrena el modelo utilizando los hiperparámetros encontrados
grid_search.fit(x_train, y_train)

# Imprime los mejores hiperparámetros encontrados
print("Los mejores hiperparámetros son: {}".format(grid_search.best_params_))

# Calcular el f1-score y el MSE medio para los mejores hiperparámetros
print("La millor score: " + str(grid_search.best_score_))
print("Error quadratic mitja: " + str(math.sqrt(mean_squared_error(y_test,grid_search.predict(x_test)))))