# KaggleChessGame
Repositori de l'estudi del Kaggle Chess Game Dataset (Lichess)

Link al Kaggle: https://www.kaggle.com/datasets/datasnaek/chess/code?datasetId=2321

### Author: Ferran Bernabé Puigmarti

## Abstract
The Lichess Chess Game Dataset is a valuable resource for anyone interested in analyzing and improving their chess game. The dataset, which can be downloaded from Kaggle, contains information on chess games played on the Lichess.org platform.

Each entry in the dataset includes information on the moves made in a given game, as well as details on the players involved and the outcome of the game. This information can be used in a variety of ways, such as analyzing common moves in won and lost games to identify patterns and improve strategy, or studying the characteristics of successful players to identify traits that may contribute to their success.

In addition to its potential use in analysis, the Lichess Chess Game Dataset could also be used to train machine learning models to predict the outcome of a game based on the moves made so far. This could be a useful tool for players looking to improve their game by learning from the strategies and tactics of others.

Overall, the Lichess Chess Game Dataset is a valuable resource for anyone interested in improving their chess skills and understanding the game at a deeper level. Whether you're an experienced player looking to refine your strategies, or a beginner looking to learn from the best, this dataset provides a wealth of information that can help you take your chess game to the next level.

## Objective of the study
Some potential uses for the dataset include:

  · Analyzing common moves and strategies used in won and lost games to identify patterns and improve game strategy.
  
  · Studying the characteristics of successful players to identify traits that may contribute to their success.
  
  · Training machine learning models to predict the outcome of a game based on the moves made so far.
  
Our main objective in this study is to analyze how the winner variable is affected by some of the other numeric variables, founding the best learnig machine model to predict this variable.

## Analyzing the dataset

### Preparing the dataset

First of all, we look the initial dataset (2 entries):

![image](https://user-images.githubusercontent.com/57755230/206905194-7e09ba59-a636-404f-8171-e855acb9adf7.png)

Now, we look the correlation between all the variables, focusing on our objective variable ("winner"):

![matriuCorrelacio](https://user-images.githubusercontent.com/57755230/206905283-07586328-5bd1-472e-9be1-45858aeda26c.png)

Looking at the matrix, we can see that no variables have a high correlation with our objective variable.

Now, we focus on preparing the dataset with the numeric variables, like I said before. Here there's the describe of the dataset:

![image](https://user-images.githubusercontent.com/57755230/206905440-c772d6c7-884e-4e60-8e5e-ee8a127ee433.png)

Finally, I modified the dataset dropping the "draw" results of our main variable to convert it as a binary variabe.

### Computational learning models

In this study, I decided to apply the following computational learning models:

  · Multivariable Lineal Regression
  
  · Decision Tree Regression
  
  · Random Forest Classifier
  
  · Logistic Regression
  
  · SGD - stochastic gradient descent

Applying a training set of the 70% of the dataset I had the following results (applying the default models):

| Model | Score | MSE |
|----------|----------|----------|
| Multivariable Lineal Regression   | 0.139  | 0.215   |
| Decision Tree Regression   |  -0.528  | 0.381   |
| Random Forest Classifier   | 0.159  | 0.210   |
| Logistic Regression   | 0.650   | 0.350   |
| SGD - stochastic gradient descent   | 0.632   | 0.268   |

### Searching the best hyperparameters

Finally, I tried to search the best hiperparameters for each model and calculating the best Score and the MSE mean. In the following table you can see the results:

| Model | Score | MSE | Best hiperparameters |
|----------|----------|----------|----------|
| Multivariable Lineal Regression   | 0.133  | 0.463   | {'copy_X': True, 'fit_intercept': True} |
| Decision Tree Regression   |  0.126  | 0.472   | {'max_depth': 30, 'max_features': 'log2', 'max_leaf_nodes': 50, 'min_weight_fraction_leaf': 0.0} |
| Random Forest Classifier   | 0.173  | 0.446   | {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 110, 'bootstrap': True} |
| Logistic Regression   | 0.656   | 0.591   | {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'} |
| SGD - stochastic gradient descent   | 0.467   | 0.471   | {'alpha': 0.0001, 'l1_ratio': 0.0, 'loss': 'squared_error', 'penalty': 'l1'} |

## Executing the code 

In first place, you can execute the notebook.

If you prefer, you can also execute the python files:

  · git clone https://github.com/Ferri-bernabe/KaggleChessGame.git
  
  · pip install -r requeriements.txt
  
  · cd src && python3 run.py
  
## Conclusion
We have seen that we don't have the best results for the score or MSE in any learning machine model, because the correlation between our objective variable and the other ones aren't to good. Anyways, it seems that the Logistic Regression model presents the best score, the Random Forest Classifier the best MSE and the SGD model the best results considering the score and the MSE.


