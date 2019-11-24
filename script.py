# Plot
import matplotlib.pyplot as plt

# Time
import time

# Pandas
import pandas as pd

# Rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Scikit Learn
import numpy as np
from sklearn import metrics, preprocessing, tree
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor

# Yellowbrick model selection and hyperparameter tuning 
from yellowbrick.regressor import AlphaSelection

# Utiliser le code R pour importer les données.

robjects.r['source']("script.R") # Il faut avoir installer les packages R tidyverse et missMDA

data = robjects.r['data']

# Conversion du format rpy2 en pandas

with localconverter(robjects.default_converter + pandas2ri.converter):
  pd_data = robjects.conversion.rpy2py(data)

# Exploration du jeu de données

list(pd_data.columns)

pd_data.head()

pd_data.describe()

pd_labels = pd_data['ViolentCrimesPerPop'] # Our goal is to predict violent crimes.

goal_vars = ["murders", "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbbPerPop",
             "assaults", "assaultPerPop", "burglaries", "burglPerPop", "larcenies", "larcPerPop",
             "autoTheft", "autoTheftPerPop", "arsons", "arsonsPerPop", "ViolentCrimesPerPop",
             "nonViolPerPop"] # Liste toutes les variables à retirer pour éviter de les utiliser poure prédire les crimes violents.

pd_features = pd_data.drop(goal_vars, axis = 1) # variables qui nous reste au final.

pd_data.LemasGangUnitDeploy.describe()
pd_features.drop("LemasGangUnitDeploy", axis = 1) # On retire cette variable car beaucoup de NA et a été mal prédite avec l'ACP

# Régression linéaire multiple
## But est de comparer nos modèles futurs avec un modèle null.

start_time = time.time()

B = 30

lm = LinearRegression()

mse_lm, varx_lm, r2_lm = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features, pd_labels, test_size = 0.25)#, random_state = 42)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  lm.fit(train_features, train_labels)
  predictions = lm.predict(test_features)

  r2_lm[b] = metrics.r2_score(test_labels, predictions)
  varx_lm[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_lm[b] = metrics.mean_squared_error(test_labels, predictions)

interval_lm = time.time() - start_time

print("Time elapsed ~= {:0.0f} s".format(interval_lm))
print("Root Mean Square Error (RMSE) : {:0.2f}".format(np.sqrt(np.mean(mse_lm))))
print("Variance explained : {:0.2f} %".format(100*np.mean(varx_lm)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_lm)))
# RMSE = 416.57 ; Var = 52.65%

start_time = time.time()

lm = LinearRegression()

lmrfe = RFE(lm, step=1)

B = 30

mse_lmrfe, varx_lmrfe, r2_lmrfe = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features, pd_labels, test_size = 0.25)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  lmrfe.fit(train_features, train_labels)
  predictions = lmrfe.predict(test_features)

  r2_lmrfe[b] = metrics.r2_score(test_labels, predictions)
  varx_lmrfe[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_lmrfe[b] = metrics.mean_squared_error(test_labels, predictions)

interval_lmrfe = time.time() - start_time 

print("Time elapsed ~= {:0.0f} s".format(interval_lmrfe))
print("Root Mean Square Error (RMSE) = {:0.2f}".format(np.sqrt(np.mean(mse_lmrfe))))
print("Variance explained = {:0.2f} %".format(100*np.mean(varx_lmrfe)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_lmrfe)))
#RMSE = 404.91 ; Var = 55.62%

# Elasticnet Regression

alphas = np.logspace(-2, 0, 4000)

model = AlphaSelection(ElasticNetCV(alphas=alphas, cv = 30))
model.fit(preprocessing.scale(pd_features), pd_labels)
model.poof()

## Il faut choisir la valeur de alpha = 0.357

start_time = time.time()

el = ElasticNet(alpha=0.357)

B = 30 # Nombre d'échantillon pour la valisation croisée MC.

mse_el, varx_el, r2_el = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features, pd_labels, test_size = 0.25)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  el.fit(train_features, train_labels)
  predictions = el.predict(test_features)

  r2_el[b] = metrics.r2_score(test_labels, predictions)
  varx_el[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_el[b] = metrics.mean_squared_error(test_labels, predictions)

interval_el = time.time() - start_time

print("Time elapsed ~= {:0.0f} s".format(interval_el))
print("Root Mean Square Error (RMSE) = {:2.2f}".format(np.sqrt(np.mean(mse_el))))
print("Variance explained = {:0.2f} %".format(100*np.mean(varx_el)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_el)))
## RMSE = 365.03 ; Var = 64.42%

# Random Forest

start_time = time.time()

rf = RandomForestRegressor(n_estimators = 100)

B = 30 # Nombre d'échantillon pour la valisation croisée MC.

mse_rf, varx_rf, r2_rf = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  print("Pourcentage effectue : {:0.2f} %".format(100*(b+1)/B))
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features, pd_labels, test_size = 0.25)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  rf.fit(train_features, train_labels)
  predictions = rf.predict(test_features)

  r2_rf[b] = metrics.r2_score(test_labels, predictions)
  varx_rf[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_rf[b] = metrics.mean_squared_error(test_labels, predictions)

interval_rf = time.time() - start_time

print("Time elapsed ~= {:0.0f} min".format(interval_rf/60))
print("Root Mean Square Error (RMSE):", round(np.sqrt(np.mean(mse_rf)), 2))
print("Variance explained : {:0.2f} %".format(100*np.mean(varx_rf)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_rf)))
# Time ~= 12 min, RMSE = 301.54, Var expalained = 70,28%, R2 = 0.70

rf.get_params()

# Nombre d'arbre dans la forêt
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Nombre de feature à consider pour séparer chaque noeud
max_features = ['auto', None, "log2"]
# Nombre maximal de niveaux pour l'arbe
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20, 50, 100, 200]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 10, 20, 50, 100]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
time_start = time.time()
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2)
# Fit the random search model
rf_random.fit(train_features, train_labels)
interval_rfrd = time.time() - time_start

# Results :
# RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=70,
#                       max_features='log2', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=5,
#                       min_weight_fraction_leaf=0.0, n_estimators=1000,
#                       n_jobs=None, oob_score=False, random_state=None,
#                       verbose=0, warm_start=False)

start_time = time.time()

rf_grid = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=70,
                                max_features='log2', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=5,
                                min_weight_fraction_leaf=0.0, n_estimators=1000,
                                n_jobs=None, oob_score=False, random_state=None,
                                verbose=0, warm_start=False)

B = 30 # Nombre d'échantillon pour la valisation croisée MC.

mse_rf_grid, varx_rf_grid, r2_rf_grid = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  print("Pourcentage effectue : {:0.2f} %".format(100*(b+1)/B))
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features, pd_labels, test_size = 0.25)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  rf.fit(train_features, train_labels)
  predictions = rf.predict(test_features)

  r2_rf_grid[b] = metrics.r2_score(test_labels, predictions)
  varx_rf_grid[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_rf_grid[b] = metrics.mean_squared_error(test_labels, predictions)

interval_rf = time.time() - start_time

print("Time elapsed ~= {:0.0f} min".format(interval_rf/60))
print("Root Mean Square Error (RMSE):", round(np.sqrt(np.mean(mse_rf)), 2))
print("Variance explained : {:0.2f} %".format(100*np.mean(varx_rf)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_rf)))


#### Modèle avec toutes les variables

pd_features_all = pd_data.drop(["ViolentCrimesPerPop", "LemasGangUnitDeploy"], axis = 1)

# Régression linéaire multiple
## But est de comparer nos modèles futurs avec un modèle null.

start_time = time.time()

B = 30

lm = LinearRegression()

mse_lm2, varx_lm2, r2_lm2 = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features_all, pd_labels, test_size = 0.25)#, random_state = 42)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  lm.fit(train_features, train_labels)
  predictions = lm.predict(test_features)

  r2_lm2[b] = metrics.r2_score(test_labels, predictions)
  varx_lm2[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_lm2[b] = metrics.mean_squared_error(test_labels, predictions)

interval_lm2 = time.time() - start_time

print("Time elapsed ~= {:0.0f} s".format(interval_lm2))
print("Root Mean Square Error (RMSE) : {:0.2f}".format(np.sqrt(np.mean(mse_lm2))))
print("Variance explained : {:0.2f} %".format(100*np.mean(varx_lm2)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_lm2)))
# RMSE : 149.09 ; Mean R2 : 0.94; Variance explained : 94.04%


#MSE selection

start_time = time.time()

lm = LinearRegression()

lmrfe = RFE(lm, step=1)

B = 30

mse_lmrfe2, varx_lmrfe2, r2_lmrfe2 = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features_all, pd_labels, test_size = 0.25)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  lmrfe.fit(train_features, train_labels)
  predictions = lmrfe.predict(test_features)

  r2_lmrfe2[b] = metrics.r2_score(test_labels, predictions)
  varx_lmrfe2[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_lmrfe2[b] = metrics.mean_squared_error(test_labels, predictions)

interval_lmrfe2 = time.time() - start_time 

print("Time elapsed ~= {:0.0f} s".format(interval_lmrfe2))
print("Root Mean Square Error (RMSE) = {:0.2f}".format(np.sqrt(np.mean(mse_lmrfe2))))
print("Variance explained = {:0.2f} %".format(100*np.mean(varx_lmrfe2)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_lmrfe2)))
# RMSE : 137.47 ; Mean R2 : 0.95; Variance explained : 94.!2%

# Elasticnet Regression

alphas2 = np.logspace(-2, 0, 4000)

model = AlphaSelection(ElasticNetCV(alphas=alphas2, cv = 30))
model.fit(preprocessing.scale(pd_features_all), pd_labels)
model.poof()

## Il faut choisir la valeur de alpha = 0.023

start_time = time.time()

el = ElasticNet(alpha=0.23)

B = 30 # Nombre d'échantillon pour la valisation croisée MC.

mse_el2, varx_el2, r2_el2 = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features_all, pd_labels, test_size = 0.25)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  el.fit(train_features, train_labels)
  predictions = el.predict(test_features)

  r2_el2[b] = metrics.r2_score(test_labels, predictions)
  varx_el2[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_el2[b] = metrics.mean_squared_error(test_labels, predictions)

interval_el2 = time.time() - start_time

print("Time elapsed ~= {:0.0f} s".format(interval_el2))
print("Root Mean Square Error (RMSE) = {:2.2f}".format(np.sqrt(np.mean(mse_el2))))
print("Variance explained = {:0.2f} %".format(100*np.mean(varx_el2)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_el2)))
## RMSE = 138.66 ; Var = 94.81%

# Random Forest

start_time = time.time()

rf = RandomForestRegressor(n_estimators = 100)

B = 30 # Nombre d'échantillon pour la valisation croisée MC.

mse_rf2, varx_rf2, r2_rf2 = np.zeros(B), np.zeros(B), np.zeros(B)

for b in range(0,B):
  print("Pourcentage effectue : {:0.2f} %".format(100*(b+1)/B))
  # Echantillonnage du jeu de données : train, test
  train_features, test_features, train_labels, test_labels = train_test_split(pd_features_all, pd_labels, test_size = 0.25)

  # Standardisation du jeu de données
  scaler = preprocessing.StandardScaler().fit(train_features)

  train_features = scaler.transform(train_features)
  test_features  = scaler.transform(test_features)

  # Apprentissage du model
  rf.fit(train_features, train_labels)
  predictions = rf.predict(test_features)

  r2_rf2[b] = metrics.r2_score(test_labels, predictions)
  varx_rf2[b] = metrics.explained_variance_score(test_labels, predictions)
  mse_rf2[b] = metrics.mean_squared_error(test_labels, predictions)

interval_rf2 = time.time() - start_time

print("Time elapsed ~= {:0.0f} min".format(interval_rf2/60))
print("Root Mean Square Error (RMSE):", round(np.sqrt(np.mean(mse_rf2)), 2))
print("Variance explained : {:0.2f} %".format(100*np.mean(varx_rf2)))
print("Mean R2 : {:0.2f}".format(np.mean(r2_rf2)))
# Time ~= 18 min, RMSE = 130.19, Var expalained = 95.58%, R2 = 0.96