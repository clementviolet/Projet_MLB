
# coding: utf-8


# Chargement des librairies
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn import tree
import keras
import numpy as np
import numpy.random as rd
from keras.models import Sequential
import  matplotlib.pyplot as plt
from keras.layers import Dense, Activation


# Réseau de neurones complètement connecté
def build_NN(inputdim):
    model = Sequential()
    model.add(Dense(20,activation='relu',input_shape=(inputdim,))) #4096
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

## la fonction partage construit les blocs de données pour l'apprentissage du modèle
def partage(nsample, nfold = 2, proptest = .3):
    prop = np.array(proptest) 
    somme = np.sum(np.array([prop]))
    prop = np.append([1-somme], prop)
    
    if(nfold != 2):
        prop = (1/nfold) * np.ones(nfold) 
    
    choix = rd.choice(nfold, nsample, True, prop)
    
    
    return tuple(choix)
    


# Classe avec plusieurs méthodes, prédict pour la prédiction, evaluation pour calculer les précisions
class modelisation:    
    def __init__(self, y, X):
        self.X = X
        self.y = y
        self.nbmodels = 0
        
    def fit(self, ntreeBag):  
         
        ##  distribution gaussienne
        self.ycarac = np.array([np.mean(self.y.flatten()), np.std(self.y.flatten())])
        self.nbmodels += 1
        
        ##  régression linéaire
        self.reg = LinearRegression().fit(self.X, self.y)
        self.nbmodels += 1
        
        ## arbre de régression
        self.regtree = tree.DecisionTreeRegressor().fit(self.X, self.y)       
        self.nbmodels += 1
        
        ## bagging 
        self.regbg = BaggingRegressor(n_estimators=ntreeBag).fit(self.X, self.y)
        self.nbmodels += 1

        ## réseau de neuronne
        self.model_NN = build_NN(self.X.shape[1])
        self.regnn = self.model_NN.fit(self.X, self.y,
                         epochs=20, batch_size=200, verbose=1)
        self.nbmodels += 1
        
        
        return (self.ycarac, self.reg, self.regtree, self.regbg, self.regnn)
        
    
    
    def predict(self,Xtest):
        #self.modmoy = self.ycarac[0] + self.ycarac[1] * rd.randn(len(Xtest)) 
        return ( (self.ycarac[0] + self.ycarac[1] * rd.randn(len(Xtest)),
               self.reg.predict(Xtest).flatten(),
               self.regtree.predict(Xtest).flatten(),
               self.regbg.predict(Xtest).flatten(),
               self.model_NN.predict(Xtest).flatten()))
    
        
    
    def stack_pred(self, Xtest):
        predictions = np.array(self.predict(Xtest))
        pred = np.mean(predictions, 0)
        return(pred)
    
    
    def evaluation(self, Xtest, ytest):
        return (np.mean((self.ycarac[0] + self.ycarac[1] * rd.randn(len(Xtest)) - ytest)**2),
                np.mean((self.reg.predict(Xtest) - ytest)**2),
                np.mean((self.regtree.predict(Xtest) - ytest)**2),
                np.mean((self.regbg.predict(Xtest) - ytest)**2),
                np.mean((self.model_NN.predict(Xtest) - ytest)**2),
                np.mean((self.stack_pred(Xtest) - ytest)**2)
               )        
        

    def cross_val(self, nbrefold, ntreeBag):
        partageDonnees = partage(len(self.X), nfold = nbrefold)
        for k in range(1,nbrefold+1):
            model = modelisation(self.y[np.array(partageDonnees) != k-1], self.X[np.array(partageDonnees) != k-1]) 
            model.fit(ntreeBag)
            if(k == 1):
                nbremodels = model.nbmodels
                tablePres = np.zeros([nbrefold, nbremodels + 1])
            tablePres[k-1] = model.evaluation(self.X[np.array(partageDonnees) == k-1], self.y[np.array(partageDonnees) == k-1])
        return (tablePres, np.mean(tablePres, 0),np.std(tablePres,0))



def resultat(models, ncv, ntreeBag, **kwargs):
    '''
    liste des arguments possibles
    ncv = 5, maxleafnodes = 500, minsampleleaf = 50, maxdepth=20, c=0.02,  nneigh = 5, ntreeForet = 5, ntreeBag = 5
    '''
    erreurEstimation = models.evaluation(models.X, models.y)
    erreurGeneralisation = models.cross_val(ncv, ntreeBag)
    
    barWidth = 0.25
    r1 = np.arange(len(erreurEstimation))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, erreurEstimation, color='yellow', width=barWidth, edgecolor='white', label="Erreur d'estimation")
    plt.bar(r2, erreurGeneralisation[1], color='#557f2d', width=barWidth, edgecolor='white', label="Erreur de généralisation (CV)")
    plt.xlabel('Modèles', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(erreurEstimation))], ['Dist', 'RL', 'TREE', 'BG', 'MLP', 'ST'])
    plt.show()
    

