# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 21:53:30 2021

@author: luca9
"""



# LIBRARIES
import numpy as np
import pandas as pd
import random
import scipy

from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing 
from scipy.sparse import hstack
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

# IMPORT DATASET and BLIND TEST DATASET
# Dataset
data = 'C:/Users/luca9/Desktop/MLHW1/MLHW1_Code/mapping_traces_O0.csv'
ds = pd.read_csv(data, sep='\t', header=0, names=['instructions', 'source', 'line', 'function', 'program', 'bug'])
print('File loaded: %d samples.\n' %(len(ds.bug)))

# Define Data and Target Columns for dataset
X_all = ds.iloc[:,:-1]
y_all = ds.iloc[:,-1]

# VECTORIZATION
# Define vectorizer
v = HashingVectorizer(analyzer="word", n_features=2**12, alternate_sign=False)


# Vectorize dataset values

a = v.fit_transform(ds.instructions)
b = v.fit_transform(ds.source)

X_all = hstack([a, b])


print("All data has been vectorized.")


# DATA SPLITTING

# Hash Vectorizer Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all, y_all, 
          test_size=0.333, random_state=23)


# LEARNING MODELS AND TRAINING
# Define type of Learning Model
l1 = tree.DecisionTreeClassifier()              # Decision Trees
l2 = RandomForestClassifier(verbose=1, oob_score=True,)                   # Random Forest 



# # GRID SEARCH with KFOLD VALIDATION
cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=15)
scores = ["precision", "recall"]

# Grid Search for Random Tree
Best = []
Scores = []
parameters = {'max_depth':[30, 60, 100], 'n_estimators':[30, 50, 70, 100],
              'criterion':('gini','entropy'),'min_samples_split':[10, 100, 500],
              'max_leaf_nodes':[1, 10, 100, None], 'min_samples_leaf':[1,3,5,10,None],
              'max_samples':[0.25, 0.5, 0.75, 1],
                'max_features':['auto', 'sqrt', 'log2']}

for score in scores:
    print("# Tuning hyper-parameters for %s" % score + "\n")

    clf = GridSearchCV(l2, parameters, scoring="%s_macro" % score, n_jobs=-1, cv=cv, verbose=2)
    clf.fit(X_train2, y_train2)
    
    s = clf.score(X_train2,y_train2)
    
    Best.add(clf.best_params_)
    Scores.add(s)
    
    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("\n")
    print(score + ":")
    print(s)
    print("Grid scores on development set:\n")
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print("\n")

print("Done.")

print(clf.best_params_)
print(clf.score(X_train2,y_train2))



# Best Found for Random Forest
# {'criterion': 'entropy',
#  'max_depth': None,
#  'max_features': 'auto',
#  'n_estimators': 70}











