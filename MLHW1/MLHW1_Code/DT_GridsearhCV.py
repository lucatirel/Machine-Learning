# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 14:50:56 2021

@author: luca9
"""

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

# Grid Search for Decision Tree
parameters1 = {'max_depth':[10, 50, 70, None],
              'criterion':('gini','entropy'), 'min_samples_leaf':[1,10,100],
              'min_samples_split':[2, 10, 100, 1000],'max_features':["auto","log2",None],
                'max_leaf_nodes':[10, 100, 1000, None], 'splitter':["best","random"]}

Best1 = []
Scores1 = []
for score in scores:
    print("# Tuning hyper-parameters for %s" % score + "\n")
    
    clf1 = GridSearchCV(l1, parameters1, scoring="%s_macro" % score, n_jobs=-1, cv=cv)
    clf1.fit(X_train2, y_train2)
    
    s1 = clf1.score(X_train2,y_train2)
    
    Best1.add(clf1.best_params_)
    Scores1.add(s1)
    
    print("Best parameters set found on development set:\n")
    print(clf1.best_params_)
    print("\n")
    print(score + ":")
    print(s1)
    print("Grid scores on development set:\n")
    means1 = clf1.cv_results_["mean_test_score"]
    stds1 = clf1.cv_results_["std_test_score"]
    for mean1, std1, params1 in zip(means1, stds1, clf1.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean1, std1 * 2, params1))
    print("\n")

print("Done.")


print(clf1.best_params_)
print(clf1.score(X_train2,y_train2))




# Best found for decision tree
# {'criterion': 'entropy',
#  'max_depth': None,
#  'max_features': None,
#  'max_leaf_nodes': None,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'splitter': 'best'}









