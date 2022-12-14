# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:05:23 2021

@author: luca9
"""


# LIBRARIES
import numpy as np
import pandas as pd
import random
import scipy

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing 
from scipy.sparse import hstack
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
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
# c = scipy.sparse.csr_matrix(ds.line).T      # this is an array of integers
# d = v.fit_transform(ds.function)
# e = v.fit_transform(ds.program)

X_all = hstack([a, b])


print("All data has been vectorized.")



# DATA SPLITTING

# Hash Vectorizer Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all, y_all, 
          test_size=0.333, random_state=23)



# COMPARISON OF PERFORMANCES AFTER TUNING
# DECISION TREES vs RANDOM FOREST using HASH VECTORIZER
# Define tuned models
m1 = tree.DecisionTreeClassifier(criterion="entropy", 
                                 max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, max_features=None,
                                 max_leaf_nodes=None).fit(X_train2, y_train2)

m2 = RandomForestClassifier(n_estimators=70, max_depth=None, criterion='gini',
                            min_samples_split = 2, min_samples_leaf = 1,
                            max_features = "auto",
                            bootstrap=True, oob_score=True, warm_start=True).fit(X_train2, y_train2)
# Predict on test set
y_pred1 = m1.predict(X_test2)
y_pred2 = m2.predict(X_test2)



# Metrics of the tuned models
print("Classification report of Decision Tree")
print(classification_report(y_test2, y_pred1, labels=None, digits=3))
print("\n")
print("Classification report of Random Forest")
print(classification_report(y_test2, y_pred2, labels=None, digits=3))
print("\n")









# PLOTS
# Confusion matrices plot
cm1 = confusion_matrix(y_test2, y_pred1)
cm2 = confusion_matrix(y_test2, y_pred2)

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot()
plt.show()
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()
plt.show()


# DECISION TREE
# Plot of the scores vs max_depth of DecisionTreeClassifier using HASH VECTORIZER
D = []
A1 = []
R1 = []
F1 = []
for depth in range(10, 51, 1):
    m = tree.DecisionTreeClassifier(criterion="entropy", 
                                  max_depth=depth, min_samples_split=2,
                                  min_samples_leaf=1, max_features=None,
                                  max_leaf_nodes=None).fit(X_train2, y_train2)
    
    a = m.score(X_test2, y_test2)
    y_p = m.predict(X_test2)
    r = recall_score(y_test2, y_p)
    f = f1_score(y_test2, y_p)
    
    D.append(depth)
    A1.append(a)
    R1.append(r)
    F1.append(f)
    
plt.plot(D, A1)
plt.plot(D, R1)
plt.plot(D, F1)
plt.xlabel("Depth of the Tree")
plt.legend(["accuracy", "recall", "f1-score"])
plt.title("Decision Tree Performances")
plt.show()
        
# Plot of the scores vs min_sample_split of DecisionTreeClassifier using HASH VECTORIZER
S = []
A2 = []
R2 = []
F2 = []
for split in range(2, 15003, 500):
    m = tree.DecisionTreeClassifier(criterion="entropy", 
                                  max_depth=None, min_samples_split=split,
                                  min_samples_leaf=1, max_features=None,
                                  max_leaf_nodes=None).fit(X_train2, y_train2)
    a = m.score(X_test2, y_test2)
    y_p = m.predict(X_test2)
    r = recall_score(y_test2, y_p)
    f = f1_score(y_test2, y_p)
    
    S.append(split)
    A2.append(a)
    R2.append(r)
    F2.append(f)
    
plt.plot(S, A2)
plt.plot(S, R2)
plt.plot(S, F2)
plt.xlabel("Minimum number of samples for splitting internal nodes")
plt.legend(["accuracy", "recall", "f1-score"])
plt.title("Decision Tree Performances")
plt.show()

# Plot of the scores vs min_samples_leaf of DecisionTreeClassifier using HASH VECTORIZER
LL = []
A3 = []
R3 = []
F3 = []
for leaf in range(1, 76, 1):
    m = tree.DecisionTreeClassifier(criterion="entropy", 
                                  max_depth=None, min_samples_split=2,
                                  min_samples_leaf=leaf, max_features=None,
                                  max_leaf_nodes=None).fit(X_train2, y_train2)
    a = m.score(X_test2, y_test2)
    y_p = m.predict(X_test2)
    r = recall_score(y_test2, y_p)
    f = f1_score(y_test2, y_p)
    
    LL.append(leaf)
    A3.append(a)
    R3.append(r)
    F3.append(f)
    
plt.plot(LL, A3)
plt.plot(LL, R3)
plt.plot(LL, F3)
plt.xlabel("Minimum number of samples required to be a leaf node")
plt.legend(["accuracy", "recall", "f1-score"])
plt.title("Decision Tree Performances")
plt.show()


# RANDOM FOREST
# Plot of the scores vs n_estimators of RandomForestClassifier using HASH VECTORIZER
FF=[]
AA4 = []
RR4 = []
FF4 = []
for est in range(10, 121, 30):
    m = RandomForestClassifier(n_estimators=est, max_depth=None, criterion='gini',
                            min_samples_split = 2, min_samples_leaf = 1,
                            max_features = "auto",
                            bootstrap=True, oob_score=True, warm_start=True).fit(X_train2, y_train2)
    a = m.score(X_test2, y_test2)
    y_p = m.predict(X_test2)
    r = recall_score(y_test2, y_p)
    f = f1_score(y_test2, y_p)
    
    FF.append(est)
    AA4.append(a)
    RR4.append(r)
    FF4.append(f)
    
plt.plot(FF, AA4)
plt.plot(FF, RR4)
plt.plot(FF, FF4)
plt.xlabel("Numbers of tree in the forest")
plt.legend(["accuracy", "recall", "f1-score"])
plt.title("Random Forest Performances")
plt.show()
    
# Plot of the scores vs min_sample_split of RandomForestClassifier using HASH VECTORIZER
SS = []
AA2 = []
RR2 = []
FF2 = []
for split in range(2, 30003, 5000):
    m = RandomForestClassifier(n_estimators=70, max_depth=None, criterion='gini',
                            min_samples_split = split, min_samples_leaf = 1,
                            max_features = "auto",
                            bootstrap=True, oob_score=True, warm_start=True).fit(X_train2, y_train2)
    a = m.score(X_test2, y_test2)
    y_p = m.predict(X_test2)
    r = recall_score(y_test2, y_p)
    f = f1_score(y_test2, y_p)
    
    SS.append(split)
    AA2.append(a)
    RR2.append(r)
    FF2.append(f)
    
plt.plot(SS, AA2)
plt.plot(SS, RR2)
plt.plot(SS, FF2)
plt.xlabel("Minimum number of samples for splitting internal nodes")
plt.legend(["accuracy", "recall", "f1-score"])
plt.title("Random Forest Performances")
plt.show()








# BEST VALUES
# Decision Tree
# {'criterion': 'entropy',
#  'max_depth': None,
#  'max_features': None,
#  'max_leaf_nodes': None,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'splitter': 'best'}













