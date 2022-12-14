# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:05:23 2021

@author: lucatirel
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

blinddata = 'C:/Users/luca9/Desktop/MLHW1/MLHW1_Code/blind_test.csv'
bds = pd.read_csv(blinddata, sep='\t', header=0, names=['instructions', 'source', 'line', 'function', 'program'])
print('Blind File loaded: %d samples.\n' %(len(bds.program)))

# Print a random sample in the data
id = random.randrange(0,len(ds.bug))
print('This is a random sample:\nID:\t %d \ninstructions:\t %s \nsource_line:\t %s \nline_number:\t\t\t %s \nfunction_name:\t %s \nprogram:\t %s \nbug:\t %s' %(id,ds.instructions[id],ds.source[id],ds.line[id],ds.function[id],ds.program[id],ds.bug[id]))

# Define Data and Target Columns for dataset
X_all = ds.iloc[:,:-1]
y_all = ds.iloc[:,-1]


# Define Data Columns for blind dataset
X_all_b = bds.iloc[:,:-1]



# VECTORIZATION
# Define vectorizer
vec1 = CountVectorizer()   
vec2 = HashingVectorizer(analyzer="word", n_features=2**12, alternate_sign=False)
vec3 = TfidfVectorizer()
V = [vec1, vec2, vec3]

# Vectorize dataset values
X_all_vec = []
for v in V:
    X_all = None
    a = v.fit_transform(ds.instructions)
    b = v.fit_transform(ds.source)
    # c = scipy.sparse.csr_matrix(ds.line).T      # this is an array of integers
    # d = v.fit_transform(ds.function)
    # e = v.fit_transform(ds.program)
    
    X_all = hstack([a, b])
    
    X_all_vec.append(X_all)
    
# Vectorize blind dataset values
a = vec2.fit_transform(bds.instructions)
b = vec2.fit_transform(bds.source)
# c = scipy.sparse.csr_matrix(bds.line).T      # this is an array of integers
# d = vec2.fit_transform(bds.function)
# e = vec2.fit_transform(bds.program)

X_all_b_vec = hstack([a, b])

print("\nAll data has been vectorized.")

# Printing Shapes
for x in range(3):
    print("Dimension using " + str(V[x]) + ": " + str(X_all_vec[x].shape[1]))
print("\n")



# DATA SPLITTING
# Count Vectorizer Split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_all_vec[0], y_all, 
          test_size=0.333, random_state=23)
# Hash Vectorizer Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all_vec[1], y_all, 
          test_size=0.333, random_state=23)
# Tfid Vectorizer Split
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_all_vec[2], y_all, 
          test_size=0.333, random_state=23)

TR = [[X_train1,y_train1], [X_train2,y_train2], [X_train3,y_train3]]
TE = [[X_test1,y_test1], [X_test2,y_test2], [X_test3,y_test3]]



# LEARNING MODELS AND TRAINING
# Define type of Learning Model
l1 = tree.DecisionTreeClassifier()              # Decision Trees
l2 = LogisticRegression(max_iter=10000)          # Logistic Regression
l3 = MultinomialNB()                            # Multinomial Naive Bayes
l4 = RandomForestClassifier()                   # Random Forest 
l5 = DummyClassifier(strategy="most_frequent")  # Dummy Classifier

L = [l1, l2, l3, l4, l5]

# Training the Models
M = [[], [], []]

for i in range(3):
    for l in L:
        l = clone(l, safe=True)
        m = l.fit(TR[i][0], TR[i][1])
        M[i].append(m)



# EVALUATION
# Evaluate Accuracy
ACC = [[], [], []]

for i in range(3):
    for m in M[i]:
        acc = m.score(TE[i][0], TE[i][1])

        ACC[i].append(acc)
        print("Model: " + str(m) + "\nVectorizer: " + str(V[i]) + "\nAccuracy: " + str(acc) + "\n")
    print("\n")

# PREDICTION ON TEST SET
# Predict on test sets
P = [[], [], []]

for i in range(3):
    for m in M[i]:
        y = m.predict(TE[i][0])
        
        P[i].append(y)






# COMPARISON OF PERFORMANCES AFTER TUNING
# DECISION TREES vs RANDOM FOREST using HASH VECTORIZER
# Define tuned models
m1 = tree.DecisionTreeClassifier(criterion="entropy", 
                                 max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, max_features=None,
                                 max_leaf_nodes=None).fit(X_train2, y_train2)
m2 = RandomForestClassifier(n_estimators=70, max_depth=None, criterion='gini', 
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


# K-fold Cross Validation
cv = ShuffleSplit(n_splits=10, test_size=0.333, random_state=23)
scores1 = cross_val_score(m1, X_all, y_all, cv=cv)
scores2 = cross_val_score(m2, X_all, y_all, cv=cv)

print("\n\nACCURACY OF ALGORITHMS USING K-FOLD CROSS VALIDATION METHOD:")
print("Accuracy of Decision Tree : %0.3f +/- %0.2f" % (scores1.mean(), scores1.std() * 2))
print("Accuracy of Random Forest : %0.3f +/- %0.2f" % (scores2.mean(), scores2.std() * 2))
print("\n")


# PREDICTION ON BLIND DATASET
# Predict on blind sets with RandomForestClassifier
PB = m2.predict(X_all_b_vec)

# Write output in txt file:
with open("1702631.txt", 'w') as f:        
    for y in PB:
        f.write(str(y) + '\n')


