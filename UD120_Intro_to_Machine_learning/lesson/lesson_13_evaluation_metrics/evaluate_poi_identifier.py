#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("C:/Users/xxx/Documents/coding/Data Science/ud120-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

data_dict = joblib.load( open("C:/Users/xxx/Documents/coding/Data Science/ud120-projects/final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.3,
                                                                                          random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

print(clf.score(features_test, labels_test))

#How many POIs are predicted for the test set for your POI identifier?
pred = clf.predict(features_test)
print('#Poi: ', np.sum(pred)) #3

# How many people total are in your test set?
print(len(pred)) #29

#If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
#1 - 3/29 = 0.89

from sklearn.metrics import *

print(precision_score(labels_test, pred)) #0
print(recall_score(labels_test, pred)) #0

#How many true positives are there? #6
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print(np.logical_and(predictions==1, true_labels==1).sum())
print(np.sum(np.logical_and(predictions == 1, true_labels == 1)))
print(confusion_matrix(true_labels, predictions))

print(precision_score(true_labels, predictions))
print(recall_score(true_labels, predictions))