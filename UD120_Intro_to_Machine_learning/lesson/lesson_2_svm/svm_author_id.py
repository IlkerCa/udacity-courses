#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("./tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]


#########################################################
### your code goes here ###

svm = SVC(kernel='rbf', gamma='auto', C=10000)

t0 = time()
svm.fit(features_train, labels_train)
print("Predicting Time train:", round(time()-t0, 3), "s")

t0 = time()
arr_pred = svm.predict(features_test)
#print(svm.predict(features_test[10].reshape(1,-1)))
#print(svm.predict(features_test[26].reshape(1,-1)))
#print(svm.predict(features_test[50].reshape(1,-1)))
print("Predicting Time predict:", round(time()-t0, 3), "s")
print(np.sum(arr_pred == 1))

print("Accuracy: ", (labels_test == arr_pred).sum()/len(labels_test))

#########################################################

