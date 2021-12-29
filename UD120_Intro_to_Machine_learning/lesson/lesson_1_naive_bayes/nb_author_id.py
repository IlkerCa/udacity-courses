#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("./tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
nb = GaussianNB()
t0 = time()
nb.fit(features_train, labels_train)
print("Predicting Time train:", round(time()-t0, 3), "s")


##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
# # < your clf.predict() line of code >
arr_pred = nb.predict(features_test)
print("Predicting Time predict:", round(time()-t0, 3), "s")

print("Accuracy: ", (labels_test == arr_pred).sum()/len(labels_test))


##############################################################