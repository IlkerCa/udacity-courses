#!/usr/bin/python3

import numpy as np
import pickle
from numpy.core.fromnumeric import sort
np.random.seed(42)
from sklearn.tree import DecisionTreeClassifier



### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "C:/Users/xxx/Documents/coding/Data Science/ud120-projects/your_word_data.pkl" 
authors_file = "C:/Users/xxx/Documents/coding/Data Science/ud120-projects/your_email_authors.pkl"

words_file = open(words_file, "rb")
word_data = pickle.load(words_file)

authors_file = open(authors_file, "rb")
authors = pickle.load(authors_file)

#word_data = pickle.load(open(words_file, "r"))
#authors = pickle.load(open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
clf = DecisionTreeClassifier(random_state=0)
clf.fit(features_train, labels_train)
preds = clf.predict(features_test)

print(np.sum(preds == labels_test)/len(labels_test))

features_imp = clf.feature_importances_

for index, feature_val in enumerate(features_imp):
    if feature_val > 0.2:
        print(index, feature_val)


indices = np.argsort(features_imp)[::-1]
print('Feature Ranking: ')
for i in range(10):
    print("{} feature no.{} ({})".format(i+1,indices[i],features_imp[indices[i]]))

print(vectorizer.get_feature_names()[21323])


