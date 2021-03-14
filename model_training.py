# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:06:24 2021

@author: Sergio Pereira
"""

# firt, load Iris dataset

from sklearn import datasets
iris = datasets.load_iris()

#separate features and target labels in different dataframes
x = iris.data
y = iris.target

#split the dataset in train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.3)

#Build the model using KNeighborgs classifier
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()

#train the model
knn.fit(x_train, y_train)

#predict with test dataset
predictions = knn.predict(x_test)

#verify accuracy of our model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

#store the coeficients in the pickle file in its serialized form
# adn we can import and unpicle this file for prediction in the future
# so that we don't need to train the model every time we want to make a prediction
import pickle
with open('./model/model.pkl','wb') as model_pkl:
    pickle.dump(knn,model_pkl)

