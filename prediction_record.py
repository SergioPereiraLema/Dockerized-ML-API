# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:30:58 2021

@author: Sergio Pereira
"""

#Import all the package
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#load the model into memory
with open('./model/model.pkl','rb') as model_pkl:
    knn = pickle.load(model_pkl)

#Test data
new_record = np.array([[1.2,3.6,3.8,3.4]])

predict_result = knn.predict(new_record)

#Print result to console
print('Predicted result for observations '+ str(new_record) + ' is ' + str(predict_result))

    