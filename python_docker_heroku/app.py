# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:49:58 2021

@author: Sergio Pereira
"""

#import all the required packages
import pickle
import numpy as np
import sys
import os
from sklearn.neighbors import KNeighborsClassifier

#import Flask for creating API
from flask import Flask, request

#define the PORT for our API
port = int(os.environ.get("PORT",5000))

#Load the trained model
with open('./model.pkl', 'rb') as model_pkl:
    knn=pickle.load(model_pkl)
    
#Initialise a flask app
app = Flask(__name__)

#create an API endpoint
@app.route('/predict')

# function to make the prediction
def predict_iris():
    #read the input parameters
    sl = request.args.get('sl')
    sw = request.args.get('sw')
    pl = request.args.get('pl')
    pw = request.args.get('pw')
    #create the array
    new_record = np.array([[sl,sw,pl,pw]])
    #use the model to get the prediction for the unseen data
    predict_result = knn.predict(new_record)

    #return the prediction back
    return 'The predicted result for the observation ' +  str(new_record) + ' is ' + str(predict_result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
    
