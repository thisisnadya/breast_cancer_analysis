
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cluster import KMeans
from sklearn import metrics
from flask import Flask, request, render_template
import re
import math
import pickle

app = Flask("__name__")
model = pickle.load(open('final_model.pickle', 'rb'))
pca = pickle.load(open('pca.pickle', 'rb'))

q = ""

@app.route("/")
def loadPage():
	return render_template('tes.html')

@app.route("/start")
def loadMainPage():
	return render_template('home.html')

def cancerClustering(perimeter_worst, radius_worst, texture_se):
    data = pd.DataFrame(data=[[perimeter_worst, radius_worst, texture_se]], 
                        columns=["Perimeter Worst", "Radius Worst", "Texture"])

    reduce_feature = pca.transform(data)

    result = model.predict(reduce_feature)[0]
    
    print(result)
    return result


#prediction function
# def ValuePredictor(to_predict_list):
#     to_predict = np.array(to_predict_list).reshape(1, 3)
#     loaded_model = pickle.load(open("final_model.pickle", "rb"))
#     result = loaded_model.predict(to_predict)
#     return result[0]


@app.route("/getCluster", methods=['POST'])
def getCluster():

    if request.method == 'POST':
        # Get the input from the form
        radius_worst = request.form['radius_worst']
        perimeter_worst = request.form['perimeter_worst']
        texture_se = request.form['texture_se']

        # to_predict_list = list(map(float, [radius_worst, perimeter_worst, texture_se]))
        result = cancerClustering(radius_worst, perimeter_worst, texture_se)

        if float(result) == 0:
            prediction = 'Tidak Kanker'
        elif float(result) == 2:
            prediction = 'Berpotensi Kanker'
        elif float(result) == 1:
            prediction = 'Kanker'

    return render_template('getCluster.html',prediction=prediction)

    
    
app.run()

