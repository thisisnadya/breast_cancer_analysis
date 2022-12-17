
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

def cancerClustering(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst):
    data = pd.DataFrame(data=[[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]], 
                        columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"])

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
        radius_mean = request.form['radius_mean']
        texture_mean = request.form['texture_mean']
        perimeter_mean = request.form['perimeter_mean']
        area_mean = request.form['area_mean']
        smoothness_mean = request.form['smoothness_mean']
        compactness_mean = request.form['compactness_mean']
        concavity_mean = request.form['concavity_mean']
        concave_points_mean = request.form['concave_points_mean']
        symmetry_mean = request.form['symmetry_mean']
        fractal_dimension_mean = request.form['fractal_dimension_mean']
        radius_se = request.form['radius_se']
        texture_se = request.form['texture_se']
        perimeter_se = request.form['perimeter_se']
        area_se = request.form['area_se']
        smoothness_se = request.form['smoothness_se']
        compactness_se = request.form['compactness_se']
        concavity_se = request.form['concavity_se']
        concave_points_se = request.form['concave_points_se']
        symmetry_se = request.form['symmetry_se']
        fractal_dimension_se = request.form['fractal_dimension_se']
        radius_worst = request.form['radius_worst']
        texture_worst = request.form['texture_worst']
        perimeter_worst = request.form['perimeter_worst']
        area_worst = request.form['area_worst']
        smoothness_worst = request.form['smoothness_worst']
        compactness_worst = request.form['compactness_worst']
        concavity_worst = request.form['concavity_worst']
        concave_points_worst = request.form['concave_points_worst']
        symmetry_worst = request.form['symmetry_worst']
        fractal_dimension_worst = request.form['fractal_dimension_worst']

        # to_predict_list = list(map(float, [radius_worst, perimeter_worst, texture_se]))
        result = cancerClustering(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst)

        if float(result) == 0:
            prediction = 'Tidak Kanker'
        elif float(result) == 2:
            prediction = 'Berpotensi Kanker'
        elif float(result) == 1:
            prediction = 'Kanker'

    return render_template('getCluster.html',prediction=prediction)

    
    
app.run()

