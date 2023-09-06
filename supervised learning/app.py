import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
classifier_celebal_rf = pickle.load(open('classifier_major_rf_celebalTechnologies.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['GET'])
def predict():

    fixed_acidity = float(request.args.get('fixed_acidity'))
    volatile_acidity = float(request.args.get('volatile_acidity'))
    citric_acid = float(request.args.get('citric_acid'))
    residual_sugar = float(request.args.get('residual_sugar'))
    chlorides = float(request.args.get('chlorides'))
    free_sulfur_dioxide = float(request.args.get('free_sulfur_dioxide'))
    total_sulfur_dioxide = float(request.args.get('total_sulfur_dioxide'))
    density = float(request.args.get('density'))
    pH = float(request.args.get('pH'))
    sulphates = float(request.args.get('sulphates'))
    alcohol = float(request.args.get('alcohol'))


# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary
    Model = (request.args.get('Model'))

    if Model=="Random_Forest":
      prediction = classifier_celebal_rf.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    if prediction == [1]:
      return render_template('index.html', prediction_text='Excelent quality of wine', extra_text =" as per Prediction by model " + Model)

    else:
      return render_template('index.html', prediction_text='Average quality wine', extra_text ="as per Prediction by model " + Model)
app.run()

