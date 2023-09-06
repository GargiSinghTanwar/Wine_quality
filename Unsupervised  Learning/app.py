import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('hc_Model_celebalTechnologies.pkl','rb'))     

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  sulphates= float(request.args.get('sulphates'))
  alcohol = float(request.args.get('alcohol'))
  sulphates1 = float(request.args.get('sulphates1'))
  alcohol1 = float(request.args.get('alcohol1'))  
  sulphates2 = float(request.args.get('sulphates2'))
  alcohol2 = float(request.args.get('alcohol2')) 
     
  predict = model.fit_predict([[sulphates,alcohol ],[sulphates1,alcohol1], [sulphates2,alcohol2]])
  
  return render_template('index.html', prediction_text='Model  has predicted  : {}'.format(predict))


app.run()