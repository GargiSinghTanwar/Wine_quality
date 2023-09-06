from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open('classifier_kmeans_celebalTechnologies.pkl', 'rb'))

print(type(model))

@app.route('/')
def home():

    return render_template("untitled.html")

@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  sulphates = int(request.args.get('sulphates'))
  alcohol = int(request.args.get('alcohol'))


  prediction = model.predict([[sulphates, alcohol]])

  if prediction==[0]:
    result="Average quality"

  elif prediction==[1]:
    result="Good quality"
  elif prediction==[2]:
    result="Excellent quality"



  return render_template('untitled.html', prediction_text='Model  has predicted  : {}'.format(result))


app.run()