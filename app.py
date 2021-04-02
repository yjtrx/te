import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('aqt.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    features = [str(x) for x in request.form.values()]
    courseint = int(features[0])
    course = int(features[1])
    classsize = int(features[2])
    final_features = []

    final_features.append(courseint)
    final_features.append(course)
    final_features.append(classsize)
    print(len(final_features))
    prediction = model.predict([final_features])
    output = prediction[0]
    return render_template('index.html',prediction_text=output)

if __name__=="__main__":
    app.run(debug=False)
