import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/HeartDisease')
def pred():
    return render_template('HeartDisease.html')

@app.route('/disease-pred')
def diseasePred():
    return render_template('disease-pred.html')

@app.route('/HeartDisease',methods=['GET','POST'])
def HeartDisease():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output==0:
        prediction_text='Congratulations! You are completely fit'
    elif output==1:
        prediction_text='Disease detected! Consult doctors'
    return render_template('results.html', prediction_text=prediction_text)

    '''
    For rendering results on HTML GUI
    '''


if __name__ == "__main__":
    app.run(debug=True)