from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/'
model = pickle.load(open("predictor.pkl", "rb"))

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
    return render_template("index.html", image=pic1)

@app.route('/predict', methods=['GET', 'POST'])
def predict():    
    Gender = int(request.form["Gender"]) 
    Age = int(request.form["Age"])
    Height = float(request.form["Height"])
    Weight = float(request.form["Weight"])
    Duration = float(request.form["Duration"])
    Heart_Rate = float(request.form["Heart_Rate"])
    Body_Temp = float(request.form["Body_Temp"])
    
    
    input_data = (Gender, Age,	Height, Weight, Duration, Heart_Rate, Body_Temp)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)[0]
    prediction = round(prediction, 2)
    return str(prediction)

    
if __name__ == '__main__':
    app.run(debug=True) 