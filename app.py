import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
#Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
#C:\Users\itsva\OneDrive\Desktop\CouseWorks\Data_Science\ML_Project_End_to_End\regmodel.pkl

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    predictied_price=regmodel.predict(new_data)[0]
    return jsonify(predictied_price)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    predicted_price=regmodel.predict(final_input)[0]
    print(predicted_price)
    return render_template("home.html",prediction_text='Predicted house price ins {}'.format(predicted_price))

if __name__=='__main__':
    app.run(debug=True)
