from flask import Flask, render_template, request
import numpy as np
import requests

app = Flask(__name__)

'''
route to land on the index page
'''
@app.route("/")
def hello_world():
    return render_template('index.html')

'''
route which collect the values, wraps them as a json, sends to endpoint api to predict the result.
'''
@app.route('/predict', methods=['POST'])
def predict():
    # collecting values
    pregnancies = float(request.form['Pregnancies'])
    glucose = float(request.form['PlasmaGlucose'])
    dbp = float(request.form['DiastolicBloodPressure'])
    triceps_thickness = float(request.form['TricepsThickness'])
    serum_insulin = float(request.form['SerumInsulin'])
    bmi = float(request.form['BMI'])
    diabetes_pedigree = float(request.form['DiabetesPedigree'])
    age = float(request.form['Age'])

    # wrapping it to send it to api
    data_arr=[pregnancies,glucose,dbp,triceps_thickness,serum_insulin,bmi,diabetes_pedigree,age]
    np_arr=np.array([data_arr])
    input_value={
        "instances":np_arr.tolist()
    }
    endpoint="http://localhost:7777/invocations"
    response=requests.post(endpoint,json=input_value)
    result= eval(response.text)["predictions"]
    print(result)
    if(result[0]==1):
        return "<h1>Patient has diabetes</h1>"
    return "<h1>Patient do not has diabetes</h1>"

if __name__ == '__main__':
    app.run(debug=True,port=8080)