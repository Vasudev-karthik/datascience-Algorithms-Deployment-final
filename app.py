from flask import Flask, request, url_for, redirect, render_template

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import decmodel
from decmodel import results
from decmodel import cls_report
from decmodel import acc
from decmodel import df
from decmodel import tr
from decmodel import dtc

from linearmodel import b0,b1
from linearmodel import rmse
from linearmodel import r2
 


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

linearmodel=pickle.load(open('linearmodel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decision')
def decision():
    return render_template('decisionmain.html')

@app.route('/prediction')
def prediction():
    return render_template('decisionpredictionpage.html')

@app.route("/predict",methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("decisionpredictionpage.html", prediction_text = "The flower species is {}".format(prediction))

@app.route("/result")
def result():
    cr_array= cls_report
    return render_template("result.html",
    result=results,
    cr=cr_array,
    accuracy=" Accuracy is : {}".format(acc))

@app.route('/decparameters')
def decparameters():
    return render_template('decpara.html')

@app.route('/decback', methods = ["post"])
def decback():
    return render_template('decisionmain.html')


@app.route("/train")
def train():
    pickle.dump(dtc, open("model.pkl", "wb"))



    #print("Hello Karthik")

    return render_template("train.html")



# ----------------------------------------LINEAR MODELS--------------------------------------------------------

@app.route('/linear')
def linear():
    return render_template('linearmain.html')

@app.route('/linear_prediction')
def linear_prediction():
    return render_template('linearpredictionpage.html')

@app.route("/linear_result")
def linear_result():
    return render_template("linearresult.html", 
    cr="coefficient of regression is: {},{}".format(b0,b1),
    root="Root mean square error is:{}".format(rmse),
    r2="R2 score is : {}".format(r2))
    
@app.route("/linear_predict",methods = ["POST"])
def linear_predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = linearmodel.predict(features)
    output = round(prediction[0], 2)
    return render_template("linearpredictionpage.html", prediction_text = "The Brain weight is {}".format(output))
 

@app.route("/lineartrain")
def lineartrain():
    pickle.dump(dtc, open("linearmodel.pkl", "wb"))
    return render_template("lineartrain.html")

    
@app.route('/linearparameters')
def linearparameters():
    return render_template('linearparameters.html')

@app.route('/back', methods = ["post"])
def back():
    return render_template('linearmain.html')
 



if __name__ == "__main__":
    app.run(debug=True)