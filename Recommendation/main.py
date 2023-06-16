import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load the model from the file
crop_recommend = joblib.load('Crop Recommendation_Random Forest.pkl')

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_crop():
    request_data = request.get_json()
    N = request_data['n']
    P = request_data['p']
    K = request_data['k']
    temperature = request_data['temperature']
    humidity = request_data['humidity']
    ph = request_data['ph']
    rainfall = request_data['rainfall']
    result = {}
    json = {}
    prediction = crop_recommend.predict_proba([[N,P,K,temperature,humidity,ph,rainfall]])
    classes = crop_recommend.classes_
    preds = prediction[0]

    for i in range(len(preds)):
        if preds[i] != 0.0:
            result[classes[i]] = preds[i]

    #Sort it by the highest value
    result = dict(sorted(result.items(), key=lambda item: item[1],reverse=True)).keys()

    for i in range(3):
        try:
            json[f"Crops {i+1}"] = list(result)[i]
        except:
            json[f"Crops {i+1}"] = "None"
    
    return jsonify(json)

if __name__ == '__main__':
    app.run(debug=True)
        
