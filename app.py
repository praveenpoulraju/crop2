from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('newcrop.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    input_query = np.array([[temperature,humidity,ph,rainfall]])

    result = model.predict(input_query)[0]

    return jsonify({'predicted crop is':str(result)})

if __name__ == '__main__':
    app.run(debug=True)