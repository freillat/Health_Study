import pickle
from flask import Flask, request, jsonify

app = Flask('model') # give an identity to your web service

def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

(dv, model) = load('model.bin')
# model = load('model1.bin')

def predict_single(patient, dv, model):
    X = dv.transform([patient])  ## apply the one-hot encoding feature to the customer data 
    y_pred = model.predict(X)
    return y_pred

# print(predict_single(client, dv, model))

@app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
def predict():
    patient = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.
    prediction = predict_single(patient, dv, model)
    return jsonify(prediction)  ## send back the data in json format to the user

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696