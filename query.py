client = {
    "sex": 1, 
    "age": 40, 
    "hypertension": "success", 
    "heart_disease": 1, 
    "ever_married":1, 
    "work_type": 4, 
    "Residence_type": 1, 
    "avg_glucose_level": 2, 
    "bmi": 35, 
    "smoking_status": 1
    }

import requests ## to use the POST method we use a library named requests
url = 'http://localhost:9696/predict' ## this is the route we made for prediction
response = requests.post(url, json=client) ## post the patient information in json format
result = response.json() ## get the server response
print(result)