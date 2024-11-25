patient = {
    "pregnancies": 1,
    "glucose": 130,
    "bloodpressure": 84,
    "skinthickness": 28,
    "insulin": 89,
    "bmi": 35,
    "diabetespedigreefunction": 0.3,
    "age": 40
    }

import requests ## to use the POST method we use a library named requests
url = 'http://localhost:9696/predict' ## this is the route we made for prediction
response = requests.post(url, json=patient) ## post the patient information in json format
result = response.json() ## get the server response
print(result)