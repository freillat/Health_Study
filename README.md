#Health Study

This repo goes through data analysis and machine learning on a healthcare dataset with Python.
The dataset is a publicly available dataset on diabetes accross a number of patients (Pima Indians Diabetes Database - originally from the National Institute of Diabetes and Digestive and Kidney Diseases).

We will attempt to model and predict whether or not a patient is likely to have diabetes. This is a binary categorization problem (effectively yes or no diabetes).

Content of the repo and steps in the project:

1. Dataset:
diabetes.csv
or can be downloaded in various places on kaggle including
https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906/input

2. Notebook:
noteboook.ipynb goes through data preparation and data cleaning, EDA (Exploratory Data Analysis) as well as model comparison and selection
It assumes that it is being run in google colab (https://colab.research.google.com/) with the dataset in a connected google drive but can be run anywhere assuming you point to the dataset by modifying wherever you have save it )

3. Model Training:
train.py is the model training (based on selection and parameters tuned in step 2) and the saving of the model file (model.bin) for later use using Pickle
To run this - you can use: pipenv run python train.py

4. Model deployement:
predict.py loads of model.bin and serves it to deliver a prediction using Flask
To run this directly - you can use: pipenv run python predict.py
Or alternatively the docker setup as described below can be used.

5. Query:
query.py is a script to submit a quarey to the service
To run this directly - you can use: pipenv run python predict.py

Notes: environment is provided using pipenv thourgh Pipfile and Pipfile.lock
model.bin in the repo is the saved output model from running train.py
Dockerfile is provided to run the service using Docker

To use the docker setup you will need to:
1. build the docker container running the following command in the project directory which contains the Dockerfile:
    docker build -t diabetes_model:latest .
2. run the docker image with the following command:
    docker run -p 9696:9696 diabetes_model:latest
3. then you can use query.py to run a query with the following command:
    pipenv run python query.py