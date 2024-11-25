import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mutual_info_score, accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import pickle 

# Importing the dataset and cleaning

file_address= 'diabetes.csv'
df = pd.read_csv(file_address)
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.dropna(inplace=True)

# splitting into 80% training / 20% validation and 20% test sets

target = "outcome"
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
y_train = df_train[target]
y_full_train = df_full_train[target]
y_val = df_val[target]
y_test = df_test[target]
del df_train[target]
del df_full_train[target]
del df_val[target]
del df_test[target]

# Model training

dv = DictVectorizer(sparse=False)
rf = RandomForestClassifier(n_estimators=150, max_depth=13, random_state=1)
full_train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dicts)
test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)
rf.fit(X_full_train, y_full_train)
y_pred = rf.predict(X_test)
print("Random Forest Classifier Stats:")
print(classification_report(y_test, y_pred))

# Model export

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, rf), f_out)