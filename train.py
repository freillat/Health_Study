import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mutual_info_score, accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import pickle 

# https://www.kaggle.com/code/voraseth/diabetes-hypertension-and-stroke-predicti-eda-014/input?select=stroke_data.csv
# https://www.kaggle.com/datasets/prosperchuks/health-dataset/data?select=stroke_data.csv
# data from https://gist.github.com/aishwarya8615/d2107f828d3f904839cbcb7eaa85bd04

file_address= 'https://gist.github.com/aishwarya8615/d2107f828d3f904839cbcb7eaa85bd04#file-healthcare-dataset-stroke-data-csv'
# file_address =  '/kaggle/input/diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv'
# file_address= 'stroke_data.csv'

df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df[df['diabetes_012']==2] = 1
# print(df.describe())
# df = df.drop(columns=['id'])
df.dropna(inplace=True)

# splitting into 80% training / 20% validation and 20% test sets

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)

y_train = df_train['diabetes_012']
y_val = df_val['diabetes_012']
y_test = df_test['diabetes_012']
del df_train['diabetes_012']
del df_val['diabetes_012']
del df_test['diabetes_012']
print(df_train.shape)
print(df_train.head())

# dealing with categorical variables

dv = DictVectorizer(sparse=False)
train_dicts = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_val)
print("Decision Tree Classifier:")
print(round(roc_auc_score(y_val, y_pred),3))
print(classification_report(y_val, y_pred))

# print(export_text(dt, feature_names=dv.get_feature_names_out()))

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train, y_train)
features = dv.get_feature_names_out().tolist()
f_importance = pd.Series(rf.feature_importances_, index = features).sort_values(ascending=False)
print(f_importance)
y_pred = rf.predict(X_val)
print("Random Forest Classifier:")
print(round(roc_auc_score(y_val, y_pred),3))
print(classification_report(y_val, y_pred))
test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)
y_tpred = rf.predict(X_test)
print("Random Forest Classifier (Test set):")
print(round(roc_auc_score(y_test, y_tpred),3))
print(classification_report(y_test, y_tpred))

for n in range(10,201,10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    print(n, round(roc_auc_score(y_val, y_pred),3))

for m in [10, 15, 20, 25]:
    mean_auc = 0
    for n in range(10,201,10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        mean_auc += roc_auc_score(y_val, y_pred)
    mean_auc = mean_auc / 20
    print(m, round(mean_auc,3))    
    
rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1)
rf.fit(X_train, y_train)
features = dv.get_feature_names_out().tolist()
f_importance = pd.Series(rf.feature_importances_, index = features).sort_values(ascending=False)
print(f_importance)
y_pred = rf.predict(X_val)
print("Random Forest Classifier 2:")
print(roc_auc_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    'eta': 0.1, 
    'max_depth': 10,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'nthread': 8,    
    'seed': 1,
    'verbosity': 0
}

watchlist = [(dtrain, 'train'), (dval,'val')]

xgb.set_config(verbosity=0)
model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=200)
y_pred = model.predict(dval)
estimate = y_pred >= 0.5
print("XGBoost:")
print(round(roc_auc_score(y_val, estimate),3))
print(classification_report(y_val, estimate))

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)