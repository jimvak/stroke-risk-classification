import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import cm
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score





mydata = pd.read_csv('healthcare-dataset-stroke-data.csv')


feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type','Residence_type','avg_glucose_level','smoking_status']
X = mydata[feature_names]

y = mydata['stroke']

#implementing labelencoder

le = LabelEncoder()

X['gender']= le.fit_transform(X['gender'])

X['ever_married']= le.fit_transform(X['ever_married'])

X['work_type']= le.fit_transform(X['work_type'])

X['Residence_type']= le.fit_transform(X['Residence_type'])

X['smoking_status']= le.fit_transform(X['smoking_status'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#creating the RandomForestClassifier model
clf=RandomForestClassifier(n_estimators=5000)

#training
clf.fit(X_train,y_train)

#testing
y_pred=clf.predict(X_test)

precision = precision_score(y_test, y_pred, average='binary')

print('Precision: %.3f' % precision)

recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.3f' % recall)

score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % score)
