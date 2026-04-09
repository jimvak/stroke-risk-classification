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
from sklearn.neighbors import KNeighborsClassifier




mydata = pd.read_csv('healthcare-dataset-stroke-data.csv')


feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
X = mydata[feature_names]

y = mydata['stroke']

#efarmogi labelencoder

le = LabelEncoder()

X['gender']= le.fit_transform(X['gender'])

X['ever_married']= le.fit_transform(X['ever_married'])

X['work_type']= le.fit_transform(X['work_type'])

X['Residence_type']= le.fit_transform(X['Residence_type'])

X['smoking_status']= le.fit_transform(X['smoking_status'])


#edo kanoume oti xreiazetai gia ton knn

feature_names_reg = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type','Residence_type','avg_glucose_level','smoking_status']
X_reg = mydata[feature_names_reg]

y_reg = mydata['bmi']

#efarmogi labelencoder

le_reg = LabelEncoder()

X_reg['gender']= le_reg.fit_transform(X_reg['gender'])

X_reg['ever_married']= le_reg.fit_transform(X_reg['ever_married'])

X_reg['work_type']= le_reg.fit_transform(X_reg['work_type'])

X_reg['Residence_type']= le_reg.fit_transform(X_reg['Residence_type'])

X_reg['smoking_status']= le_reg.fit_transform(X_reg['smoking_status'])


X_reg_train = pd.DataFrame(columns=feature_names_reg,index=range(4909))
y_reg_train=pd.Series()

count=0

#epilegoume ekeines tis eggrafes oi opoies den exoun naN sto bmi
#aytes einai oi eggrafes oi opoies tha xrisimopoihthoun gia tin ekapaideysi 
#tou montelou mas 
for i in range(len(X_reg)):
    if not np.isnan(y_reg[i]):
        X_reg_train['gender'].loc[count] = X_reg['gender'].loc[i]
        X_reg_train['age'].loc[count] = X_reg['age'].loc[i]
        X_reg_train['hypertension'].loc[count] = X_reg['hypertension'].loc[i]
        X_reg_train['heart_disease'].loc[count] = X_reg['heart_disease'].loc[i]
        X_reg_train['ever_married'].loc[count]=X_reg['ever_married'].loc[i]
        X_reg_train['work_type'].loc[count]=X_reg['work_type'].loc[i]
        X_reg_train['Residence_type'].loc[count] = X_reg['Residence_type'].loc[i]
        X_reg_train['avg_glucose_level'].loc[count]=X_reg['avg_glucose_level'].loc[i]
        X_reg_train['smoking_status'].loc[count]=X_reg['smoking_status'].loc[i]
        
        y_reg_train.loc[count] = y_reg.loc[i]
        count=count+1

        
        
lab_enc =LabelEncoder()
y_reg_train_encoded = lab_enc.fit_transform(y_reg_train)        

#dimiourgia montelou knn

classifier = KNeighborsClassifier(n_neighbors=5)

#ekpaideysi me knn
classifier.fit(X_reg_train, y_reg_train_encoded)



#diatreksi tou dataset
for i in range(len(X)):
    #vriskoume tis eggrafes opu exoun nan stin timi tou bmi
    if np.isnan(X['bmi'].loc[i]):
        #dedomena pou tha dosoume ston knn gia na ginoun predict
        myinput = X_reg.loc[i]
        #ginetai o katallilos metasximatismos gia na mporesei na epiteyxuei to prediction
        myinput=myinput.values.reshape(1,-1)
        #edo simplironetai i timi pou leipei me tin timi pou ginetai predict
        X['bmi'].loc[i] = classifier.predict(myinput)
        





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# # #dimiourgia montelou
clf=RandomForestClassifier(n_estimators=6000)

# # #ekpaideysi
clf.fit(X_train,y_train)

# # #test
y_pred=clf.predict(X_test)

precision = precision_score(y_test, y_pred, average='binary')

print('Precision: %.3f' % precision)

recall = recall_score(y_test, y_pred, average='binary')
print('Recall: %.3f' % recall)

score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % score)
