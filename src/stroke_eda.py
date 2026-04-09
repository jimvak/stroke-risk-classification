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




mydata = pd.read_csv('healthcare-dataset-stroke-data.csv')


sns.countplot(mydata['gender'],label="Count")
plt.show()

sns.countplot(mydata['age'],label="Count")
plt.show()

sns.countplot(mydata['hypertension'],label="Count")
plt.show()

sns.countplot(mydata['heart_disease'],label="Count")
plt.show()

sns.countplot(mydata['ever_married'],label="Count")
plt.show()

sns.countplot(mydata['work_type'],label="Count")
plt.show()

sns.countplot(mydata['Residence_type'],label="Count")
plt.show()

sns.countplot(mydata['avg_glucose_level'],label="Count")
plt.show()

sns.countplot(mydata['bmi'],label="Count")
plt.show()

sns.countplot(mydata['smoking_status'],label="Count")
plt.show()

sns.countplot(mydata['stroke'],label="Count")
plt.show()



    
