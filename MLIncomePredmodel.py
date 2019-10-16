import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_absolute_error
from math import sqrt

##Reading of files
X1=pd.read_csv(r'tcd ml 2019-20 income prediction training (with labels).csv')
X2=pd.read_csv(r'tcd ml 2019-20 income prediction test (without labels).csv')
X3=X1.append(X2)
print(X3)

##Data Preprocessing
X3.isnull().sum()
X3=X3.drop(["Wears Glasses","Hair Color","Body Height [cm]","Instance"],axis=1)
X3['Year of Record'] = X3['Year of Record'].ffill(axis=0) ##
X3['Profession'] = X3['Profession'].ffill(axis=0)
X3.Gender=X3.Gender.replace('0','other')
X3['Gender'] = X3['Gender'].ffill(axis=0)
#numerical values
imp=SimpleImputer(strategy='mean')
X3['Age']=imp.fit_transform(X3.Age.values.reshape(-1,1)) ##
X3['University Degree']=X3['University Degree'].replace('0','No') ##
X3['University Degree'] = X3['University Degree'].ffill(axis=0)
X3['Gender'] = X3['Gender'].map({'male': 0, 'female': 1, 'other': 2, 'unknown': 3}).astype(int)
X3['University Degree'] = X3['University Degree'].map({'Bachelor': 1, 'Master': 2, 'PhD': 3, 'No': 0}).astype(int) ##
X3['Income in EUR'] = abs(X3['Income in EUR']) ##
X3['Income in EUR'] = np.log(X3['Income in EUR'])
X3.dtypes
df=pd.DataFrame(X3)

##Data Encoding
labelencoder_x=LabelEncoder()
df['Profession']=labelencoder_x.fit_transform(df['Profession'])

labelencoder_x=LabelEncoder()
df['Country']=labelencoder_x.fit_transform(df['Country'])
Country_en=OneHotEncoder(sparse=False,handle_unknown='ignore')
N=Country_en.fit_transform(df.Country.values.reshape(-1,1))
dfOneHot_N=pd.DataFrame(N, columns = ["Country_"+str(int(i)) for i in range(N.shape[1])])

X4=df.iloc[0:111992,:]
O4=dfOneHot_N.iloc[0:111992,:]
X4=pd.concat([X4, O4], axis=1)
X4=X4.drop(["Income"],axis=1)

X5=df.iloc[111993:185223,:]
O5=dfOneHot_N.iloc[111993:185223,:]
O5=O5.reset_index(drop=True)
X5=pd.concat([X5, O5], axis=1)
X5=X5.drop(["Income in EUR"],axis=1)

X=X4.loc[:, X4.columns!='Income in EUR']
y=X4['Income in EUR']

##Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=500)
#from sklearn.model_selection import KFold
#kfold = KFold(n_splits=100, shuffle=False, random_state=0)
#X=np.array(X.copy())
#y=np.array(y.copy())
#for train_index, test_index in kfold.split(X):  
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]

##Data training model
#from sklearn.ensemble import RandomForestRegressor
#lm=RandomForestRegressor(n_estimators = 1000, random_state = 42)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
predictions=np.exp(predictions)    
plt.scatter(np.exp(y_test),predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print(predictions)
from sklearn import metrics

##Calculating root mean square error
print('MAE :'," ", metrics.mean_absolute_error(np.exp(y_test),abs(predictions)))
print('MSE :'," ", metrics.mean_squared_error(np.exp(y_test),abs(predictions)))
print('RMAE :'," ", sqrt(mean_squared_error(np.exp(y_test),abs(predictions))))

##Data Prediction
dfOneHot_N.isnull().sum()
z_test=X5.loc[:, X5.columns!='Income']
X5['Income'] = lm.predict(z_test)
X5['Income']=np.exp(X5['Income'])
X5['Income']=abs(X5['Income'])
print(X5['Income'])
X5.to_csv(r'C:\Users\tanvi\OneDrive\Documents\ML\tcdml1920_income_ind\tcd ml 2019-20 income prediction submission file.csv')
