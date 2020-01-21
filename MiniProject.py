#Project: 311 Service Request
# Author :  Manas Kumar Mukul 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import math

from sklearn.metrics import accuracy_score

print(os.getcwd())
os.chdir("C:\\DWDM")
f_read=pd.read_csv("311_Service_Requests_from_2010_to_Present.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
print(f_read.shape)
print(f_read.info())
print(f_read.columns)
data_set=f_read.loc[:,['Created Date', 'Closed Date', 'Agency', 'Complaint Type','Location Type','City',
                 'Facility Type','Status', 'Due Date','Resolution Action Updated Date','Borough','Park Borough']]
print(data_set)
data_set.columns=['Created_Date', 'Closed_Date', 'Agency', 'Complaint_Type','Location_Type','City',
                 'Facility_Type','Status', 'Due_Date','Resolution_Action_Upd_Date','Borough','Park_Borough']
print(data_set)
print(data_set.isnull().sum())
city_complaint_type=data_set.groupby(['City','Complaint_Type'])
print(city_complaint_type.size())
print(data_set['Complaint_Type'].unique())
complaint_type = data_set.groupby('Complaint_Type')
compaint_analysis=complaint_type.size()
df=compaint_analysis.to_frame().reset_index()
df.columns=['Complaint_Type','FREQ']
print(df.columns)
sort_1=df.sort_values('FREQ',ascending=False)
print(sort_1)
top10=sort_1.head(10)
print(top10)
x=range(10)
plt.bar(top10.Complaint_Type, top10.FREQ)
plt.xticks(x, top10.Complaint_Type, rotation='vertical')
plt.legend(['complaint'])
plt.show()
print(data_set.dtypes)
print(data_set.isnull().sum())
data_set['City']=data_set['City'].fillna('Unknown')
a=data_set['Facility_Type'].mode()[0]
data_set['Facility_Type']=data_set['Facility_Type'].fillna(a)
data_set['Location_Type']=data_set['Location_Type'].fillna(data_set['Facility_Type'].mode()[0])
lbe=LabelEncoder()
data_set['Agency']=lbe.fit_transform(data_set['Agency'])
data_set['Complaint_Type']=lbe.fit_transform(data_set['Complaint_Type'])
data_set['Location_Type']=lbe.fit_transform(data_set['Location_Type'])
data_set['City']=lbe.fit_transform(data_set['City'])
data_set['Facility_Type']=lbe.fit_transform(data_set['Facility_Type'])
data_set['Borough']=lbe.fit_transform(data_set['Borough'])
data_set['Park_Borough']=lbe.fit_transform(data_set['Park_Borough'])
data_set['Status']=lbe.fit_transform(data_set['Status'])
print(data_set.describe())
print(data_set.dtypes)
x_train=data_set.drop(['Created_Date','Closed_Date','Due_Date','Resolution_Action_Upd_Date','Agency'],axis=1,inplace = True)
x=data_set.iloc[:,0:5]
y=data_set.iloc[:,6]
print(y.isnull().sum())
xtrain,xtest,ytrain,ytest = model_selection.train_test_split(x,y,test_size=0.3,random_state=42)
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)
ss= StandardScaler()
x_train=ss.fit_transform(xtrain)
x_test=ss.fit_transform(xtest)

print(len(y))
print(math.sqrt(len(ytest)) )
category= KNeighborsClassifier(n_neighbors=20)
category.fit(x_train,ytrain)
ypredict=category.predict(x_test)
print(ypredict)

accuracy_score(ytest,ypredict)