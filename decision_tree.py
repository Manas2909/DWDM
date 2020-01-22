
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
print(os.getcwd())
os.chdir("C:\\DWDM")
data=pd.read_csv("2010-capitalbikeshare-tripdata.csv")
#print(data)
X=data[['Duration','Start station number','End station number']]
y=data['Member type']



#labelencoder_y =LabelEncoder()
#y = labelencoder_y.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2,test_size=0.3)
sns.pairplot(data=data,hue='Member type',height=3)
#plt.show()
dtclf=DecisionTreeClassifier(criterion='entropy')
dtclf.fit(X_train,y_train)
y_pred = dtclf.predict(X_test)
print(y_pred,"\n")
#print(dtclf.score(X_test,y_test),"\n")
print(accuracy_score(y_test,y_pred),"\n")

features=data.drop(['Start date','End date','Start station','End station','Bike number','Member type'],axis=1)
print(features.columns)

dot_data=export_graphviz(dtclf,out_file=None,feature_names=features.columns,class_names=data['Member type'].unique(),filled=True,rounded=True,special_characters=True)
graph= graphviz.Source(dot_data)
graph.render("C:\\DWDM\\member_try")

cm = confusion_matrix(y_test, y_pred)
print(cm)
