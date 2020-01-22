import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
print(os.getcwd())
os.chdir("C:\\DWDM")
data=pd.read_csv("iris.csv")
print(data)
X=data[['sepal_length','sepal_width','petal_length','petal_width']]
y=data['class']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2,test_size=0.3)
sns.pairplot(data=data,hue='class',height=3)
plt.show()
dtclf=DecisionTreeClassifier(criterion='entropy')
dtclf.fit(X_train,y_train)
y_pred = dtclf.predict(X_test)
print(y_pred)
print(dtclf.score(X_test,y_test))
print(accuracy_score(y_test,y_pred))
dot_data=export_graphviz(dtclf,out_file=None,feature_names=data.drop('class',1).columns,class_names=data['class'].unique(),filled=True,rounded=True,special_characters=True)
graph= graphviz.Source(dot_data)
graph.render("C:\\DWDM\\iris_decision_tree")
cm = confusion_matrix(y_test, y_pred)
print(cm)
