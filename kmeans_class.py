
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

#%matplotlib inline
df=pd.DataFrame({
    'x':[2,5,8,1,2,6,3,8],
    'y':[3,6,7,4,2,7,4,6]
})
kmeans=KMeans(n_clusters=2)
centroid=np.array([[2,3],[5,6]])
'''
kmeans.fit(df)
labels=kmeans.predict(df)
centroid=np.array([[2,3],[5,6]])
#print(type(centroid))

#centroid=[[2,3],[5,6]]
colmap={1:'r',2:'g'}
fig=plt.figure(figsize=(5,5))
colors=map(lambda x:colmap[x+1],labels)
color1=list(colors)
plt.scatter(df['x'],df['y'],color=color1,alpha=0.5,edgecolor='k')
for idx,centroid in enumerate(centroid):
    plt.scatter(*centroid,color=colmap[idx+1])
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()

'''
y_pred=kmeans.fit_predict(df)
df['cluster']=y_pred
print(df)
df1=df[df.cluster==0]
df2=df[df.cluster==1]
plt.scatter(df1.x,df1.y,color='green')
plt.scatter(df2.x,df2.y,color='red')
plt.scatter(centroid[:,0],centroid[:,1],color='yellow',marker="*",label=centroid)
plt.show()