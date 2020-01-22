#initialization phase

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#%matplotlib inline
df=pd.DataFrame({
    'x':[2,5,8,1,2,6,3,8],
    'y':[3,6,7,4,2,7,4,6]
})
np.random.seed(200)
k=2
centroid={
    1:[2,3],
    2:[5,6]
    
    }
colmap={1:'r',2:'g'}

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color='k')
#print(centroid.keys())

for i in centroid.keys():
    plt.scatter(*centroid[i],color=colmap[i])
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()

def assignment(df,centroid):
    for i in centroid.keys():
        df['distance_from_{}'.format(i)]=(
            np.sqrt(
                (df['x'] - centroid[i][0])**2 + (df['y'] - centroid[i][1])**2
            )
        )
    centroid_distance_cols=['distance_from_{}'.format(i) for i in centroid.keys()]
    df['closest']=df.loc[:,centroid_distance_cols].idxmin(axis=1)
    df['closest']=df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color']=df['closest'].map(lambda x: colmap[x])
    return df

df=assignment(df,centroid)

print(df)
fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroid.keys():
    plt.scatter(*centroid[i],color=colmap[i])
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()

import copy
old_centroid=copy.deepcopy(centroid)
def update(k):
    for i in centroid.keys():
        centroid[i][0]=np.mean(df[df['closest']==i]['x'])
        centroid[i][1]=np.mean(df[df['closest']==i]['y'])
    return k
centroid=update(centroid)

fig=plt.figure(figsize=(5,5))
ax=plt.axes()
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroid.keys():
    plt.scatter(*centroid[i],color=colmap[i])
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()


#repeat assignment phase

df=assignment(df,centroid)

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroid.keys():
    plt.scatter(*centroid[i],color=colmap[i])
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()

while True:
    colsest_centroid=df['closest'].copy(deep=True)
    centroid=update(centroid)
    df=assignment(df,centroid)
    if colsest_centroid.equals(df['closest']):
        break
fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroid.keys():
    plt.scatter(*centroid[i],color=colmap[i])
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()
#print(df)



