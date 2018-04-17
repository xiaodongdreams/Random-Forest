import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import *
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from compiler.ast import flatten
from collections import Counter

dataSet=pd.read_csv('all.csv')
dataSet=np.mat(dataSet)
m,n=dataSet.shape
dataSet=dataSet.tolist()

for i in range(m):


data_nor=[]
data_nan=[]
t=0
for i in range(m):
    for j in range(n):
        if np.isnan(dataSet[i][j])==1:
            data_nan.append(dataSet[i])
            t=1
            break
    if t==1 :t=0;continue
    data_nor.append(dataSet[i])

data_nor=np.mat(data_nor)
#print data_nor[0]
X_1=data_nor[:,0:10]
X_2=data_nor[:,12:17]
X=np.hstack((X_1,X_2))
#print X
Y_1=data_nor[:,10]
#Y_1=Y_1.tolist()
Y_2=data_nor[:,11]
#print Y_2[0]

#m,n=Y.shape
#print m,n
#print dataX[:10]
#print Y_1
#print flatten(Y_1.tolist())
X_random=

nums=range(1,50)
#for num in nums:
#    print num
clst=cluster.AgglomerativeClustering(n_clusters=10,linkage='ward')

predicted_labels=clst.fit_predict(X)
count=Counter(predicted_labels)
print count
