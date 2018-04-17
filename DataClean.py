import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import scipy as sp
from sklearn.preprocessing import *
'''
dataSet=[]
with open('all.csv', 'r') as file:
    csvReader = csv.reader(file)
    for line in csvReader:
        dataSet.append(line)
#print dataSet


Data=np.mat(dataSet)
Data=Data.T
m,n=np.shape(Data)
print m,n
#Data[2]=Data[2]*10
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
nums=np.arange(0,n,step=1)
nums=np.mat(nums)
print np.shape(nums.T)
print np.shape(Data[0].T)
ax.plot(nums.T,Data[1].T,label="CO")
ax.plot(nums.T,Data[3].T,label="HC")
ax.plot(nums.T,Data[5].T,label="NO")
ax.set_xlabel("numbers")
ax.set_ylabel("")
ax.legend(loc="upper left")
plt.suptitle("Exhaust")
plt.show()
#plt.savefig("Exhaust.jpg")
'''

dataSet=pd.read_csv('all.csv')
dataSet=np.mat(dataSet)
m,n=dataSet.shape
print m,n
for i in range(n):
    print i,sp.sum(sp.isnan(dataSet[:,i]))

