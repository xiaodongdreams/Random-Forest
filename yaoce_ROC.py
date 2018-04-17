import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn.preprocessing import *
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from compiler.ast import flatten
from collections import Counter
from numpy import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation,decomposition
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from scipy import interp





dataSet=pd.read_csv('yaoce.csv')

dataSet=mat(dataSet)
print dataSet
m,n=dataSet.shape
#print m,n
Y_1=dataSet[:,0]
Y_2=dataSet[:,1]
#print Y_2

mean_tpr = 0.0
mean_fpr = linspace(0, 1, 100)
all_tpr = []

fpr,tpr,thresholds=roc_curve(Y_2,Y_1)
print fpr,tpr
mean_tpr+=interp(mean_fpr,fpr,tpr)
mean_tpr[0]=0.0
roc_auc=auc(fpr,tpr)
print roc_auc
plt.plot(fpr,tpr,lw=1,label='ROC(area=%0.4f)' %(roc_auc))
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck")

mean_tpr /= m

mean_tpr[-1] = 1.0
#print mean_fpr
mean_auc = auc(mean_fpr, mean_tpr)


# print mean_fpr,len(mean_fpr)
# print mean_tpr
'''
plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.4f)' % mean_auc, lw=2)
'''
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()