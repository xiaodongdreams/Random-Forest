#-*- coding: utf-8 -*-
# function read CSV file to dataSet

import csv
from numpy import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation,decomposition,svm,preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from scipy import interp
import random

def loadCSV(filename):
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet

def column_to_float(dataSet):
    featLen=len(dataSet[0])
    for data in dataSet:
        for column in range(featLen):
            if data[column]=='':
                data[column]='nan';continue
            data[column]=float(data[column])

def replaceNanWithMean(dataSet):
    m=len(dataSet)
    n=len(dataSet[0])
    for i in range(n-1):
        k=0;sum=0
        for j in range(m):
            if dataSet[j][i]=='nan':continue
            else:
                k+=1;sum+=dataSet[j][i]
        for j in range(m):
            if dataSet[j][i]=='nan':dataSet[j][i]=float(sum/k)

def splitData(dataSet):
    data0=[]
    data1=[]
    m=len(dataSet)
    for i in range(m):
        if dataSet[i][-1] == 0:
            data0.append(dataSet[i])
        elif dataSet[i][-1] == 1:
            data1.append(dataSet[i])
    return data0,data1

def splitData1(dataSet,number=750):
    data1=random.sample(dataSet,number)
    return data1

def proceData(dataSet):
    dataX=[];dataY=[]
    length=len(dataSet)
    feature=len(dataSet[0])
    for i in range(length):
        Temp = dataSet[i][0:feature - 1]
        dataX.append(Temp)
    dataY=[row[-1] for row in dataSet]
    return dataX,dataY
            
def RandomForest(trainX,trainY,testX):
    rf=RandomForestClassifier(n_estimators=30)
    rf.fit(trainX,trainY)
    predict=rf.predict(testX)
    return predict

def Predict(predict_values, actural):
    correct = 0
    #    print predict_values
    for i in range(len(actural)):
        if actural[i] == predict_values[i]:
            correct += 1

    return correct / float(len(actural))


def Recall(predict_values, actural):
    r00 = 0;r01 = 0;r10 = 0;r11 = 0
    for i in range(len(actural)):
        if actural[i] == 0 and predict_values[i] == 0:
            r00 += 1
        elif actural[i] == 0 and predict_values[i] == 1:
            r01 += 1
        elif actural[i] == 1 and predict_values[i] == 0:
            r10 += 1
        elif actural[i] == 1 and predict_values[i] == 1:
            r11 += 1
    return r00, r01, r10, r11



#def main(filename):
#    sum = 0
dataSet = loadCSV('all_1.csv')
#print dataSet
column_to_float(dataSet)
replaceNanWithMean(dataSet)
X_1, Y_1 = proceData(dataSet)
X_1=preprocessing.scale(X_1)


'''
X=mat(X)
Y=array(Y)


accuracy=[]
r00=[]; r01=[]; r10=[]; r11=[]

cv = StratifiedKFold(Y, n_folds=5)
rf = RandomForestClassifier(n_estimators=20, max_depth=6)      #n_estimators=20, max_depth=5
for i, (train, test) in enumerate(cv):
    rf.fit(X[train], Y[train])
#    print rf.feature_importances_
    predict=rf.predict(X[test])
    accuracy.append(Predict(predict, Y[test]))
    r00_, r01_, r10_, r11_ = Recall(predict, Y[test])
    r00.append(r00_);r01.append(r01_);r10.append(r10_);r11.append(r11_)
        #        sum+=accur
for i in range(5):
    print('the %d iteration : %s' % (i, accuracy[i]))

#    print r00,r01,r10,r11
        #       print('the recall is : %s'% float())

        #   print('mean : %s'% (sum(accur)/float(len(iteration)))



accuracy=[]
r00=[]; r01=[]; r10=[]; r11=[]

    #plot ROC
cv = StratifiedKFold(Y, n_folds=5)
rf=RandomForestClassifier()
mean_tpr=0.0
mean_fpr=linspace(0,1,100)
all_tpr=[]

for i,(train,test) in enumerate(cv):
    rf.fit(X[train], Y[train])
    predict = rf.predict(X[test])
    accuracy.append(Predict(predict, Y[test]))
    r00_, r01_, r10_, r11_ = Recall(predict, Y[test])
    r00.append(r00_);r01.append(r01_);r10.append(r10_);r11.append(r11_)
    fpr,tpr,thresholds=roc_curve(Y[test],predict)
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area=%0.4f)' %(i,roc_auc))

for i in range(5):
    print('the %d iteration : %s' % (i, accuracy[i]))
for i in range(5):
    print('the %d iteration :r00:%s,r01:%s,r10:%s,r11:%s'%(i,r00[i],r01[i],r10[i],r11[i]))


plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck")
mean_tpr /= len(cv)  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
# 画平均ROC曲线
# print mean_fpr,len(mean_fpr)
# print mean_tpr
plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.4f)' % mean_auc, lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''
accuracy=[]
accuracy_1=[]
r00=[]; r01=[]; r10=[]; r11=[]
rf=RandomForestClassifier()
mean_tpr=0.0
mean_fpr=linspace(0,1,100)
all_tpr=[]

X, Y = proceData(dataSet)
#print len(X)
X = preprocessing.scale(X)
for i in range(5):
    #data0, data1 = splitData(dataSet)
    #print len(data1)
    #data0=data0+data0+data0+data0
    #data1 = splitData1(data1, number=500)
    #print len(data1)
    #print len(data0)
    #data = data0 + data1

    #print len(data)
    #   dataSet=decomposition.PCA(n_components=10)

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)

    data_train=vstack((X_train.T,Y_train)).T
    print data_train
    data0, data1 = splitData(data_train)

    print len(data1)
    data0=data0+data0+data0
    #data0 = data0 + data0 + data0 + data0
    print len(data0)
    #data=data0+data1
    data=splitData1(data1,500)+data0
    #data=data0+data1
    #print len(data0)
    #print len(data1)
    X_train,Y_train=proceData(data)
    print len(Y_train)
    print len(X_train)

        #        m=shape(Y_test)
        #        print m

    #predict = RandomForest(X_train, Y_train, X_test)
    #cls=svm.SVC(kernel='rbf')
    #cls.fit(X_train,Y_train)
    #predict=cls.predict(X_test)
    rf = RandomForestClassifier(n_estimators=30)
    rf.fit(X_train, Y_train)
    predict = rf.predict(X_test)
    pre=rf.predict(X_train)
    print predict

    accuracy.append(Predict(predict, Y_test))
    accuracy_1.append(Predict(pre,Y_train))
        #        sum+=accur
#    print('the %d iteration : %s' % (i, accuracy))
    r00_, r01_, r10_, r11_ = Recall(predict, Y_test)
    r00.append(r00_);r01.append(r01_);r10.append(r10_);r11.append(r11_)
    fpr, tpr, thresholds = roc_curve(Y_test, predict)
    print fpr
    print tpr
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area=%0.4f)' % (i, roc_auc))

for i in range(5):
    print('the %d iteration : %s' % (i, accuracy[i]))
for i in range(5):
    print('the %d iteration :r00:%s,r01:%s,r10:%s,r11:%s' % (i, r00[i], r01[i], r10[i], r11[i]))
for i in range(5):
    print('the %d iteration : %s' % (i, accuracy_1[i]))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label="Luck")
mean_tpr /= 5  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
# 画平均ROC曲线
# print mean_fpr,len(mean_fpr)
# print mean_tpr
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.4f)' % mean_auc, lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
        #        print r00,r01,r10,r11
        #       print('the recall is : %s'% float())

        #   print('mean : %s'% (sum(accur)/float(len(iteration)))
