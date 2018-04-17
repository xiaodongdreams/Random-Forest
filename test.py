import matplotlib.pyplot as plt
import sciranfor
import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import random

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

def test_num(filename):
    dataSet=sciranfor.loadCSV(filename)
    sciranfor.column_to_float(dataSet)
    sciranfor.replaceNanWithMean(dataSet)
    data0, data1 = splitData(dataSet)
    data1 = splitData1(data1, number=500)
    data = data0 + data1
    X,Y=sciranfor.proceData(data)
    X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.4,random_state=0)
    nums=np.arange(1,100,step=1)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        clf=RandomForestClassifier(n_estimators=num)
        clf.fit(X_train,Y_train)
        training_scores.append(clf.score(X_train,Y_train))
        testing_scores.append(clf.score(X_test,Y_test))
    print testing_scores
    ax.plot(nums,training_scores,label="Training Score")
    ax.plot(nums,testing_scores,label="Testing Score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5,1.05)
    plt.suptitle("RandomForestClassfier_Estimator")
#    plt.savefig("test_treenumbers.jpg")
    plt.show()
    
def test_max_depth(filename):
    dataSet=sciranfor.loadCSV(filename)
    sciranfor.column_to_float(dataSet)
    sciranfor.replaceNanWithMean(dataSet)
    data0, data1 = splitData(dataSet)
    data1 = splitData1(data1, number=500)
    data = data0 + data1
    X,Y = sciranfor.proceData(data)
    X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.4,random_state=0)
    nums=np.arange(1,40,step=1)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        clf=RandomForestClassifier(max_depth=num)
        clf.fit(X_train,Y_train)
        training_scores.append(clf.score(X_train,Y_train))
        testing_scores.append(clf.score(X_test,Y_test))
    ax.plot(nums,training_scores,label="Training Score")
    ax.plot(nums,testing_scores,label="Testing Score")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5,1.05)
    plt.suptitle("RandomForestClassfier_max_depth")
#    plt.savefig("test_max_depth.jpg")
    plt.show()
    
def test_max_features(filename):
    dataSet=sciranfor.loadCSV(filename)
    sciranfor.column_to_float(dataSet)
    sciranfor.replaceNanWithMean(dataSet)
    data0, data1 = splitData(dataSet)
    data1 = splitData1(data1, number=500)
    data = data0 + data1
    X, Y = sciranfor.proceData(data)
    X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.4,random_state=0)
    nums=np.linspace(0.01,1.0)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        clf=RandomForestClassifier(max_features=num)
        clf.fit(X_train,Y_train)
        training_scores.append(clf.score(X_train,Y_train))
        testing_scores.append(clf.score(X_test,Y_test))
    ax.plot(nums,training_scores,label="Training Score")
    ax.plot(nums,testing_scores,label="Testing Score")
    ax.set_xlabel("max_featrues")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5,1.05)
    plt.suptitle("RandomForestClassfier_max_features")
    #plt.savefig("test_max_features.jpg")
    plt.show()
