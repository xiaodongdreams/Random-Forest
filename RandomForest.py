from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation,decomposition
from numpy import *

def RandomForest(trainX,trainY,testX):
    rf=RandomForestClassifier(n_estimators=20,max_depth=4)
    rf.fit(trainX,trainY)
    predict=rf.predict(testX)
    return predict
