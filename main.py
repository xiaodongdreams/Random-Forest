from sklearn import cross_validation
from numpy import *
import loadCSV
import proData
import RandomForest
import result

def main(filename,iteration=5):
    sum=0
    dataSet=loadCSV.loadCSV(filename)
    proData.column_to_float(dataSet)
    proData.replaceNanWithMean(dataSet)
 #   dataSet=decomposition.PCA(n_components=10)
    X,Y=proData.proceData(dataSet)
    for i in range(iteration):
        X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.4,random_state=0)
#        m=shape(Y_test)
#        print m
        predict=RandomForest.RandomForest(X_train,Y_train,X_test)

        accuracy=result.Predict(predict,Y_test)   
#        sum+=accur
        print('the %d iteration : %s'% (i,accuracy))
        r00,r01,r10,r11=result.Recall(predict,Y_test)
#        print r00,r01,r10,r11
 #       print('the recall is : %s'% float())
        
 #   print('mean : %s'% (sum(accur)/float(len(iteration))))
