from numpy import *



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

def proceData(dataSet):
    dataX=[];dataY=[]
    length=len(dataSet)
    feature=len(dataSet[0])
    for i in range(length):
        Temp = dataSet[i][0:feature - 1]
        dataX.append(Temp)
    dataY=[row[-1] for row in dataSet]
    return dataX,dataY
            
