def Predict(predict_values,actural):
    correct=0
#    print predict_values
    for i in range(len(actural)):
        if actural[i]==predict_values[i]:
            correct+=1
            
    return correct/float(len(actural))   

def Recall(predict_values,actural):
    r00=0;r01=0;r10=0;r11=0
    for i in range(len(actural)):
        if actural[i]==0 and predict_values[i]==0:
            r00+=1
        elif actural[i]==0 and predict_values[i]==1:
            r01+=1
        elif actural[i]==1 and predict_values[i]==0:
            r10+=1
        elif actural[i]==1 and predict_values[i]==1:
            r11+=1
    return r00,r01,r10,r11
