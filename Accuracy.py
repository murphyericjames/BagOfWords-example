def Accuracy(ClassP,ClassA,Docs):
    
    #here we calculate the accuracy of the predictor
    Ctally=0
    
    for i in range(Docs):
        #must off-set the predicted class by 1 since its runs from 0-19 rather than 1-20
        if ClassP[i]+1==ClassA[i]:
            Ctally += 1
            #print(Ctally)
            
    Acc=float(Ctally)/float(Docs)
    return Acc

def ClassAcc(ClassP, ClassA, Docs, classes):

    Ctally=[0]*classes
    AccClass=[0]*classes
    docscount=[0]*classes
    
    for i in range(Docs):
        #count proper normalization, i.e. number in class
        docscount[ClassA[i]-1] += 1
        #count number of matching predictions for each class - indexed by actual class
        if ClassP[i]+1==ClassA[i]:
            Ctally[ClassA[i]-1] += 1
            
            
    for j in range(classes):
        AccClass[j]=float(Ctally[j])/float(docscount[j])
        
    return AccClass

def Confusion(ClassP, ClassA, Docs, classes):

    Confuse=[[]]
    Contemp=[0]*classes
    
    #construct the form of the curious
    for i in range(classes):
        if i == 0:
            Confuse[i]=Contemp[:]
        else:
            Confuse.append(Contemp[:])
     
    print('halfway there')
    #now tally up the entries in the confusion
    for i in range(Docs):
        #account for offset again, now its the other way around used as an index
        Confuse[ClassA[i]-1][ClassP[i]] += 1
        
    return Confuse