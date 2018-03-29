def ReadTraingdata(datafile):

    import csv
    trainfile = open(datafile)
    datareader = csv.reader(trainfile)
              
    n=1
    docid=[[]]
    wordid=[[]]
    wordcount=[[]]
    #read the data for formulating the 
    for row in datareader:
        if n==1:
            docid[0] = int(row[0])
            wordid[0] = int(row[1])
            wordcount[0] = int(row[2])
            n=n+1
        else:
            docid.append(int(row[0]))
            wordid.append(int(row[1]))
            wordcount.append(int(row[2]))
            
    return docid, wordid, wordcount

def CountTheWords(docid, wordid, wordcount, trainlabels, classes):
    
    Totwordcase=[[0]]
    Totwords=[[]]
    #count for entries in the training data
    j=0
    #count for the documents with words - must start at 1
    i=1
    #count for distinct words
    k=0
    
    #cycle over all classes
    for h in range(classes):
        #cycle over all training examples, i.e. documents, in the class
        while trainlabels[i-1] <= h+1:
            #cycle over all distinct words in the document
            while docid[j] <= i:
                #set initialization case
                if(k==0):
                    Totwordcase[k]=wordcount[k]
                    #go forward in distinct word count and position in training entry
                    k=k+1
                    j=j+1
                    
                #now case where a new word is found
                if(wordid[j]>k):
                    Totwordcase.append(wordcount[k])
                    k=k+1
                    j=j+1

                #case where a new word is not found, add to the existing wordcount
                else:
                    Totwordcase[wordid[j]-1] = Totwordcase[wordid[j]-1] + wordcount[j]
                    j=j+1
                
                #if(h==19):
                #    print(j,len(docid))
                #exit loop before end of word list is achieve
                if(j == len(docid)):
                    #print('Das Erstes ist gebrochen')
                    break
                    
            i=i+1
            #exit loop if end of documents is achieved
            if(i == len(trainlabels)):
                #print('Das Zweitens ist gebrochen')
                break
            
        #print('made it through')
        #create a list of these lists to pass to the constructor of the conditional pij matrices
        if(h==0):
            Totwords[0]=Totwordcase[:]
        else:
            Totwords.append(Totwordcase[:])
        
        #set all values in the Totwordcase to zero to go over the next case - want to retain the structure,
        #i.e. number of previous entries without overwriting stuff - need to slice when appending
        Totwordcase[:]=[0]*len(Totwordcase)
        
    #print(k)
    #extract total number of words  in the vocabulary
    Vocabulary=k

    #print(Vocabulary)
    return Totwords, Vocabulary

def ConstructConditionalProbMLE(Vocabulary, Totwords, classes):
    #Maximum Likelihood estimation of conditional probability matrices
    
    
    #the total number of words in a case
    nwords=[0]*(classes)
    #the number of a specific word in a class/total words in a class
    PMLEclass=[0]*Vocabulary
    PMLE=[[]]
    
    #first get total words in the documents in a class
    for i in range(classes):
        #get total words in a class
        Totwordcase=Totwords[i]
        
        for j in Totwordcase:
            nwords[i]=nwords[i]+j
            
        m=0
        for k in Totwordcase:
            PMLEclass[m]= float(k) / float(nwords[i])
            m=m+1
        #calculate the remaining values which had 0 entries
        while m<len(PMLEclass):
            PMLEclass[m]= (1) / float(nwords[i]+Vocabulary)
            m=m+1
            
            
        #clear list for use in next class
        Totwordcase=[]
        
        #make list of lists to contain prediciton of conditional probability
        if(i==0):
            PMLE[0]=PMLEclass[:]
        else:
            PMLE.append(PMLEclass[:])
        
        PMLEclass=[0]*Vocabulary
    
    #print(PMLE[19]) 
    return PMLE

def ConstructConditionalProbBE(Vocabulary, Totwords, classes):
    #Bayesian estimation of conditional probability matrices
    
    #the total number of words in a case
    nwords=[0]*(classes)
    #the number of a specific word in a class/total words in a class
    PBEclass=[0]*Vocabulary
    PBE=[[]]
    
    #first get total words in the documents in a class
    for i in range(classes):
        #get total words in a class
        Totwordcase=Totwords[i]
        
        for j in Totwordcase:
            nwords[i]=nwords[i]+j
            
        m=0
        for k in Totwordcase:
            PBEclass[m]= float(k+1) / float(nwords[i]+Vocabulary)
            m=m+1
        #calculate the remaining values which had 0 entries
        while m<len(PBEclass):
            PBEclass[m]= (1) / float(nwords[i]+Vocabulary)
            m=m+1
            
        #clear list for use in next class
        Totwordcase=[]
        
        #make list of lists to contain prediciton of conditional probability
        if(i==0):
            PBE[0]=PBEclass[:]
        else:
            PBE.append(PBEclass[:])
        
        PMLEclass=[0]*Vocabulary
        
    #print(PBE[19])    
    return PBE