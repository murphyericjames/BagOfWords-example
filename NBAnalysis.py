def LogifytheProbs(Prior, PCest, classes):
    #have to account for 0 entries for the MLE
    #using largenum=math.log(10**(-100)) can help
    import math
    logPCest=[[]]
    
    #calculate the initial logarithms
    logPrior = [math.log(k) for k in Prior]
    #now for the conditionals    
    for i in range(classes):
        CondProbs = PCest[i][:]
        #set tolerance so NaN is not received from log, but a huge penalty is still applied
        tol=10**(-100)
        for k in range(len(CondProbs)):
            CondProbs[k]=max(CondProbs[k],tol)
        
        logCondProbs = [math.log(k) for k in CondProbs]
        if(i==0):
            logPCest[i]=logCondProbs[:]
        else:
            logPCest.append(logCondProbs[:])
            
    return logPCest, logPrior

def LogNumeratorInNB(wordcount):
    import math
    #this calculates the logarithm of the numerator in the Naive Bayes Classifier
    #the summation inside the factorial
    Ndksum = math.fsum(wordcount)

    #now the ordered sum that the factorial becomes
    Numerator = math.fsum(math.log(x) for x in xrange(1,int(Ndksum+1)))
    
    return Numerator

def LogDenominator(wordcount):
    import math
    
    logdenom=0

    #this calculates the logarithm of the denominator - it will be negative in the eventual summation
    for i in wordcount:
        if(i==0):
            #this is important, zero counts do not add to the log likelihood (0!=1)
            kthlogterm=0
        else:
            kthlogterm=math.fsum(math.log(x) for x in xrange(1,i+1))
        
        logdenom = logdenom + kthlogterm
    
    return logdenom

def LogPostSum(logPCest, wordid, wordcount, class1, Vocabulary):
    
    Logsum=0
    j=0
        
    tol=10**(-10)    
    
    for i in wordid:
        #must make sure that we can handle new words that weren't in the training set
        #we simply ignore them
        if i <= Vocabulary:
            #only count if it the word appears to avoid problems of large numbers appearing from the log term
            if wordcount[j] > tol:
                Logsum = Logsum + wordcount[j]*logPCest[class1][i-1]
            j=j+1
        else:
            break
          
    return Logsum
        

def FindDocumentNumber(testlabels):

    DocNum=len(testlabels)
    return DocNum

def FindEntries(Document, docid):
    #want to find indices for elements that have a certain value
    #namely, which elements in docid belong to which document
    #will use this to sort through wordids and wordcounts
    GoodRange=[]
    
    #old method was too slow to perform 11000 times
    #GoodRange=[i for i,x in enumerate(docid) if x==Document]
    
    Firstentry=docid.index(Document)
    Entry=Firstentry
    while docid[Entry]==Document:
        GoodRange.append(Entry)
        Entry=Entry+1
        #must break out if index is exceeded
        if Entry == len(docid):
            break
    
    return GoodRange

def StupidSlice(Range1, A):
    #equivalent to slicing when you just have a range
    B=[[]]
    c=0
    for i in Range1:
        if(c==0):
            B[c]=A[i]
            c=c+1
        else:
            B.append(A[i])
    return B

            
def CalcPosteriors(docid, testlabels, wordid, wordcount, classes, PCest, Prior, Vocabulary):
    import timeit
    #this is the main component of this so-called module
    #print('1')
    #define the eventual posteriors
    Pxw=[[]]
    Pwgivenx=[0]*(classes)
    Range1=[]
    
    #print('2')
    #take care of the trained conditional probabilities
    logPCest, logPrior = LogifytheProbs(Prior, PCest, classes)
    Docs=FindDocumentNumber(testlabels)
    #print('3')
    i=1
    while i <= Docs:
       
        #find the entries in the vector that are from a given document
        #start_time = timeit.default_timer()
        Range1=FindEntries(i,docid)
        #elapsed = timeit.default_timer() - start_time
        #print('elapsed in finding range', elapsed)
        
        #if i==1:
        #    print('4')
        
        #slice the wordids and wordcounts
        #start_time = timeit.default_timer()
        wordidtemp=StupidSlice(Range1, wordid)
        wordcttemp=StupidSlice(Range1, wordcount)
        Range1=[]
        #elapsed = timeit.default_timer() - start_time
        #print('elapsed in stupid slicingfirst', elapsed)
        #if i==1:
        #    print('5')
        #Calc the log denom and numerators
        #actually unnecessary because its the same for each prospective class
        #Logdenom=LogDenominator(wordcttemp)
        #Lognum=LogNumeratorInNB(wordcttemp)
        
        #now find the class specific pieces
        for j in range(classes):
            Postsum=LogPostSum(logPCest, wordidtemp, wordcttemp, j, Vocabulary)
            #print(Postsum, logPrior[j], Lognum, Logdenom)
            #add all 4 pieces together logPrior, 
            #Pwgivenx[j]= Postsum + logPrior[j] + Lognum - Logdenom
            
            Pwgivenx[j]= Postsum + logPrior[j]
            #if j ==0:
            #    Pwgivenx[j]= Postsum + logPrior[j] + Lognum - Logdenom
            #else:
            #    Pwgivenx.append(Postsum + logPrior[j] + Lognum - Logdenom)
     
        #start_time = timeit.default_timer()
        if i ==1:
            Pxw[i-1]=Pwgivenx[:]
            #print('6')
        else:
            Pxw.append(Pwgivenx[:])
        #elapsed = timeit.default_timer() - start_time
        #print('elapsed in slicing', elapsed)
        
        Pwgivenx=[0]*(classes)
        #print(i, Docs)
        i=i+1
        
    #print(i,Docs)    
    return Pxw, Docs

def FindBestPxw(Pxw, Docs, classes):
    import time
    
    #initialize the classification vector
    Pxclass=[0]*Docs
    Pxwclass=[]
    
    #print('in the latest')
    for i in range(Docs):
        #start_time = time.time()
        Pxwclass=Pxw[i][:]
        #print(Pxwclass)
        #elapsed=time.time()-start_time
        #print('time to slice', elapsed)
        
        #start_time = time.time()
        Pxbest=max(Pxwclass)
        #elapsed=time.time()-start_time
        #print('time to find max out of entries in classes', elapsed)
        
        
        #find the class that the document is in
        #start_time = time.time()
        Pxclass[i]=Pxwclass.index(Pxbest)
        #elapsed=time.time()-start_time
        #print('time to to call entry that is maximum', elapsed)
        
        Pxwclass=[]
        #if i == 0:
        #    print(Pxclass[i])
        #    exit
        #print(i, Pxclass[i]) 
    
    #print('got out of the loop')
    #print(Pxclass)
    return Pxclass