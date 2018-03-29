def ReadLabels(trainfile):

    import csv
    n=0
    csvfile = open(trainfile)
    classreader = csv.reader(csvfile)
    #Testlabels[:]=[int(x) for x in classreader] 
    Testlabels=[[]]
    
    for row in classreader:
        for column in row:
            if n==0:
                x=int(column)
                Testlabels[0] = x
                n = n + 1
            else:
                x=int(column)
                Testlabels.append(x)
                n=n+1
                
    return Testlabels

def Calcpriors(labels, classes):
    
    tally=[0]*classes
    Priors=[0]*classes
    totcounted=0
    
    for speccase in labels:
        for classnum in range(classes+1):
            #print(speccase, type(speccase), type(classnum))
            if speccase==classnum:
                #calculate number of specific occurences
                totcounted = totcounted + 1
                #calculate total, i.e. normalization constant
                tally[classnum-1] = tally[classnum-1] + 1
                #print(classnum)
    
    #print(tally)
    Priors[:] = [float(x) / float(totcounted) for x in tally]
    
    return Priors