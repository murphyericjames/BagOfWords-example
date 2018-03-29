#the os.path piece turns forward slashes into backwards ones to make designating path names easier
import os
import gc
#gc.enable()

#load the necessary files
testlabel=os.path.normpath("20newsgroups/test_label.csv")
trainlabel=os.path.normpath("20newsgroups/train_label.csv")
traindata=os.path.normpath("20newsgroups/train_data.csv")
testdata=os.path.normpath=("20newsgroups/test_data.csv")

#define the number of classes of materials
classes=20

#calculate priors
import ClassPrior
print('Read the labeled training data')
Trainlabels= ClassPrior.ReadLabels(trainlabel)
print('Read the labeled testing data')
Testlabels= ClassPrior.ReadLabels(testlabel)

print('Calculating Prior Probabilities from training data')
Prior= ClassPrior.Calcpriors(Trainlabels, classes)
print('Class', 'Prior')
for j in range(classes):
    print(j+1, Prior[j])

#construct the conditional probabilities
import ConditionalProb
print('Constructing conditional probabilities p(w|\omega):')
print('Read training Data')
docid, wordid, wordcount= ConditionalProb.ReadTraingdata(traindata)
print('Counting Training Data')
Totwords, Vocab= ConditionalProb.CountTheWords(docid, wordid, wordcount, Trainlabels, classes)
print('Calculating Max Likelihood estimate')
PMLEest = ConditionalProb.ConstructConditionalProbMLE(Vocab, Totwords, classes)
print('Calculating Bayesian estimate')
PBEest = ConditionalProb.ConstructConditionalProbBE(Vocab, Totwords, classes)

#Classify the training set: using Bayesian estimate first
import NBAnalysis
#step one: construct the Posterior estimates
print('Calculating Posterior estimates')
Pxw, Docs = NBAnalysis.CalcPosteriors(docid, Trainlabels, wordid, wordcount, classes, PBEest, Prior, Vocab)
#Docs is total number of documents
print('Select best class estimate for prediction')
#step two: choose best estimate
TrainClassEst=NBAnalysis.FindBestPxw(Pxw, Docs, classes)

import Accuracy
print('Calculating class Accuracy for Training set')
ClassAccTr = Accuracy.ClassAcc(TrainClassEst, Trainlabels, Docs, classes)
print('Class', 'Accuracy')
for j in range(classes):
    print(j+1, ClassAccTr[j])

print('Calculating Overall Accuracy for Training set')
#step three: calculate Accuracy
AccuTrain = Accuracy.Accuracy(TrainClassEst, Trainlabels, Docs)
print('Overall Accuracy of the BE on the Training Data =', AccuTrain)
#step four: confusion matrix
print('Now calculating the confusion matrix for the training data')
ConfuseTrain = Accuracy.Confusion(TrainClassEst, Trainlabels, Docs, classes)
print('Confusion matrix entries through (4,4)')
print(ConfuseTrain[1][1], ConfuseTrain[1][2], ConfuseTrain[1][3], ConfuseTrain[1][4])
print(ConfuseTrain[2][1], ConfuseTrain[2][2], ConfuseTrain[2][3], ConfuseTrain[2][4])
print(ConfuseTrain[3][1], ConfuseTrain[3][2], ConfuseTrain[3][3], ConfuseTrain[3][4])
print(ConfuseTrain[4][1], ConfuseTrain[4][2], ConfuseTrain[4][3], ConfuseTrain[4][4])

##########################################################################################
print('Beginning to look at Test data using BE')
print('Read test Data')
tdocid, twordid, twordcount= ConditionalProb.ReadTraingdata(testdata)
print(len(tdocid), len(twordid), len(twordcount))
print(len(docid), len(wordid), len(wordcount))
print(max(wordid), max(twordid))
#Classify the testing set: using Bayesian estimate first
#step one: construct the Posterior estimates
print('Calculating Posterior estimates for test data')
tPxw, tDocs = NBAnalysis.CalcPosteriors(tdocid, Testlabels, twordid, twordcount, classes, PBEest, Prior, Vocab)
#step two: choose best estimate
print('Select best class estimate for prediction for testdata')
TestClassEst=NBAnalysis.FindBestPxw(tPxw, tDocs, classes)
#step three: calculate Accuracy
print('Calculating class Accuracy for Test set with BE')
ClassAccTest = Accuracy.ClassAcc(TestClassEst, Testlabels, tDocs, classes)
print('Class', 'Accuracy')
for j in range(classes):
    print(j+1, ClassAccTest[j])

print('Calculating Overall Accuracy for Test set with BE')
#step three: calculate Accuracy
AccuTest = Accuracy.Accuracy(TestClassEst, Testlabels, tDocs)
print('Overall Accuracy of the BE on the Test Data =', AccuTest)
#step four: confusion matrix
print('Now calculating the confusion matrix for the test data')
ConfuseTest = Accuracy.Confusion(TestClassEst, Testlabels, tDocs, classes)
print('Confusion matrix entries through (4,4) for the the test Data')
print(ConfuseTest[1][1], ConfuseTest[1][2], ConfuseTest[1][3], ConfuseTest[1][4])
print(ConfuseTest[2][1], ConfuseTest[2][2], ConfuseTest[2][3], ConfuseTest[2][4])
print(ConfuseTest[3][1], ConfuseTest[3][2], ConfuseTest[3][3], ConfuseTest[3][4])
print(ConfuseTest[4][1], ConfuseTest[4][2], ConfuseTest[4][3], ConfuseTest[4][4])

################################################################################
#Classify the testing set: using Maximum Likelihood estimate second
print('Beginning to look at Test data using the Maximum Likelihood estimator')
#Classify the testing set: using Maximum Likelihood estimate
#step one: construct the Posterior estimates
print('Calculating Posterior estimates for test data using MLE')
tPxw2, tDocs = NBAnalysis.CalcPosteriors(tdocid, Testlabels, twordid, twordcount, classes, PMLEest, Prior, Vocab)
#step two: choose best estimate
print('Select best class estimate for prediction for testdata using MLE')
TestClassEst2=NBAnalysis.FindBestPxw(tPxw2, tDocs, classes)
#step three: calculate Accuracy
print('Calculating Accuracy for Test set with MLE')
#step three: calculate Accuracy
print('Calculating class Accuracy for Test set with BE')
ClassAccTest2 = Accuracy.ClassAcc(TestClassEst2, Testlabels, tDocs, classes)
print('Class', 'Accuracy')
for j in range(classes):
    print(j+1, ClassAccTest2[j])

AccuTest2 = Accuracy.Accuracy(TestClassEst2, Testlabels, tDocs)
print('Overall Accuracy of the MLE on the Test Data =', AccuTest2)
#step four: confusion matrix
print('Now calculating the confusion matrix for the test data using MLE')
ConfuseTest2 = Accuracy.Confusion(TestClassEst2, Testlabels, tDocs, classes)
print('Confusion matrix entries through (4,4) for the the test Data using MLE')
print(ConfuseTest2[1][1], ConfuseTest2[1][2], ConfuseTest2[1][3], ConfuseTest2[1][4])
print(ConfuseTest2[2][1], ConfuseTest2[2][2], ConfuseTest2[2][3], ConfuseTest2[2][4])
print(ConfuseTest2[3][1], ConfuseTest2[3][2], ConfuseTest2[3][3], ConfuseTest2[3][4])
print(ConfuseTest2[4][1], ConfuseTest2[4][2], ConfuseTest2[4][3], ConfuseTest2[4][4])