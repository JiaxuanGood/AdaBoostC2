from time import time
import numpy as np
from mReadData import *
from mEvaluation import evaluate
from Base import *
import random

datasnames = ["3Sources_bbc1000","3Sources_guardian1000","3Sources_inter3000","3Sources_reuters1000","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck",
    "GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess",
    "Coffee","VirusGO","Yeast","Yelp","Corel5k","Philosophy"]
rd = ReadData(datas=datasnames,genpath='arff/')

def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y

def randorder(Q):
    return np.array(random.sample(range(Q),Q))
    
numBase = 1
for dataIdx in range(1):
    print(dataIdx)
    X,Y,Xt,Yt = rd.readData(dataIdx)
    # k_fold,X_all,Y_all = rd.readData_CV(dataIdx)
    # for train, test in k_fold.split(X_all, Y_all):
    #     X = X_all[train]
    #     Y = Y_all[train]
    #     Xt = X_all[test]
    #     Yt = Y_all[test]
    Y = fill1(Y)
    start_time = time()
    ensembleLearner = []
    for t in range(numBase):
        mlClassifier = CC(randorder(np.shape(Y)[1]))
        mlClassifier.train(X,Y)
        ensembleLearner.append(mlClassifier)
    mid_time = time()
    prediction = np.zeros(np.shape(Yt))
    for t in range(numBase):
        prediction += ensembleLearner[t].test(Xt)
    prediction /= numBase
    resolveResult(datasnames[dataIdx], 'ECC', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
