from time import time
from mlrl.boosting import Boomer
import numpy as np
from mReadData import *
from mEvaluation import evaluate

datasnames = ["3Sources_bbc1000","3Sources_guardian1000","3Sources_inter3000","3Sources_reuters1000","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck",
    "GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess",
    "Coffee","VirusGO","Yeast","Yelp","Corel5k","Philosophy"]
rd = ReadData(datas=datasnames,genpath='arff/')

for dataIdx in range(4,5):
    print(dataIdx)
    # X,Y,Xt,Yt = rd.readData(dataIdx)
    k_fold,X_all,Y_all = rd.readData_CV(dataIdx)
    for train, test in k_fold.split(X_all, Y_all):
        X = X_all[train]
        Y = Y_all[train]
        Xt = X_all[test]
        Yt = Y_all[test]
        start_time = time()
        mlClassifier = Boomer(loss='logistic-example-wise',
            parallel_rule_refinement='false', parallel_statistic_update='false', parallel_prediction='false')
        mlClassifier.fit(X,Y)
        mid_time = time()
        prediction = mlClassifier.predict(Xt)
        resolveResult(datasnames[dataIdx], 'BOOMER', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
