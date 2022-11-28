from time import time
import numpy as np
from mReadData import *
from mEvaluation import evaluate
from Base import *
from saveRst import saveMat

class AdaboostMH():
    def __init__(self, Q, T, delta=0.01):
        self.Q = Q
        self.T = T
        self.delta = delta
        self.allLearner = []    #(T,Q)
        self.Alpha_s = []   #(T,Q)
    def induce(self, X, Y):
        Dst_s = np.ones((self.Q,len(X)))   # initial distribution: Uniform distribution
        for t in range(self.T):
            baseLearner = BR()
            # baseLearner = CC()
            baseLearner.train(X, Y, Dst_s)
            alpha,Dst_s = self.hamming(Y, baseLearner.test(X), Dst_s)
            if(alpha==0):
                return
            self.Alpha_s.append(alpha)
            self.allLearner.append(baseLearner)
    def hamming(self, T, prediciton, Dst):
        tmp1 = np.int32(np.round(prediciton))
        result = tmp1==T
        error = 1 - np.sum(result*np.transpose(Dst))/(len(T)*self.Q)
        print(sum(result)/len(result))
        if(error>0.5):
            return 0,Dst
        if(error < self.delta):
            return 0,np.ones((self.Q,len(X)))
        alpha = 0.5*np.log((1-error)/error)
        Dst3 = []
        result = np.array(np.transpose(result))
        for i in range(self.Q):
            Dst2 = Dst[i]*np.exp(-(result[i]-0.5)*2*alpha)
            Dst3.append(self.distribution_adj(Dst2))
        return alpha,Dst3
    def distribution_adj(self, Dst):
        gap = min(Dst)
        if(gap<=0):
            print('dst error!!!')
            Dst = Dst - gap + 0.01
        ssum = sum(Dst)
        Dst = Dst * len(Dst)
        Dst = Dst/ssum
        return Dst
    def test(self, Xt):
        prediction = np.zeros((len(Xt),self.Q))
        print(len(self.Alpha_s))
        saveMat([len(self.Alpha_s)])
        for tt in range(len(self.Alpha_s)):
            prediction += self.allLearner[tt].test(Xt) * self.Alpha_s[tt]
        return prediction
def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y

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
        Y = fill1(Y)
        start_time = time()
        mlClassifier = AdaboostMH(np.shape(Y)[1], 10)
        mlClassifier.induce(X,Y)
        mid_time = time()
        prediction = mlClassifier.test(Xt)
        resolveResult(datasnames[dataIdx], 'Adaboost_MH', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
