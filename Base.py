from cmath import log
from time import time
import numpy as np
from sklearn import svm
from mReadData import *
from mEvaluation import evaluate
from skmultilearn.problem_transform.cc import ClassifierChain
import random

def randorder(Q):
    return np.array(random.sample(range(Q),Q))

class BR():
    def __init__(self):
        self.baseLearner = []
        self.Q = 0
    def train(self,X,Y,distribution):
        self.Q = np.shape(Y)[1]
        for j in range(self.Q):
            singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            singleLearner.fit(X,Y[:,j],distribution[j])
            self.baseLearner.append(singleLearner)
    def test(self,Xt):
        prediction = []
        for j in range(self.Q):
            prediction_a = self.baseLearner[j].predict_proba(Xt)[:,1]
            prediction.append(prediction_a)
        return np.array(np.transpose(prediction))
class CC():
    def __init__(self,order=[]):
        self.baseLearner = []
        self.num_label = 0
        self.order = order
    def train(self,X,Y,distribution=[]):
        X_train = np.array(X)
        self.num_label = np.shape(Y)[1]
        if(len(self.order)==0):
            self.order = randorder(self.num_label)
        for j in self.order:
            singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            if(len(distribution)==0):
                singleLearner.fit(X_train,Y[:,j])
            else:
                singleLearner.fit(X_train,Y[:,j],distribution[j])
            self.baseLearner.append(singleLearner)
            X_train = np.hstack((X_train, Y[:,[j]]))
    def test(self,Xt):
        Xt_train = np.array(Xt)
        prediction= [[] for _ in range(self.num_label)]
        for i in range(len(self.order)):
            j = self.order[i]
            prediction_a = self.baseLearner[i].predict_proba(Xt_train)[:,1]
            prediction[j] = prediction_a
            prediction_a = np.reshape(prediction_a, (-1, 1))
            Xt_train = np.hstack((Xt_train, prediction_a))
        return np.transpose(prediction)
    def test_PCC(self,Xt,Q):
        Xt_test = np.array(Xt)
        num_test = len(Xt)
        prediction = np.zeros((num_test,Q))
        for i in range(num_test):
            candidates = pow(2,Q)
            probs = np.zeros(candidates)
            for j in range(candidates):
                this_test = Xt_test[i]
                candidate = self.candidate_decode(j,Q)
                tmp_p = 1
                for rr in range(Q):
                    r = candidate[rr]
                    predict_this = self.baseLearner[rr].predict_proba([this_test])[:,1]
                    if(r==0):
                        predict_this = 1 - predict_this
                    this_test = np.append(this_test,r)
                    tmp_p *= predict_this
                probs[j] = tmp_p
            select = np.argmax(probs)
            prediction[i] = self.candidate_decode(select,Q)
        return prediction
    def candidate_decode(self,idx,Q):
        tmp = idx
        candidate = np.zeros(Q)
        for i in range(Q):
            candidate[i] = tmp % 2
            tmp = int(tmp/2)
        return candidate[::-1]

class adaboost():
    def __init__(self, T=10):
        self.T = T
    def induce(self, X, Y_a, Xt):
        Dst = np.ones(len(X))
        Alphas = []
        Learners = []
        for t in range(self.T):
            learner = self.train(X, Y_a, Dst)
            Learners.append(learner)
            alpha,Dst = self.boosting(X, Y_a, Dst, learner)
            if(alpha==0):
                break
            Alphas.append(alpha)
        if(len(Alphas)==0):
            Alphas.append(1)
        Alphas = np.array(Alphas)/len(Alphas)
        print(Alphas)
        prediction = np.zeros(len(Xt))
        thisT = len(Alphas)
        for t in range(thisT):
            prediction += np.array(Learners[t].predict_proba(Xt))[:,1] * Alphas[t]
        return np.array(prediction/thisT)
    def train(self, X, Y_a, Dst):
        singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
        gap = min(Dst)
        if(gap<=0):
            Dst = Dst - gap + 0.01
        singleLearner.fit(X, Y_a, sample_weight=Dst)
        return singleLearner
    def boosting(self, X, Y_a, Dst, learner):
        result = np.array(learner.predict(X)!=Y_a)
        error = sum(result*Dst)/len(X)
        print(error, end='\t')
        alpha = 0.5*np.log((1-error)/error)
        if(error<0.0001):
            return 0,np.zeros(len(X))
        Dst2 = Dst*np.exp((result-0.5)*2*alpha)
        Dst2 = Dst2/sum(Dst2)
        return alpha,Dst2

if __name__=="__main__":
    '''Ablation Study'''
    def fill1(Y):
        Y = np.array(Y)
        for j in range(np.shape(Y)[1]):
            if(np.sum(Y[:,j])==0):
                Y[0][j] = 1
        return Y

    numBase = 10
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
            prediction = []
            for q in range(np.shape(Y)[1]):
                boostLearner = adaboost(T=numBase)
                prd = boostLearner.induce(X, Y[:,q], Xt)
                prediction.append(prd)
            prediction = np.transpose(prediction)
            resolveResult(datasnames[dataIdx], 'AdaBoost.BR.Mul', evaluate(prediction, Yt), (time()-start_time))
