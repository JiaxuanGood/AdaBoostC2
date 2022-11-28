from time import time
import numpy as np
from sklearn import svm
from mReadData import *
from mEvaluation import evaluate
from saveRst import saveMat
from operator import itemgetter
import random

class AdaboostC3():
    def __init__(self, Q, T, delta=0.01):
        self.Q = Q
        self.T = T
        self.delta = delta
        self.allLearner = []    #(T,Q)
        self.Alpha_s = []   #(T,Q)
        self.Order_s = []   #(T,Q)
    def induce(self, X, Y):
        Dst_s = np.ones((self.Q, len(X)))   # initial distribution: Uniform distribution
        order = randorder(self.Q)   # initial order of classifiers chain: random
        ok = [] # the indexes of exactly classificated labels
        for t in range(self.T):
            Dst_s,error_s = self.trainCC(X, Y, Dst_s, order, ok)
            ok = np.argwhere(np.array(error_s)<self.delta).flatten()
            indices, L_sorted = zip(*sorted(enumerate(np.array(error_s)), key=itemgetter(1)))
            order = np.array(indices)
            order2 = randorder2(len(order),ok)
            order[len(ok):] = order2
    def trainCC(self, X, Y, Dst_s, order, ok):
        self.Order_s.append(order)
        order = order[len(ok):]
        X_train = np.array(X)
        if(len(ok)>0):
            for q in ok:
                X_train = np.hstack((X_train, Y[:,[q]]))
        Alpha = ['']*self.Q
        baseLearner = ['']*self.Q
        Dst_s2 = ['']*self.Q
        error_s = np.zeros(self.Q)
        for qq in order:
            singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            # print(np.sum(Dst_s[qq]), len(Dst_s[qq]))
            singleLearner.fit(X_train, Y[:,qq], Dst_s[qq])
            baseLearner[qq] = singleLearner
            alpha,Dst2,error = self.boosting_a(X_train, Y[:,qq], Dst_s[qq], singleLearner)
            Alpha[qq] = alpha
            Dst_s2[qq] = Dst2
            error_s[qq] = error
            X_train = np.hstack((X_train, Y[:,[qq]]))
        self.allLearner.append(baseLearner)
        self.Alpha_s.append(Alpha)
        return Dst_s2, error_s
    def boosting_a(self, X, Y_a, Dst, learner):
        tmp1 = np.int32(np.round(learner.predict(X)))
        result = np.array(tmp1!=Y_a)
        error = sum(result*Dst)/len(X)
        if(error>0.5):
            return 0,Dst,error
        if(error < self.delta):
            return 0,np.ones(len(X)),error #np.ones(len(X))
        alpha = 0.5*np.log((1-error)/error)
        Dst2 = Dst*np.exp(-(result-0.5)*2*alpha)
        Dst3 = self.distribution_adj(Dst2)
        return alpha,Dst3,error
    def distribution_adj(self, Dst):
        gap = min(Dst)
        if(gap<=0):
            print('dst error!!!')
            Dst = Dst - gap + 0.01
        ssum = sum(Dst)
        Dst = Dst * len(Dst)
        Dst = Dst/ssum
        return Dst
    def test(self, Xt, mode=1):
        if(mode==1):
            Alpha_weights = self.get_alphaweights()
        elif(mode==2):
            Alpha_weights = self.get_alphaweights2()
        elif(mode==3):
            Alpha_weights = self.get_alphaweights3()
            saveMat(1/Alpha_weights[0])
        else:
            print('wrong')
        prediction = np.zeros((self.Q,len(Xt)))
        prediction_aLabel = ['']*self.Q
        # saveMat(Alpha_weights)
        for tt in range(self.T):
            # print('base round:', tt)
            Xt_train = np.array(Xt)
            prediction_t = np.zeros((self.Q,len(Xt)))
            for qq in self.Order_s[tt]:
                if(Alpha_weights[tt][qq]==0):
                    Xt_train = np.hstack((Xt_train, np.reshape(prediction_aLabel[qq], (-1, 1))))
                    continue
                # print(Alpha_weights[tt][qq], np.shape(Xt_train))
                # print(tt,qq)
                prediction_a = self.allLearner[tt][qq].predict_proba(Xt_train)[:,1]
                prediction_aLabel[qq] = prediction_a
                prediction_t[qq] = np.array(prediction_a) * Alpha_weights[tt][qq]
                if(Alpha_weights[tt][qq]<0):
                    prediction_t[qq] = -prediction_t[qq]
                Xt_train = np.hstack((Xt_train, np.reshape(prediction_a, (-1, 1))))
            prediction = prediction + np.array(prediction_t)
        return np.transpose(prediction)
    def get_alphaweights(self): #adjust weight
        Alpha_weights = np.zeros((self.T, self.Q))
        for i in range(self.T):
            for j in range(self.Q):
                if(self.Alpha_s[i][j]!=''):
                    Alpha_weights[i][j] = self.Alpha_s[i][j]
        for j in range(self.Q):
            if(Alpha_weights[0][j] == 0):
                Alpha_weights[0][j] = 1
        Alpha_weights = np.transpose(Alpha_weights)
        for i in range(self.Q):
            Alpha_weights[i] = Alpha_weights[i]/sum(np.abs(Alpha_weights[i]))
        Alpha_weights = np.array(np.transpose(Alpha_weights))
        return Alpha_weights
    def get_alphaweights2(self): #original weight
        Alpha_weights = np.zeros((self.T, self.Q))
        for i in range(self.T):
            for j in range(self.Q):
                if(self.Alpha_s[i][j]!=''):
                    Alpha_weights[i][j] = self.Alpha_s[i][j]
        for j in range(self.Q):
            if(Alpha_weights[0][j] == 0):
                Alpha_weights[0][j] = 1
        Alpha_weights = np.transpose(Alpha_weights)
        Alpha_weights = np.array(np.transpose(Alpha_weights))
        return Alpha_weights
    def get_alphaweights3(self):    #equal weight
        Alpha_weights = np.zeros((self.T, self.Q))
        for i in range(self.T):
            for j in range(self.Q):
                if(self.Alpha_s[i][j]!=''):
                    Alpha_weights[i][j] = 1
        for j in range(self.Q):
            if(Alpha_weights[0][j] == 0):
                Alpha_weights[0][j] = 1
        Alpha_weights = np.transpose(Alpha_weights)
        for i in range(self.Q):
            Alpha_weights[i] = Alpha_weights[i]/sum(np.abs(Alpha_weights[i]))
        Alpha_weights = np.array(np.transpose(Alpha_weights))
        return Alpha_weights

def randorder(Q):
    return np.array(random.sample(range(Q),Q))
    # return np.arange(Q)
def randorder2(Q, ok):
    lst = randorder(Q)
    for i in range(len(ok)):
        lst[np.argwhere(lst==ok[i]).flatten()] = -1
    lst2 = []
    for i in range(len(lst)):
        if(lst[i]!=-1):
            lst2.append(lst[i])
    return lst2
def logmat(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            print(type(mat[i][j]), end=' ')
        print()
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
        mlClassifier = AdaboostC3(np.shape(Y)[1], T=10, delta=0.01)
        mlClassifier.induce(X,Y)
        mid_time = time()
        prediction = mlClassifier.test(Xt)
        resolveResult(datasnames[dataIdx], 'AdaCC1', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        prediction = mlClassifier.test(Xt,2)
        resolveResult(datasnames[dataIdx], 'AdaCC2', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        prediction = mlClassifier.test(Xt,3)
        resolveResult(datasnames[dataIdx], 'AdaCC3', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
        