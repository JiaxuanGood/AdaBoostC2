from skmultilearn.ext import Meka, download_meka
from mReadData import *
from time import time
from mEvaluation import *

def RAkEL(X_train,y_train,X_test,y_test):
    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.RAkEL",
        weka_classifier="weka.classifiers.trees.J48",
        meka_classpath=download_meka(),
        java_command='C:\\Program Files\\Java\\jdk1.8.0_181\\bin\\java')
    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)
    return (predictions)

def AdaBoost_MH(X_train,y_train,X_test,y_test):
    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.RAkEL",
        weka_classifier="weka.classifiers.trees.J48",
        meka_classpath=download_meka(),
        java_command='C:\\Program Files\\Java\\jdk1.8.0_181\\bin\\java')
    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)
    return (predictions)

datasnames = ["Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","Genbase","GnegativeGO","GpositiveGO","Image","Langlog","Medical",
"PlantGO","Scene","Slashdot","Chemistry","Chess","Coffee","VirusGO","Yeast","Yelp"]
rd = ReadData(genpath='arff/')

def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y

for dataIdx in range(6):
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
        prediction = RAkEL(X,Y,Xt,Yt)
        prediction = np.array(prediction.todense())
        print(type(prediction))
        print(np.shape(prediction),np.shape(Yt))
        # prediction = np.transpose(prediction)
        mid_time = time()
        resolveResult(datasnames[dataIdx], 'RAkEL', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))