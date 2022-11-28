from os import close
from numpy.__config__ import show
from numpy.core.fromnumeric import shape
from sklearn import svm
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.svm import SVR
import sklearn.metrics as metrics
from lasso import *
from read_arff import *
import sys
sys.path.append("util")
import scipy
import time
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.adapt import MLkNN
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.ext import Meka, download_meka
from skmultilearn.model_selection import IterativeStratification
import mEvaluation
from mReadData import *

def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y

# BR classifier
def BR(X_train,y_train,X_test,new_X_test):
    classifier = BinaryRelevance(
        classifier=SVC(probability=True,C=1.0, kernel='rbf',gamma='scale'),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)    #X_train.dim1==y_train.dim1, X_train.dim2==numFeat, y_train.dim2==numLabel
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    t1 = time.time()
    new_pro_predictions = classifier.predict_proba(new_X_test)
    time_test = time.time()-t1
    return(predictions,pro_predictions,new_pro_predictions,time_test)

# CC classifier
def CC(X_train,y_train,X_test,new_X_test):
    classifier = ClassifierChain(
        classifier=SVC(probability=True,C=1.0, kernel='rbf',gamma='scale'),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    t1 = time.time()
    new_pro_predictions = classifier.predict_proba(new_X_test)
    time_test = time.time()-t1
    return(predictions,pro_predictions,new_pro_predictions,time_test)

# LP classifier
def LP(X_train,y_train,X_test,new_X_test):
    classifier = ClassifierChain(
        classifier=RandomForestClassifier(n_estimators=20),
        require_dense=[False, True]
    )
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    pro_predictions = classifier.predict_proba(X_test)
    t1 = time.time()
    new_pro_predictions = classifier.predict_proba(new_X_test)
    time_test = time.time()-t1
    return(predictions,pro_predictions,new_pro_predictions,time_test)

# MLS classifier
def MLS(X_train,y_train,X_test,y_test):
    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.BR",
        weka_classifier="weka.classifiers.meta.Stacking",
        meka_classpath=download_meka(),
        java_command='C:\\Program Files\\Java\\jdk1.8.0_201\\bin\\java')
    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)
    return (predictions)

# Features combine the output of each base classifier
def combine_feature(pro_br,pro_cc,pro_lp):
    br = pd.DataFrame(pro_br.todense())
    cc = pd.DataFrame(pro_cc.todense())
    lp = pd.DataFrame(pro_lp.todense())
    # mlknn=pd.DataFrame(pro_mlknn.todense())
    # x_se = pd.DataFrame(X_se)
    X_new = pd.concat([br, cc, lp], axis=1)
    return(X_new)

if __name__ == '__main__':
    datasnames = ["3Sources_bbc1000","3Sources_guardian1000","3Sources_inter3000","3Sources_reuters1000","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck",
        "GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess",
        "Coffee","VirusGO","Yeast","Yelp","Corel5k","Philosophy"]
    rd = ReadData(datas=datasnames,genpath='arff/')
    path = 'arff/'
    label_counts = rd.getnum_label()

    for dataIdx in range(4,5):
        print(dataIdx)
        label_count = label_counts[dataIdx]
        X,Y,f = read_arff(path+datasnames[dataIdx], label_count)
        k_fold = IterativeStratification(n_splits=10, order=1)
        for train, test in k_fold.split(X,Y):
            temp_mean=list()
            temp_std=list()
            output = list()
            j_fold = IterativeStratification(n_splits=2, order=1)
            for new_train, new_test in j_fold.split(X[train],Y[train]):
                start_time = time.time()

                mat_mid = np.array(Y[train][new_train].todense())
                sum_list = np.sum(mat_mid, axis=0)
                for r in range(label_count):
                    if(sum_list[r]==0):
                        mat_mid[0][r] = 1
                        isOrgDataModify = True
                this_Y = np._mat.mat(mat_mid)

                predictions_BR, pro_predictions_BR, new_pro_predictions_BR, t_br = BR(X[train][new_train], this_Y,
                                                        X[train][new_test], X[test])
                predictions_CC, pro_predictions_CC, new_pro_predictions_CC, t_cc = CC(X[train][new_train], this_Y,
                                                        X[train][new_test], X[test])
                predictions_LP, pro_predictions_LP, new_pro_predictions_LP, t_lp = LP(X[train][new_train], this_Y,
                                                        X[train][new_test], X[test])
                stacking = combine_feature(pro_predictions_BR, pro_predictions_CC, pro_predictions_LP)  # 3 matrix -> 1 matrix
                # training w
                w = lasso(stacking.values, Y[train][new_test].todense(),
                        math.pow(10, -4),math.pow(10, -3), 0.1, 200,0.0001)

                time_2 = time.time()
                new_stacking=combine_feature(new_pro_predictions_BR, new_pro_predictions_CC, new_pro_predictions_LP)
                pre_score = np.dot(new_stacking.values, w)
                testtime = time.time() - time_2
                testtime += (t_br+t_cc+t_lp)

                model_time=time.time() - start_time
                result = mEvaluation.evaluate(np.array(pre_score), np.array(Y[test].todense()))
                result.append((model_time-testtime)*2)
                result.append(testtime*2)
                output.append(result)

            data = pd.DataFrame(output)
            temp_mean.append(data.mean())
            temp_std.append(data.std())
            del output
            result_mean = pd.DataFrame(temp_mean)
            result_std = pd.DataFrame(temp_std)
            # output result
            result=pd.DataFrame({'Evaluate':['Accuracy','Precision','Recall','F1-Score','hitrate','subsetAcc','hamming_loss','oneerror','coverage','ranking_loss','average_precision','time1','time2'],
                                'Mean':result_mean.mean().values,'Std':result_std.mean().values})
            rst = np.zeros(13)
            for jk in range(13):
                rst[jk] = result.at[jk,'Mean']
            resolveResult(datasnames[dataIdx], 'MLWSE', rst)
