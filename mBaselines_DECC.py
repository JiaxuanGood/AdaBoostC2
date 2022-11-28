from sklearn.neighbors import NearestNeighbors
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances
from skmultilearn.problem_transform import BinaryRelevance
from operator import itemgetter
from mReadData import *
from mDECC_kNN import CCkNN
from mEvaluation import evaluate
from time import time

def br_predict(X,Y,Xt):
    br = BinaryRelevance(
        classifier=SVC(probability=True,C=1.0, kernel='rbf',gamma='scale'),
        require_dense=[False, True]
    )
    br.fit(X,Y)
    prediction = br.predict(Xt).todense()
    return np.array(prediction)

def fuzzyDist(X, Xt_a, beta=5):
    fDists = pairwise_distances([Xt_a], X)
    fDists = np.array(fDists)
    return np.exp(-beta*(fDists*fDists))

def calF1(V_and_D_memdeg,V_rem_D_memdeg,D_rem_V_memdeg,N_memdeg):
    tp_memdeg,fn_memdeg,fp_memdeg = [],[],[]
    for i in range(len(N_memdeg)):
        tp_memdeg.append(min(V_and_D_memdeg[i],N_memdeg[i]))
        fn_memdeg.append(min(V_rem_D_memdeg[i],N_memdeg[i]))
        fp_memdeg.append(min(D_rem_V_memdeg[i],N_memdeg[i]))
    tp = np.sum(tp_memdeg)  # cardinality of a fuzzy set
    fn = np.sum(fn_memdeg)
    fp = np.sum(fp_memdeg)
    if(tp+fp+fn==0):
        return 0.0
    else:
        return 2*tp/(2*tp+fp+fn)

def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y

datasnames = ["3Sources_bbc1000","3Sources_guardian1000","3Sources_inter3000","3Sources_reuters1000","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck",
    "GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess","Coffee","VirusGO","Yeast","Yelp"]
rd = ReadData(datas=datasnames,genpath='arff/')

K_num_baseLearners = 10 #defination
k_CCkNN = 10 #{1,3,5,7,9,11}
beta_order = 5 #{1,2,3,4,5,6,7,8,9,10}
k_labelorder_num_neibors = 10
for dataIdx in range(4,5):
    print(dataIdx)
    # X_org,Y_org,Xt,Yt= rd.readData(data_index)  # data->(train,test)
    k_fold,X_all,Y_all = rd.readData_CV(dataIdx)
    for train, test in k_fold.split(X_all, Y_all):
        X_org = X_all[train]
        Y_org = Y_all[train]
        Xt = X_all[test]
        Yt = Y_all[test]
        Y_org = fill1(Y_org)
        cap_test,num_label = np.shape(Yt)
        predictions = np.zeros((cap_test,num_label))

        start_time = time()
        train_time = 0
        test_time = 0
        for baseLearner in range(K_num_baseLearners):
            train_time1 = time()
            X,Xv,Y,Yv = model_selection.train_test_split(X_org,Y_org, train_size=0.66)  # train->(train1,valid)
            # V_org = np.hstack((Xv,Yv))
            YvBR = br_predict(X,fill1(Y),Xv)
            
            num_neibors = k_labelorder_num_neibors
            findNb = NearestNeighbors(n_neighbors=num_neibors, algorithm='ball_tree')
            findNb.fit(Xv)
            train_time2 = time()
            train_time += (train_time2-train_time1)

            '''testing phase'''
            '''dynamic generate'''
            test_time1 = time()
            distances,indices = findNb.kneighbors(Xt)
            all_labels_order = []
            for i in range(cap_test):
                # # Vq,Dq,Nq have same members but different member_degrees, so only member_degrees calculation is enough
                Xv_neibor_this = Xv[indices[i]]
                Yv_neibor_this = Yv[indices[i]]
                # N_V_D_mem = np.hstack((Xv_neibor_this, Yv_neibor_this))
                YvBR_neibor_this = YvBR[indices[i]]
                N_memdeg = fuzzyDist(Xv_neibor_this, Xt[i], beta_order)[0]
                F1 = []
                for q in range(num_label):
                    V_and_D_memdeg = np.zeros(num_neibors)
                    V_rem_D_memdeg = np.zeros(num_neibors)
                    D_rem_V_memdeg = np.zeros(num_neibors)
                    
                    for ii in range(num_neibors):
                        if(Yv_neibor_this[ii][q]==1 and YvBR_neibor_this[ii][q]==1):
                            V_and_D_memdeg[ii] = 1
                        else:
                            if(Yv_neibor_this[ii][q]==1):
                                V_rem_D_memdeg[ii] = 1
                            if(YvBR_neibor_this[ii][q]==1):
                                D_rem_V_memdeg[ii] = 1
                    F1.append(calF1(V_and_D_memdeg,V_rem_D_memdeg,D_rem_V_memdeg,N_memdeg))
                F1_indices, F1_sorted = zip(*sorted(enumerate(-np.array(F1)), key=itemgetter(1)))
                all_labels_order.append(F1_indices)
            all_labels_order = np.array(all_labels_order)
            tmp = np.zeros((cap_test,1))-1
            all_labels_order = np.hstack((tmp,all_labels_order))

            predictions_this = CCkNN(X_org,Y_org,Xt, all_labels_order.astype(int))
            predictions += np.array(predictions_this)
            test_time2 = time()
            test_time += (test_time2-test_time1)
        predictions /= K_num_baseLearners
        resolveResult(datasnames[dataIdx], 'DECC', evaluate(predictions, Yt), train_time, test_time, (time()-start_time) )
