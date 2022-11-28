from sklearn.neighbors import NearestNeighbors
from mReadData import *
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from mEvaluation import evaluate

def generate_many(X,Y,Xt,Yt, newLabel):
    if(newLabel==-1):
        return X,Xt
    X = np.hstack((X, Y[:,[newLabel]]))
    Xt = np.hstack((Xt, Yt[:,[newLabel]]))
    return X,Xt

def generate(X,Y,Xt_a,Yt_a, newLabel):
    if(newLabel==-1):
        return X,Xt_a
    X = np.hstack((X, Y[:,[newLabel]]))
    Xt_a = np.append(Xt_a, Yt_a[newLabel])
    return X,Xt_a

def generate_a(X,Y,Xt_a,Yt_a_aLabel, newLabel):
    '''Xt_a:(num_feature,), Yt_a_aLabel:(1,)'''
    '''newLabel: the index of label to be predicted'''
    if(newLabel==-1):
        return X,Xt_a
    X = np.hstack((X, Y[:,[newLabel]]))
    Xt_a = np.append(Xt_a, Yt_a_aLabel)
    return X,Xt_a

def predict(X,Y,Xt, testLabel, k=5):
    findNb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    findNb.fit(X)
    indices = findNb.kneighbors(Xt, return_distance=False)
    prediction = np.sum(Y[indices,testLabel])/k
    return prediction

def CCkNN_rep(X_org,Y,Xt, all_labels_order, k_CCkNN=5):
    cap_test = np.shape(Xt)[0]
    num_label = np.shape(Y)[1]
    results = np.zeros((cap_test,num_label))
    for i in range(cap_test):
        Xt_a = Xt[i]
        X = X_org
        labels = all_labels_order[i]
        predict_tmp_for_next_label = 0
        for q in range(num_label):
            X,Xt_a = generate_a(X,Y,Xt_a,predict_tmp_for_next_label, labels[q])
            results[i][labels[q+1]] = predict(X,Y,[Xt_a], labels[q+1], k_CCkNN)
            predict_tmp_for_next_label = results[i][labels[q+1]]
    return results

def update_distances(distances,Y,Y_this_label, this_label):
    if(this_label == -1):
        return distances
    tmp = Y[:,this_label]-Y_this_label
    distance_add = [num*num for num in tmp]
    distances += np.array(distance_add)
    return distances

def CCkNN(X_org,Y,Xt, all_labels_order, k_CCkNN=5):
    cap_test = np.shape(Xt)[0]
    num_label = np.shape(Y)[1]
    results = np.zeros((cap_test,num_label))

    distanceses_eu = pairwise_distances(Xt,X_org)    # (num_train,num_feature),(1,num_feature) -> (num_train,1)
    distanceses = [[distanceses_eu[i][j]**2 for j in range(len(distanceses_eu[i]))] for i in range(len(distanceses_eu))]
    distanceses = np.array(distanceses)
    for i in range(cap_test):
        Xt_a = Xt[i]
        X = X_org
        distances = distanceses[i].flatten()
        labels = all_labels_order[i]
        predict_tmp_for_this_label = 0
        for q in range(num_label):
            distances = update_distances(distances,Y,predict_tmp_for_this_label, labels[q])
            indexs = np.argsort(distances)[:k_CCkNN]
            prediction = np.sum(Y[indexs,labels[q+1]])/k_CCkNN
            results[i][labels[q+1]] = prediction
            predict_tmp_for_this_label = prediction
    return results

if __name__ == '__main__':
    X_org,Y,Xt,Yt = ReadData().readData(2)  # data->(train,test)
    labels = [-1,1,2,3,4,5,0]
    cap_test,num_label = np.shape(Yt)
    results = np.zeros((cap_test,num_label))

    labels = np.array(labels*(np.shape(Xt)[0]))
    labels = np.reshape(labels,(cap_test,num_label+1))
    results = CCkNN(X_org,Y,Xt,labels)
    print(evaluate(results,Yt))

    results = CCkNN_rep(X_org,Y,Xt,labels)
    print(evaluate(results,Yt))

