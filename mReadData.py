import numpy as np
from skmultilearn.dataset import load_from_arff
from skmultilearn.model_selection import IterativeStratification

def read_arff(path, label_count, wantfeature=False):
    path_to_arff_file=path+".arff"
    arff_file_is_sparse = False
    X, Y, feature_names, label_names = load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location="end",
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )
    if(~wantfeature):
        return X, Y, None
    else:
        featype = []
        for i in range(len(feature_names)):
            if(feature_names[i][1] == 'NUMERIC'):
                featype.append([0])
            else:
                if(not feature_names[i][1][0].isdigit()):
                    feature_nomimal = np.arange(0,len(feature_names[i][1]))
                    featype.append([int(number) for number in feature_nomimal])
                else:
                    featype.append([int(number) for number in feature_names[i][1]])
        return X, Y, featype

class ReadData:
    def __init__(self, datas=[], genpath=""):
        self.genpath = genpath
        '''ALL datasets from KDIS (http://www.uco.es/kdis/mllresources/)'''
        datasnames_ = ["3Sources_bbc1000","3Sources_guardian1000","3Sources_inter3000","3Sources_reuters1000","Birds","CAL500","CHD_49","Corel5k","Enron","Flags",
            "Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess","Coffee","Philosophy","VirusGO","Yeast","Yelp"]
        dimALL_ = [352,302,169,294,645,502,555,5000,1702,194,407,1392,519,2000,1460,978,978,2407,3782,6961,1675,225,3971,207,2417,10810]
        num_labels_ = [6,6,6,6,19,174,6,374,53,7,12,8,4,5,75,45,12,6,22,175,227,123,233,6,14,5]
        dimTrains_ = [240,204,112,201,432,327,372,3332,1151,132,275,931,347,1501,978,659,644,1618,2546,4686,1107,153,2623,131,1629,7240]
        dimTests_ = [112,98,57,93,213,175,183,1668,551,62,132,461,172,499,482,319,334,789,1236,2275,568,72,1348,76,788,3566]
        num_attr_ = [1000,1000,3000,1000,260,68,49,499,1001,19,21,1717,912,294,1004,1449,3091,294,1079,540,585,1763,842,749,103,671]
        self.datasnames, self.dimALL, self.dimTrains, self.dimTests, self.num_feats, self.num_labels = [],[],[],[],[],[]
        allIndices = []
        for i in range(len(datas)):
            allIndices.append(datasnames_.index(datas[i]))
        for i in range(len(allIndices)):
            self.datasnames.append(datasnames_[allIndices[i]])
            self.dimALL.append(dimALL_[allIndices[i]])
            self.dimTrains.append(dimTrains_[allIndices[i]])
            self.dimTests.append(dimTests_[allIndices[i]])
            self.num_feats.append(num_attr_[allIndices[i]])
            self.num_labels.append(num_labels_[allIndices[i]])

    def readData_org(self, index):
        label_count = self.num_labels[index]
        path = self.genpath + self.datasnames[index]
        X, Y, featype = read_arff(path, label_count, False)
        dimTrain = self.dimTrains[index]
        dimTest = self.dimTests[index]
        print(self.datasnames[index],np.shape(X),np.shape(Y),dimTrain,dimTest)
        train_idx = np.arange(dimTrain)
        test_idx = np.arange(dimTrain,dimTrain+dimTest)
        return X[train_idx],Y[train_idx],X[test_idx],Y[test_idx],featype

    def readData(self, index):
        X,Y,Xt,Yt,f = self.readData_org(index)
        X,Y,Xt,Yt = np.array(X.todense()), np.array(Y.todense()), np.array(Xt.todense()), np.array(Yt.todense())
        return X,Y,Xt,Yt

    def readData_CV(self, index, CV=10):
        label_count = self.num_labels[index]
        # print(self.datasnames[index],self.dimALL[index],self.num_labels[index])
        path = self.genpath + self.datasnames[index]
        X, Y, f = read_arff(path, label_count)
        k_fold = IterativeStratification(n_splits=CV, order=1)
        # for train, test in k_fold.split(X, Y):
        #     print(np.shape(train),np.shape(test))
        return k_fold, np.array(X.todense()), np.array(Y.todense())
    
    def getnum_label(self):
        return self.num_labels

def resolveResult(dataName='', algName='', result=[], time1=0, time2=0, time3=0):
    f = open('result.txt', 'a')
    print(dataName, end='\t', file=f)
    print(algName, end='\t', file=f)
    for i in range(np.shape(result)[0]):
        print(result[i], end='\t', file=f)
    if(time1>0):
        print(time1, end='\t', file=f)
    if(time2>0):
        print(time2, end='\t', file=f)
    if(time3>0):
        print(time3, end='\t', file=f)
    print('', file=f)
    f.close()
