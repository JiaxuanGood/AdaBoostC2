from skmultilearn.dataset import load_from_arff
import numpy

def read_arff(path,label_count):
    path_to_arff_file=path+".arff"
    arff_file_is_sparse = False
    X, Y, feature_names, label_names = load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location="end",
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )
    n_samples, n_features = X.shape
    n_samples, label=Y.shape
    
    featype = []
    for i in range(len(feature_names)):
        if(feature_names[i][1] == 'NUMERIC'):
            featype.append([0])
        else:
            if(not feature_names[i][1][0].isdigit()):
                feature_nomimal = numpy.arange(0,len(feature_names[i][1]))
                featype.append([int(number) for number in feature_nomimal])
            else:
                featype.append([int(number) for number in feature_names[i][1]])
    print("n_samples："+str(n_samples)+"  n_features："+str(n_features)+"  label_count："+str(label))
    return X, Y, featype

if __name__ == '__main__':
    datasets = ["3sources_bbc1000","3sources_guardian1000","3sources_inter3000","3sources_reuters1000","Birds","CHD_49","Emotions","Flags","Foodtruck",
    "GnegativePseAAC","GpositiveGO","GpositivePseAAC","Image","PlantPseAAC","Scene","VirusGO","VirusPseAAC","Water_quality","Yeast",
    "EukaryotePseAAC","Genbase","GnegativeGO","HumanPseAAC","Medical","PlantGO","Yelp",
    "HumanGO","CAL500", "Coffee","tmc2007_500"]
    label_counts = [6,6,6,6,19,6,6,7,12,8,4,4,5,12,6,6,6,14,14,22,27,8,14,
        45,12,5,14,174,123,22]
    for index in range(7,9):
        # index = 4
        print(datasets[index])
        path = "data30/" + datasets[index]
        label_count = label_counts[index]
        X, Y, f= read_arff(path, label_count)
        print(f)
        # print(type(X))
        # print(type(Y))
        # print(pd.DataFrame(Y.todense()))
