import numpy as np

def evaluate(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    if(np.shape(predict) != np.shape(target)):
        return
    dimData,numLabel = np.shape(predict)

    seq = getSeq(predict)
    rank = getRank(seq)

    oneerror,coverage,rankingloss,avg_precision = 0,0,0,0
    for i in range(dimData):
        dim_yi = np.sum(np.rint(predict[i]))
        dim_ti = np.sum(np.rint(target[i]))
        
        if(round(target[i][int(seq[i][0])]) != 1):
            oneerror += 1	#error, when the most confident one is incorrectly estimated
        cnt_cov = dim_ti
        r=0
        while(r<numLabel and cnt_cov!=0):
            if( target[i][int(seq[i][r])]==1 ):
                cnt_cov -= 1
            r += 1
        coverage += r
        
        # test_rankingloss = rankingloss
        if(dim_ti!=0 and dim_ti!=numLabel):
            cnt_rank = 0
            for j in range(numLabel):
                if(target[i][j]==0):
                    continue
                for k in range(numLabel):
                    if(target[i][k]==1):
                        continue
                    if(rank[i][j] > rank[i][k]):
                        cnt_rank += 1
            rankingloss += cnt_rank/(dim_ti*(numLabel-dim_ti))
        # print(rankingloss- test_rankingloss)

        if(dim_ti == numLabel or dim_ti == 0):
            avg_precision += 1
        else:
            cnt_pre = 0
            for j in range(numLabel):
                if(target[i][j]==0):
                    continue
                tmp = 0
                for k in range(numLabel):
                    if(target[i][k]==0):
                        continue
                    if(rank[i][j] >= rank[i][k]):
                        tmp += 1
                cnt_pre += tmp/rank[i][j]
            avg_precision += cnt_pre/dim_ti

    oneerror /= dimData
    coverage /= dimData
    coverage -= 1
    coverage /= numLabel
    rankingloss /= dimData
    avg_precision /= dimData
    
    predict = np.rint(predict)
    
    acc,precision,recall,f1,hamming,hitrate,subsetAcc = 0,0,0,0,0,0,0
    
    for i in range(dimData):
        a,b,c,d = 0,0,0,0
        for j in range(numLabel):
            if(predict[i][j]>=0.5 and target[i][j]>=0.5):
                a +=1
            if(predict[i][j]>=0.5 and target[i][j]<0.5):
                c +=1
            if(predict[i][j]<0.5 and target[i][j]>=0.5):
                b +=1
            if(predict[i][j]<0.5 and target[i][j]<0.5):
                d +=1
        if(a+b+c==0):
            acc += 1
        else:
            acc += a/(a+b+c)
        if(a>0):
            hitrate += 1
        if(a+b+c == 0):
            precision += 1
            recall += 1
        else:
            if(a+c !=0 ):
                precision += a/(a+c)
            if(a+b !=0 ):
                recall += a/(a+b)
        if(2*a+b+c==0):
            f1 += 1
        else:
            f1 += 2*a/(2*a+b+c)
        hamming += (b+c)/(a+b+c+d)
        if(b==0 and c==0):
            subsetAcc += 1
    acc /= dimData
    precision /= dimData
    recall /= dimData
    f1 /= dimData
    hitrate /= dimData
    subsetAcc /= dimData
    hamming /= dimData

    value = list()
    value.append(acc)
    value.append(precision)
    value.append(recall)
    value.append(f1)
    value.append(hitrate)
    value.append(subsetAcc)
    value.append(hamming)

    value.append(oneerror)
    value.append(coverage)
    value.append(rankingloss)
    value.append(avg_precision)
    return value

def getSeq(y):
    seq = []
    for i in range(np.shape(y)[0]):
        seq.append(arraysort(y[i]))
    return np.array(seq)

def getRank(seq):
    rank = np.zeros(np.shape(seq))
    for i in range(np.shape(seq)[0]):
        for j in range(np.shape(seq)[1]):
            indexb = int(seq[i][j])
            rank[i][indexb] = j+1
    return rank

def arraysort(org_arr):
    length = np.shape(org_arr)[0]
    index = np.zeros(length)
    arr = np.zeros(length)
    for i in range(length):
        index[i] = int(i)
        arr[i] = org_arr[i]
    temp = 0
    thisIndex = 0
    for i in range(length):
        for j in range(length-i-1):
            if(arr[j] < arr[j + 1]):
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp

                thisIndex = index[j]
                index[j] = index[j + 1]
                index[j + 1] = thisIndex
    return index

def evaluate_SLacc(predict, target):
    num_data,num_label = np.shape(predict)
    predict = np.transpose(predict)
    target = np.transpose(target)
    accs = []
    for i in range(num_label):
        accs.append(sum((predict[i]>=0.5)-target[i]==0))
    return np.array(accs)/num_data

if __name__ == '__main__':
    tp = [[0.2591821, 0.487243769, 0.632291599, 0.040377205, 0.0892396, 0.070158688],
        [0.2591821, 0.26925271, 0.637690909, 0.87574167, 0.20609274, 0.070158688],
        [0.062028116, 0.106169666, 0.823550372, 0.319658476, 0.580624134, 0.070158688],
        [0.118468189, 0.258392936, 0.632291599, 0.7183586, 0.791985774, 0.070158688]]
    # seq = getSeq(tp[0:4])
    # print(seq)
    # print(getRank(seq))
    # pp = [[1,1,1,1,0,0], [1,1,1,1,0,0], [1,1,1,1,0,0], [1,1,1,1,0,0]]
    # print(evaluate_SLacc(tp, pp))
