import numpy as np

def saveMat(mat, algname='rst'):
    fname = algname + '.txt'
    if(len(np.shape(mat))==2):
        saveMat2(mat, fname)
    else:
        saveMat1(mat, fname)

def saveMat2(mat, fname):
    f = open(fname, 'a')
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            print(mat[i][j], end='\t', file=f)
        print('', file=f)
    f.close()

def saveMat1(mat, fname):
    f = open(fname, 'a')
    for i in range(len(mat)):
        print(mat[i], end='\t', file=f)
    print('', file=f)
    f.close()
