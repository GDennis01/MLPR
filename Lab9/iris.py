import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.special
import sklearn.datasets
from svm import SVM
from prettytable import PrettyTable
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)
def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc
def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbfKernelFunc
def logreg_scores(w,b,DTE):
    return np.dot(w.T,DTE) + b

def get_dcf(conf_matrix,prior,Cfn,Cfp,normalized=False):
    _dcf = prior*Cfn*get_false_negatives(conf_matrix) + (1-prior)*Cfp*get_false_positives(conf_matrix)
    if normalized:
        return _dcf/min(prior*Cfn,(1-prior)*Cfp)
    return _dcf
def get_confusion_matrix(predicted,actual):
    n_labels = len(np.unique(actual))
    confusion_matrix = np.zeros((n_labels,n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            confusion_matrix[i,j] = np.sum((predicted==i)&(actual==j))
    return confusion_matrix
def get_false_negatives(conf_matrix):
    return conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[1,1])
def get_false_positives(conf_matrix):
    return conf_matrix[1,0]/(conf_matrix[0,0]+conf_matrix[1,0])
def get_true_positives(conf_matrix):
    return 1-get_false_negatives(conf_matrix)
def get_min_dcf(SVAL,LVAL,prior,Cfn,Cfp):
    thresholds = np.concatenate([np.array([-np.inf]), SVAL, np.array([np.inf])])
    min_dcf = np.min([get_dcf(get_confusion_matrix((SVAL>t),LVAL),prior,1,1,normalized=True) for t in thresholds])
    return min_dcf
if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    svm = SVM(DTR, LTR)

    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            w, b = svm.train(C, 'linear', K)
            SVAL = (vrow(w) @ DVAL + b).ravel()
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            print ('Error rate: %.1f' % (err*100))
            conf_matrix = get_confusion_matrix(PVAL, LVAL)
            print ('minDCF - pT = 0.5: %.4f' % get_min_dcf(SVAL, LVAL, 0.5, 1.0, 1.0))
            print ('actDCF - pT = 0.5: %.4f' % get_dcf(conf_matrix, 0.5, 1.0, 1.0,normalized=True))
            print ()
        print("--------------------")

    # for kernelFunc in [polyKernel(2, 0), polyKernel(2, 1), rbfKernel(1.0), rbfKernel(10.0)]:
    #     for eps in [0.0, 1.0]:
    #         fScore = svm.train(1.0,'kernel',eps,kernelFunc)
    #         SVAL = fScore(DVAL)
       
    #         PVAL = (SVAL > 0) * 1
    #         err = (PVAL != LVAL).sum() / float(LVAL.size)
    #         print ('Error rate: %.1f' % (err*100))
    #         conf_matrix = get_confusion_matrix(PVAL, LVAL)

    #         print ('minDCF - pT = 0.5: %.4f' % get_min_dcf(SVAL, LVAL, 0.5, 1.0, 1.0))
    #         print ('actDCF - pT = 0.5: %.4f' % get_dcf(conf_matrix, 0.5, 1.0, 1.0,normalized=True))
    #         print ()