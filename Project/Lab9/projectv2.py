import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
from Project.libs.svm import SVM
from prettytable import PrettyTable
from Project.libs.utils import load,vcol,vrow,split_db_2to1
from Project.libs.bayes_risk import compute_optimal_Bayes_binary_threshold,get_dcf,get_min_dcf,get_confusion_matrix


# needed for the "quadratic" logistic regression, basically apply the expanded feature set to the normal logreg
def quadratic_feature_expansion(X):
    X_T = X.T
    X_expanded = []
    for x in X_T:
        outer_product = np.outer(x, x).flatten()
        expanded_feature = np.concatenate([outer_product, x])
        X_expanded.append(expanded_feature)
    X_expanded = np.array(X_expanded).T
    return X_expanded



def plot_dcf_vs_c(lambdas,dcfs,min_dcfs):
    plt.figure()
    plt.plot(lambdas,dcfs,label='DCFs')
    plt.plot(lambdas,min_dcfs,label='Min DCFs')
    plt.xscale('log',base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs C')
    plt.legend()
    plt.show()

# Optimize SVM


# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
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


    
def Lab9():
    features,classes=load("project/data/trainData.txt")
    # take only 1/10 of the data
    features = features[:,:int(features.shape[1]/10)]
    classes = classes[:int(classes.size/10)]
    prior = 0.1
    svm = SVM(features,classes)

    #region Linear SVM with DCF and MinDCF as C varies
    dcfs = []
    min_dcfs = []
    for C in np.logspace(-5,0,11):
        w,b= svm.train(C,'linear',K=1)
        scores = svm.scores

        min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
        min_dcfs.append(min_dcf)

        dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0,normalized=True,threshold='optimal')
        dcfs.append(dcf)
    plot_dcf_vs_c(np.logspace(-5,0,11),dcfs,min_dcfs)
    #endregion

    #region Polynomial Kernel SVM with DCF and MinDCF as C varies
    dcfs = []
    min_dcfs = []
    for C in np.logspace(-5,0,11):
        fscore= svm.train(C,'kernel',K=1,kernelFunc=polyKernel(2,1))
        scores = fscore(svm.DTE)

        min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
        min_dcfs.append(min_dcf)

        dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0,normalized=True,threshold='optimal')
        dcfs.append(dcf)
    plot_dcf_vs_c(np.logspace(-5,0,11),dcfs,min_dcfs)
    #endregion

    #region RBF Kernel SVM with DCF and MinDCF as C varies
    gamma = [("e-4",np.exp(-4)),("e-3",np.exp(-3)),("e-2",np.exp(-2)),("e-1",np.exp(-1))]
    Cs = np.logspace(-3,2,11)
    dcfs = []
    min_dcfs = []
    for l,g in gamma:
        dcfs = []
        min_dcfs = []
        for C in Cs:
            fscore= svm.train(C,'kernel',kernelFunc=rbfKernel(g))
            scores = fscore(svm.DTE)

            min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
            min_dcfs.append(min_dcf)

            dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0,normalized=True,threshold='optimal')
            dcfs.append(dcf)
        plt.plot(Cs,dcfs,label=f'DCF with Gamma: {l}')
        plt.plot(Cs,min_dcfs,label=f'MinDCF with Gamma: {l}')

    plt.xscale('log',base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs C')
    plt.legend()
    plt.show()
    #endregion
        

