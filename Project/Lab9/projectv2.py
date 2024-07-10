import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
from Project.libs.svm import SVM
from prettytable import PrettyTable
from Project.libs.utils import load,vcol,vrow,split_db_2to1
from Project.libs.bayes_risk import compute_optimal_Bayes_binary_threshold,get_dcf,get_min_dcf,get_confusion_matrix
BEST_SETUP_SVM = {'type':'SVM','min_dcf': np.inf,'act_dcf':None, 'C':None,'mode':None,'gamma':None,'Centered':None,'scores':None}
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
    # to speed up computations, only 1/10 of the data is used
    # features = features[:,:int(features.shape[1]/10)]
    # classes = classes[:int(classes.size/10)]
    prior = 0.1

    #region Linear SVM with DCF and MinDCF as C varies
    svm = SVM(features,classes)
    dcfs = []
    min_dcfs = []
    for C in np.logspace(-5,0,11):
        w,b= svm.train(C,'linear',K=1)
        scores = svm.scores

        min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
        min_dcfs.append(min_dcf)

        dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0,normalized=True,threshold='optimal')
        dcfs.append(dcf)
        if min_dcf < BEST_SETUP_SVM['min_dcf']:
                BEST_SETUP_SVM['min_dcf'] = min_dcf
                BEST_SETUP_SVM['act_dcf'] = dcf
                BEST_SETUP_SVM['C'] = C
                BEST_SETUP_SVM['mode'] = 'Linear'
                BEST_SETUP_SVM['Centered'] = False
                BEST_SETUP_SVM['scores'] = scores.tolist()
    plot_dcf_vs_c(np.logspace(-5,0,11),dcfs,min_dcfs)

    #testing with centered data
    svm = SVM(features,classes)
    svm.DTR = svm.DTR - svm.DTR.mean(1,keepdims=True)
    svm.DTE = svm.DTE - svm.DTR.mean(1,keepdims=True)
    dcfs = []
    min_dcfs = []
    for C in np.logspace(-5,0,11):
        w,b= svm.train(C,'linear',K=1)
        scores = svm.scores

        min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
        min_dcfs.append(min_dcf)

        dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0,normalized=True,threshold='optimal')
        dcfs.append(dcf)
        if min_dcf < BEST_SETUP_SVM['min_dcf']:
                BEST_SETUP_SVM['min_dcf'] = min_dcf
                BEST_SETUP_SVM['act_dcf'] = dcf
                BEST_SETUP_SVM['C'] = C
                BEST_SETUP_SVM['mode'] = 'Linear'
                BEST_SETUP_SVM['Centered'] = True
                BEST_SETUP_SVM['scores'] = scores.tolist()
    plot_dcf_vs_c(np.logspace(-5,0,11),dcfs,min_dcfs)
    #endregion

    #region Polynomial Kernel SVM with DCF and MinDCF as C varies
    svm = SVM(features,classes)
    dcfs = []
    min_dcfs = []
    for C in np.logspace(-5,0,11):
        fscore= svm.train(C,'kernel',K=1,kernelFunc=polyKernel(2,1))
        scores = fscore(svm.DTE)

        min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
        min_dcfs.append(min_dcf)

        dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0,normalized=True,threshold='optimal')
        dcfs.append(dcf)
        if min_dcf < BEST_SETUP_SVM['min_dcf']:
                best_min_dcf = min_dcf
                BEST_SETUP_SVM['min_dcf'] = min_dcf
                BEST_SETUP_SVM['act_dcf'] = dcf
                BEST_SETUP_SVM['C'] = C
                BEST_SETUP_SVM['mode'] = 'Kernel Poly 2-1'
                BEST_SETUP_SVM['Centered'] = False
                BEST_SETUP_SVM['scores'] = scores.tolist()
    plot_dcf_vs_c(np.logspace(-5,0,11),dcfs,min_dcfs)
    #endregion

    #region RBF Kernel SVM with DCF and MinDCF as C varies
    svm = SVM(features,classes)
    gamma = [("e-4",np.exp(-4)),("e-3",np.exp(-3)),("e-2",np.exp(-2)),("e-1",np.exp(-1))]
    Cs = np.logspace(-3,2,11)
    dcfs = []
    min_dcfs = []
    best_setup=[]
    best_min_dcf = np.inf
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
            if min_dcf < BEST_SETUP_SVM['min_dcf']:
                best_min_dcf = min_dcf
                BEST_SETUP_SVM['min_dcf'] = min_dcf
                BEST_SETUP_SVM['C'] = C
                BEST_SETUP_SVM['gamma'] = g
                BEST_SETUP_SVM['act_dcf'] = dcf
                BEST_SETUP_SVM['mode'] = 'Kernel RBF'
                BEST_SETUP_SVM['Centered'] = False
                BEST_SETUP_SVM['scores'] = scores.tolist()
        plt.plot(Cs,dcfs,label=f'DCF with Gamma: {l}')
        plt.plot(Cs,min_dcfs,label=f'MinDCF with Gamma: {l}')
        plt.xscale('log',base=10)
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.title('DCF vs C')
        plt.legend()
        plt.show()
    print(f'Best setup: {BEST_SETUP_SVM["mode"]} with MinDCF: {BEST_SETUP_SVM["min_dcf"]} and Actual DCF: {BEST_SETUP_SVM["act_dcf"]} with C: {BEST_SETUP_SVM["C"]} and Gamma: {BEST_SETUP_SVM["gamma"]}')
    import json
    with open('Project/best_setups/best_setup_svm.json', 'w') as f:
        json.dump(BEST_SETUP_SVM, f)
    #endregion
        

