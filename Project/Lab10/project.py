#models
from Project.libs.gmm import *
from Project.libs.logistic_regression import *
from Project.libs.svm import *
#utils
from Project.libs.utils import load,split_db_2to1
from Project.libs.bayes_risk import get_dcf,get_min_dcf

import numpy as np
BEST_SETUP_GMM = {'type':'GMM','min_dcf': np.inf,'act_dcf':None, 'nc0':None,'nc1':None,'covType0':None,'covType1':None}
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



def gmm_scores(gmm0,gmm1,DTE):
    scores0 = logpdf_GMM(DTE, gmm0)
    scores1 = logpdf_GMM(DTE, gmm1)
    return scores1 - scores0
def Lab10():
    (features,classes) = load("project/data/trainData.txt")
    (DTR, LTR), (DTE, LTE) = split_db_2to1(features, classes)
    prior = 0.1


    #region Full and Diagonal GMM
    #FIXME: with 32 it crashes when trying to invert the covariance matrix,
    gmm0_list = {'full':{},'diagonal':{}}
    gmm1_list = {'full':{},'diagonal':{}}

    for covType in ['full','diagonal']:
        print(f'{covType} GMM')
        
        for numC in [1,2,4,8,16]:
            gmm0 = train_GMM_LBG_EM(DTR[:,LTR==0],numC,covType,verbose=False)
            gmm1 = train_GMM_LBG_EM(DTR[:,LTR==1],numC,covType,verbose=False)

            gmm0_list[covType][numC] = gmm0
            gmm1_list[covType][numC] = gmm1
    best_setup_gmm = []
    best_min_dcf = np.inf
    for gmm0_elem in gmm0_list[covType]:   
        for gmm1_elem in gmm1_list[covType]:
            for covType in ['full','diagonal']:
                gmm0 = gmm0_list[covType][gmm0_elem]
                for covType2 in ['full','diagonal']:
                    # print(f'Trying {covType} GMM with {gmm0_elem} and {gmm1_elem} components')
                    gmm1 = gmm1_list[covType2][gmm1_elem]

                    scores = gmm_scores(gmm0,gmm1,DTE)
                    dcf = get_dcf(scores,LTE,prior,1.0,1.0,normalized=True,threshold='optimal')
                    min_dcf = get_min_dcf(scores,LTE,prior,1.0,1.0)
                    if min_dcf < best_min_dcf:
                        best_min_dcf = min_dcf
                        best_setup_gmm = [covType,covType2,gmm0_elem,gmm1_elem,min_dcf,dcf]  
                        BEST_SETUP_GMM['nc0'] = gmm0_elem
                        BEST_SETUP_GMM['nc1'] = gmm1_elem
                        BEST_SETUP_GMM['covType0'] = covType
                        BEST_SETUP_GMM['covType1'] = covType2
                        BEST_SETUP_GMM['min_dcf'] = min_dcf
                        BEST_SETUP_GMM['act_dcf'] = dcf
    #endregion

    #region Loading best setups
    import json
    with open('Project/best_setups/best_setup_gmm','w') as f:
        json.dump(BEST_SETUP_GMM,f)
    print(f'Best setup is obtained with class 0 with {BEST_SETUP_GMM["covType0"]} GMM with {BEST_SETUP_GMM["nc0"]} components and class 1 with {BEST_SETUP_GMM["covType1"]} GMM with {BEST_SETUP_GMM["nc1"]} components')
    print(f'MinDCF: {BEST_SETUP_GMM["min_dcf"]}')
    print(f'Actual DCF: {BEST_SETUP_GMM["act_dcf"]}')
    print("------------------------------------")
    with open('Project/best_setups/best_setup_logreg.json') as f:
        best_setup_logreg = json.load(f)
    print(f'Best setup is obtained with {best_setup_logreg["model"]} {best_setup_logreg["type"]} LogReg with lambda={best_setup_logreg["l"]} and {"quadratic" if best_setup_logreg["expanded_feature"] else "non-quadratic"} features')
    print(f'MinDCF: {best_setup_logreg["min_dcf"]}')
    print(f'Actual DCF: {best_setup_logreg["act_dcf"]}')
    print(best_setup_logreg)
    print("------------------------------------")
    with open('Project/best_setups/best_setup_svm.json') as f:
        best_setup_svm = json.load(f)
    print(f'Best setup is obtained with {best_setup_svm["mode"]} SVM with C={best_setup_svm["C"]} and gamma={best_setup_svm["gamma"]}')
    print(f'MinDCF: {best_setup_svm["min_dcf"]}')
    print(f'Actual DCF: {best_setup_svm["act_dcf"]}')
    print("------------------------------------")
    #endregion


    #region Bayes error plots for different applications with the chosen setups
    
    N_STOPS = 20

    eff_prior_log_odds = np.linspace(-4,4,N_STOPS)
    priors = [1 / (1 + np.exp(-x)) for x in eff_prior_log_odds]

    Models = ['LogReg','SVM','GMM']
    dcfs = np.empty((len(Models),N_STOPS))
    min_dcfs = np.empty((len(Models),N_STOPS))  
    for model in Models:  
        for prior in priors:  
            match model:
                case 'LogReg':
                    lrc = LogRegClassifier(features,classes)
                    lrc.train(best_setup_logreg['model'],l=best_setup_logreg['l'],pT=prior,expaded_feature=best_setup_logreg['expanded_feature'])
                    empirical_prior = (lrc.LTR == 1).sum()/lrc.LTR.size
                    if best_setup_logreg["model"] == "binary":
                        scores_llr = lrc.logreg_scores - np.log(empirical_prior/(1-empirical_prior))
                    elif best_setup_logreg["model"] == "weighted":
                        scores_llr = lrc.logreg_scores - np.log(prior/(1-prior))
                    dcf = get_dcf(scores_llr, lrc.LTE, prior, 1.0, 1.0, normalized=True, threshold='optimal')
                    min_dcf =  get_min_dcf(scores_llr, lrc.LTE, prior, 1.0, 1.0)
                    dcfs[Models.index(model),priors.index(prior)] = dcf
                    min_dcfs[Models.index(model),priors.index(prior)] = min_dcf
                case 'SVM':
                    svm = SVM(features,classes)
                    if best_setup_svm['mode'] == 'Kernel RBF':
                        kernelFunc = rbfKernel(best_setup_svm['gamma'])
                        fscore = svm.train(best_setup_svm['C'],svm_type='kernel',kernelFunc=kernelFunc)
                        scores = fscore(svm.DTE)
                    elif best_setup_svm['mode'] == 'Kernel Poly 2-1':
                        kernelFunc = polyKernel(2,1)
                        fscore = svm.train(best_setup_svm['C'],svm_type='kernel',kernelFunc=kernelFunc)
                        scores = fscore(svm.DTE)
                    elif best_setup_svm['mode'] == 'Linear':
                        w,b= svm.train(best_setup_svm[1],best_setup_svm[0])
                        scores = svm.scores
                    min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
                    dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0, normalized=True, threshold='optimal')
                    dcfs[Models.index(model),priors.index(prior)] = dcf
                    min_dcfs[Models.index(model),priors.index(prior)] = min_dcf
                case 'GMM':
                    gmm0 = train_GMM_LBG_EM(DTR[:,LTR==0],BEST_SETUP_GMM['nc0'],BEST_SETUP_GMM['covType0'],verbose=False)
                    gmm1 = train_GMM_LBG_EM(DTR[:,LTR==1],BEST_SETUP_GMM['nc1'],BEST_SETUP_GMM['covType1'],verbose=False)
                    scores = gmm_scores(gmm0,gmm1,DTE)
                    dcf = get_dcf(scores,LTE,prior,1.0,1.0,normalized=True,threshold='optimal')
                    min_dcf = get_min_dcf(scores,LTE,prior,1.0,1.0)
                    dcfs[Models.index(model),priors.index(prior)] = dcf
                    min_dcfs[Models.index(model),priors.index(prior)] = min_dcf
    for model in Models:
        plt.plot(eff_prior_log_odds,dcfs[Models.index(model)],label=f'{model} DCF')
        plt.plot(eff_prior_log_odds,min_dcfs[Models.index(model)],label=f'{model} MinDCF')
        plt.xlabel('Log odds of effective prior')
        plt.ylabel(f'{model} - (Min)DCF')
        plt.legend()
        plt.show()
    #endregion