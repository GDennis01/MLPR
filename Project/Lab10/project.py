#models
from Project.libs.gmm import *
from Project.libs.logistic_regression import *
from Project.libs.svm import *
#utils
from Project.libs.utils import load,split_db_2to1
from Project.libs.bayes_risk import get_dcf,get_min_dcf

import numpy as np

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
    # gmm0_list = {'full':{},'diagonal':{}}
    # gmm1_list = {'full':{},'diagonal':{}}

    # for covType in ['full','diagonal']:
    #     print(f'{covType} GMM')
        
    #     for numC in [1,2,4,8,16]:
    #         gmm0 = train_GMM_LBG_EM(DTR[:,LTR==0],numC,covType,verbose=False)
    #         gmm1 = train_GMM_LBG_EM(DTR[:,LTR==1],numC,covType,verbose=False)

    #         gmm0_list[covType][numC] = gmm0
    #         gmm1_list[covType][numC] = gmm1
    # best_setup_gmm = []
    # best_min_dcf = np.inf
    # for gmm0_elem in gmm0_list[covType]:   
    #     for gmm1_elem in gmm1_list[covType]:
    #         for covType in ['full','diagonal']:
    #             gmm0 = gmm0_list[covType][gmm0_elem]
    #             for covType2 in ['full','diagonal']:
    #                 print(f'Trying {covType} GMM with {gmm0_elem} and {gmm1_elem} components')
    #                 gmm1 = gmm1_list[covType2][gmm1_elem]

    #                 scores = gmm_scores(gmm0,gmm1,DTE)
    #                 dcf = get_dcf(scores,LTE,prior,1.0,1.0,normalized=True,threshold='optimal')
    #                 min_dcf = get_min_dcf(scores,LTE,prior,1.0,1.0)
    #                 if min_dcf < best_min_dcf:
    #                     best_min_dcf = min_dcf
    #                     best_setup_gmm = [covType,covType2,gmm0_elem,gmm1_elem,min_dcf,dcf]  

    #                 print("Number of components: ",numC)
    #                 print("DCF: ",dcf)
    #                 print("MinDCF: ",min_dcf)
    #                 print()
    #endregion

    #region Best candidate for LogReg
    best_min_dcf = np.inf
    best_lreg_setup = []
    for quadratic in [True,False]:
        for logreg_type in ['binary','weighted']:
            print(f'Trying {logreg_type} LogReg with {"quadratic" if quadratic else "non-quadratic"} features')
            # for l in np.logspace(-5,5,11):
            for l in np.logspace(-4,2,13):
            
                # lrc = LogRegClassifier(DTR,LTR)
                lrc = LogRegClassifier(features,classes)
                lrc.train(logreg_type,l,pT=prior,expaded_feature=quadratic)
    

                empirical_prior = (lrc.LTR == 1).sum()/lrc.LTR.size
                if logreg_type == "binary":
                    scores_llr = lrc.logreg_scores - np.log(empirical_prior/(1-empirical_prior))
                elif logreg_type == "weighted":
                    scores_llr = lrc.logreg_scores - np.log(prior/(1-prior))

                dcf = get_dcf(scores_llr, lrc.LTE, prior, 1.0, 1.0, normalized=True, threshold='optimal')
                min_dcf =  get_min_dcf(scores_llr, lrc.LTE, prior, 1.0, 1.0)
                if min_dcf < best_min_dcf:
                    best_min_dcf = min_dcf
                    best_lreg_setup = [logreg_type,l,'quadratic'if quadratic else 'non-quadratic',min_dcf,dcf]
                print(f'MinDCF: {min_dcf}')
    print(f'Best setup is obtained with {best_lreg_setup[2]} {best_lreg_setup[0]} LogReg with lambda={best_lreg_setup[1]}')
    print(f'MinDCF: {best_lreg_setup[3]}')
    #endregion

    #region Best candidate for SVM
    # best_min_dcf = np.inf
    # best_setup_svm = []
    # for svm_type in ['linear','kernel']:
    #     for C in np.logspace(-5,5,11):
    #         if svm_type == 'kernel':
    #             # gamma = e-2 empirically chosen as "best gamma" in Lab9 through observations on mindcf graphs
    #             for kernel in [polyKernel(2,1),rbfKernel(np.exp(-2))]:
    #                 svm = SVM(DTR,LTR)
    #                 fscore = svm.train(C,svm_type,kernelFunc=kernel)
    #                 scores = fscore(svm.DTE)

    #                 min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
    #                 dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0, normalized=True, threshold='optimal')
    #                 if min_dcf < best_min_dcf:
    #                     best_min_dcf = min_dcf
    #                     best_setup_svm = [svm_type,C,kernel.__name__,min_dcf,dcf]
    #         else:
    #             svm = SVM(DTR,LTR)
    #             w,b= svm.train(C,svm_type)
    #             scores = svm.scores

    #             min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
    #             dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0, normalized=True, threshold='optimal')
    #             if min_dcf < best_min_dcf:
    #                 best_min_dcf = min_dcf
    #                 best_setup_svm = [svm_type,C,min_dcf,dcf]
    #         print(f'MinDCF: {min_dcf}')
    # print(f'Best setup is obtained with {best_setup_svm[0]} SVM with C={best_setup_svm[1]}')
    # if best_setup_svm[0] == 'kernel':
    #     print(f'Best kernel is {best_setup_svm[2]}')
    #     print(f'MinDCF: {best_setup_svm[3]}')
    # else:
    #     print(f'MinDCF: {best_setup_svm[2]}')
    #Best setup is obtained with kernel SVM with C=10.0
    #Best kernel is rbfKernelFunc
    #MinDCF: 0.215666748341587
    #endregion

    #region Best candidate for GMM
    # print(f'Best setup is obtained with class 0 as {best_setup_gmm[0]} GMM with {best_setup_gmm[2]} components and class 1 as {best_setup_gmm[1]} GMM with {best_setup_gmm[3]} components with minDCF: {best_setup_gmm[4]}')

    #endregion

    #region Bayes error plots for different applications with the chosen setups
    #debug only
    # best_setup_svm = ['kernel',10,'rbfKernelFunc',0.215666748341587,0.215666748341587]
    # best_lreg_setup =[ 'binary', 0.01, 'quadratic', 0.215666748341587, 0.215666748341587]
    # best_setup_gmm=['diagonal','full',8,16,0.215666748341587,0.215666748341587]
    
    # N_STOPS = 20

    # eff_prior_log_odds = np.linspace(-4,4,N_STOPS)
    # priors = [1 / (1 + np.exp(-x)) for x in eff_prior_log_odds]

    # Models = ['LogReg','SVM','GMM']
    # dcfs = np.empty((len(Models),N_STOPS))
    # min_dcfs = np.empty((len(Models),N_STOPS))  
    # for model in Models:  
    #     for prior in priors:  
    #         match model:
    #             case 'LogReg':
    #                 lrc = LogRegClassifier(DTR,LTR)
    #                 lrc.train(best_lreg_setup[0],best_lreg_setup[1],pT=prior,expaded_feature=best_lreg_setup[2]=='quadratic')
    #                 empirical_prior = (lrc.LTR == 1).sum()/lrc.LTR.size
    #                 if best_lreg_setup[0] == "binary":
    #                     scores_llr = lrc.logreg_scores - np.log(empirical_prior/(1-empirical_prior))
    #                 elif best_lreg_setup[0] == "weighted":
    #                     scores_llr = lrc.logreg_scores - np.log(prior/(1-prior))
    #                 dcf = get_dcf(scores_llr, lrc.LTE, prior, 1.0, 1.0, normalized=True, threshold='optimal')
    #                 min_dcf =  get_min_dcf(scores_llr, lrc.LTE, prior, 1.0, 1.0)
    #                 dcfs[Models.index(model),priors.index(prior)] = dcf
    #                 min_dcfs[Models.index(model),priors.index(prior)] = min_dcf
    #             case 'SVM':
    #                 svm = SVM(DTR,LTR)
    #                 if best_setup_svm[0] == 'kernel':
    #                     if best_setup_svm[2] == 'polyKernelFunc':
    #                         kernelFunc = polyKernel(2,1)
    #                     elif best_setup_svm[2] == 'rbfKernelFunc':
    #                         kernelFunc = rbfKernel(np.exp(-2))

    #                     fscore = svm.train(best_setup_svm[1],best_setup_svm[0],kernelFunc=kernelFunc)
    #                     scores = fscore(svm.DTE)
    #                 else:
    #                     w,b= svm.train(best_setup_svm[1],best_setup_svm[0])
    #                     scores = svm.scores
    #                 min_dcf =  get_min_dcf(scores, svm.LTE, prior, 1.0, 1.0)
    #                 dcf = get_dcf(scores, svm.LTE, prior, 1.0, 1.0, normalized=True, threshold='optimal')
    #                 dcfs[Models.index(model),priors.index(prior)] = dcf
    #                 min_dcfs[Models.index(model),priors.index(prior)] = min_dcf
    #             case 'GMM':
    #                 gmm0 = train_GMM_LBG_EM(DTR[:,LTR==0],best_setup_gmm[2],best_setup_gmm[0],verbose=False)
    #                 gmm1 = train_GMM_LBG_EM(DTR[:,LTR==1],best_setup_gmm[3],best_setup_gmm[1],verbose=False)
    #                 scores = gmm_scores(gmm0,gmm1,DTE)
    #                 dcf = get_dcf(scores,LTE,prior,1.0,1.0,normalized=True,threshold='optimal')
    #                 min_dcf = get_min_dcf(scores,LTE,prior,1.0,1.0)
    #                 dcfs[Models.index(model),priors.index(prior)] = dcf
    #                 min_dcfs[Models.index(model),priors.index(prior)] = min_dcf
    # for model in Models:
    #     plt.plot(eff_prior_log_odds,dcfs[Models.index(model)],label=f'{model} DCF')
    #     plt.plot(eff_prior_log_odds,min_dcfs[Models.index(model)],label=f'{model} MinDCF')
    #     plt.xlabel('Log odds of effective prior')
    #     plt.ylabel(f'{model} - (Min)DCF')
    #     plt.legend()
    #     plt.show()
    #endregion