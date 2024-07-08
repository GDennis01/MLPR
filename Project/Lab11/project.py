from Project.libs.svm import *
from Project.libs.logistic_regression import LogRegClassifier
from Project.libs.gmm import *
from Project.libs.utils import load,split_db_2to1
from Project.libs.bayes_risk import *
import numpy as np
import json

TARGET_PRIOR = 0.1
KFOLD = 5

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbfKernelFunc
def extract_train_val_folds_from_ary(X, idx):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]
def calibrate(scores,LTE,prior_cal):
    lr_scores = []
    labels = []
    for idx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(scores, idx)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, idx)

        lrc = LogRegClassifier.with_details(vrow(SCAL), LCAL, vrow(SVAL), LVAL)
        w,b = lrc.train('weighted',l=0,pT=prior_cal)

        lr_score = lrc.llr_scores
        lr_scores.append(lr_score)
        labels.append(LVAL)
    return lr_scores, labels
def bayes_cal_plot(cal_dcfs,dcfs,min_dcfs,eff_prior_log_odds,model):
    plt.plot(eff_prior_log_odds,cal_dcfs,label=f'{model} DCF(pre-calibration)')
    plt.plot(eff_prior_log_odds,dcfs,label=f'{model} DCF(post-calibration)')
    plt.plot(eff_prior_log_odds,min_dcfs,label=f'{model} MIN DCF',linestyle='--') 
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('(MIN) DCF')  
    plt.legend()
    plt.show()

def Lab11():
    (features,classes) = load("project/data/trainData.txt")
    _, (DTE, LTE) = split_db_2to1(features, classes)
    #importing best setups from previous labs
    with open('Project/best_setups/best_setup_svm.json') as f:
        BEST_SETUP_SVM = json.load(f)
        BEST_SETUP_SVM['scores'] = np.array(BEST_SETUP_SVM['scores'])
    with open('Project/best_setups/best_setup_gmm.json') as f:
        BEST_SETUP_GMM = json.load(f)
        BEST_SETUP_GMM['scores'] = np.array(BEST_SETUP_GMM['scores'])
    with open('Project/best_setups/best_setup_logreg.json') as f:
        BEST_SETUP_LOGREG = json.load(f)
        BEST_SETUP_LOGREG['scores'] = np.array(BEST_SETUP_LOGREG['scores'])


    priors = np.linspace(0.1,0.9,9)
    best_scores_transformations = {
        'gmm':{'prior':None,'actdcf':None,'scores':None,'labels':None},
        'svm':{'prior':None,'actdcf':None,'scores':None,'labels':None},
        'logreg':{'prior':None,'actdcf':None,'scores':None,'labels':None}
        }
    for model in ['svm','gmm','logreg']:
        best_act_dcf = np.inf 
        current_setup = eval("BEST_SETUP_"+model.upper())
        for prior in priors:
            lr_scores,labels = calibrate(current_setup['scores'],LTE,prior)

            calibrated_scores = np.hstack(lr_scores)
            labels = np.hstack(labels)

            #computing act dcf and min dcf, then choosing the best score transformation for each model based on act dcf
            calibrated_adcf = get_dcf(calibrated_scores,labels,TARGET_PRIOR,normalized=True,threshold='optimal')
            calibrated_mdcf = get_min_dcf(calibrated_scores,labels,TARGET_PRIOR)

            if calibrated_adcf < best_act_dcf:
                best_act_dcf = calibrated_adcf
                best_scores_transformations[model]['scores'] = calibrated_scores
                best_scores_transformations[model]['labels'] = labels
                best_scores_transformations[model]['actdcf'] = calibrated_adcf
                best_scores_transformations[model]['prior'] = prior
    #fusion
    fused_scores = []
    fused_labels = []
    for idx in range(KFOLD):
        SCAL_GMM, SVAL_GMM = extract_train_val_folds_from_ary(BEST_SETUP_GMM['scores'], idx)
        SCAL_SVM, SVAL_SVM = extract_train_val_folds_from_ary(BEST_SETUP_SVM['scores'], idx)
        SCAL_LOGREG, SVAL_LOGREG = extract_train_val_folds_from_ary(BEST_SETUP_GMM['scores'], idx)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, idx)

        SCAL = np.vstack([SCAL_GMM,SCAL_SVM,SCAL_LOGREG])
        SVAL = np.vstack([SVAL_GMM,SVAL_SVM,SVAL_LOGREG])

        lrc = LogRegClassifier.with_details(vrow(SCAL), LCAL, vrow(SVAL), LVAL)
        w,b = lrc.train('weighted',l=0,pT=TARGET_PRIOR)

        lr_score = lrc.llr_scores
        fused_scores.append(lr_score)
        fused_labels.append(LVAL)
        
    fused_scores = np.hstack(fused_scores)
    fused_labels = np.hstack(fused_labels)
    fused_adcf = get_dcf(fused_scores,fused_labels,TARGET_PRIOR,normalized=True,threshold='optimal')
    fused_mdcf = get_min_dcf(fused_scores,fused_labels,TARGET_PRIOR)
    print(f'Fused ADCF: {fused_adcf}')
    print(f'Fused MDCF: {fused_mdcf}')
        

    # best_precal_dcfs = bayes_error_plot(current_setup['scores'],best_scores_transformations[model]['labels'],left=-4,right=4,n_points=100)[1]
    # eff_lo_priors,dcfs,mindcfs = bayes_error_plot(best_scores_transformations[model]['scores'],best_scores_transformations[model]['labels'],left=-4,right=4,n_points=100)
    # bayes_cal_plot(best_precal_dcfs,dcfs,mindcfs,eff_lo_priors,model)
    
    
    print(f'Best scores transformations: {best_scores_transformations}')
 

