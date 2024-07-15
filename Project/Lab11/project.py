from Project.libs.svm import *
from Project.libs.logistic_regression import LogRegClassifier
from Project.libs.gmm import *
from Project.libs.utils import load,split_db_2to1
from Project.libs.bayes_risk import *
import numpy as np
import json

TARGET_PRIOR = 0.1
KFOLD = 5
BEST_SETUP_SVM = np.array([])
BEST_SETUP_GMM = np.array([])
BEST_SETUP_LOGREG = np.array([])
BEST_SCORE_TRANSFORMATIONS = {
        'gmm':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None},
        'svm':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None},
        'logreg':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None},
        'fused':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None}
        }

def extract_train_val_folds_from_ary(X, idx):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]
def calibrate_kfold(model,LTE,prior_cal,fusion=False):
    """
    Perform a kfold calibration for a given model and a given prior(application)
    Returns:
    lr_scores: The log likelihood ratio scores for each fold
    labels: The labels for each fold
    """
    global BEST_SETUP_SVM,BEST_SETUP_GMM,BEST_SETUP_LOGREG
    if fusion == False:
        scores = eval("BEST_SETUP_"+model.upper())['scores']
    else:
        scores_svm = BEST_SETUP_SVM['scores']
        scores_gmm = BEST_SETUP_GMM['scores']
        scores_logreg = BEST_SETUP_LOGREG['scores']
    lr_scores = []
    labels = []
    for idx in range(KFOLD):
        if fusion == False:
            SCAL, SVAL = extract_train_val_folds_from_ary(scores, idx)
            LCAL, LVAL = extract_train_val_folds_from_ary(LTE, idx)
            SCAL = vrow(SCAL)
            SVAL = vrow(SVAL)
        else:
            SCAL_GMM, SVAL_GMM = extract_train_val_folds_from_ary(scores_gmm, idx)
            SCAL_SVM, SVAL_SVM = extract_train_val_folds_from_ary(scores_svm, idx)
            SCAL_LOGREG, SVAL_LOGREG = extract_train_val_folds_from_ary(scores_logreg, idx)
            LCAL, LVAL = extract_train_val_folds_from_ary(LTE, idx)
            SCAL = np.vstack([SCAL_GMM,SCAL_SVM,SCAL_LOGREG])
            SVAL = np.vstack([SVAL_GMM,SVAL_SVM,SVAL_LOGREG])
        lrc = LogRegClassifier.with_details(SCAL, LCAL, SVAL, LVAL)
        w,b = lrc.train('weighted',l=0,pT=prior_cal)

        lr_score = lrc.llr_scores
        lr_scores.append(lr_score)
        labels.append(LVAL)
    return lr_scores, labels
def set_best_scores_transformations(LTE,priors,model,fusion=False):
    """
    Compute and set best score transformation for a given model and a given prior(application).
    Best scores are set in the global variable BEST_SCORE_TRANSFORMATIONS
    """
    global BEST_SCORE_TRANSFORMATIONS
    best_act_dcf = np.inf 
    for prior in priors:
        lr_scores,calibrated_labels = calibrate_kfold(model,LTE,prior,fusion)

        calibrated_scores = np.hstack(lr_scores)
        calibrated_labels = np.hstack(calibrated_labels)

        #computing act dcf and min dcf, then choosing the best score transformation for each model based on act dcf
        calibrated_adcf = get_dcf(calibrated_scores,calibrated_labels,TARGET_PRIOR,normalized=True,threshold='optimal')
        calibrated_mdcf = get_min_dcf(calibrated_scores,calibrated_labels,TARGET_PRIOR)
        if calibrated_adcf < best_act_dcf:
            best_act_dcf = calibrated_adcf
            BEST_SCORE_TRANSFORMATIONS[model]['scores'] = calibrated_scores
            BEST_SCORE_TRANSFORMATIONS[model]['labels'] = calibrated_labels
            BEST_SCORE_TRANSFORMATIONS[model]['actdcf'] = calibrated_adcf
            BEST_SCORE_TRANSFORMATIONS[model]['mindcf'] = calibrated_mdcf
            BEST_SCORE_TRANSFORMATIONS[model]['prior'] = prior
    print(f'Best act dcf for {model}: {BEST_SCORE_TRANSFORMATIONS[model]["actdcf"]} with min dcf: {BEST_SCORE_TRANSFORMATIONS[model]["mindcf"]} and training-prior: {BEST_SCORE_TRANSFORMATIONS[model]["prior"]} and target-prior: {TARGET_PRIOR}')
def bayes_cal_plot(cal_dcfs,dcfs,min_dcfs,eff_prior_log_odds,model):
    if not model == 'fused':
        plt.plot(eff_prior_log_odds,cal_dcfs,label=f'{model} DCF(pre-calibration)')
    plt.plot(eff_prior_log_odds,dcfs,label=f'{model} DCF(post-calibration)')
    plt.plot(eff_prior_log_odds,min_dcfs,label=f'{model} MIN DCF',linestyle='--') 
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('(MIN) DCF')  
    plt.legend()
    plt.show()
    # return best_act_dcf,best_min_dcf,best_prior,best_calscores,best_labels

def Lab11():
    global BEST_SETUP_SVM,BEST_SETUP_GMM,BEST_SETUP_LOGREG,BEST_SCORE_TRANSFORMATIONS
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
    #CALIBRATION

    priors = np.linspace(0.1,0.9,9)

    set_best_scores_transformations(LTE,priors,'gmm')
    set_best_scores_transformations(LTE,priors,'svm')
    set_best_scores_transformations(LTE,priors,'logreg')
    set_best_scores_transformations(LTE,priors,'fused',fusion=True)


    fused_scores = np.hstack(BEST_SCORE_TRANSFORMATIONS['fused']['scores'])
    fused_labels = np.hstack(BEST_SCORE_TRANSFORMATIONS['fused']['labels'])
    fused_adcf = get_dcf(fused_scores,fused_labels,TARGET_PRIOR,normalized=True,threshold='optimal')
    fused_mdcf = get_min_dcf(fused_scores,fused_labels,TARGET_PRIOR)
    eff_lo_priors,dcfs,mindcfs = bayes_error_plot(fused_scores,fused_labels,left=-4,right=4,n_points=10)
    bayes_cal_plot(None,dcfs,mindcfs,eff_lo_priors,'fused')

        
    for model in ['svm','gmm','logreg']:
        cur_scores_precal = eval("BEST_SETUP_"+model.upper())['scores']
        cur_scores_postcal = BEST_SCORE_TRANSFORMATIONS[model]['scores']
        cur_labels = BEST_SCORE_TRANSFORMATIONS[model]['labels']

        _,best_precal_dcfs,_ = bayes_error_plot(cur_scores_precal,LTE,left=-4,right=4,n_points=10)
        eff_lo_priors,dcfs,mindcfs = bayes_error_plot(cur_scores_postcal,cur_labels,left=-4,right=4,n_points=10)

        bayes_cal_plot(best_precal_dcfs,dcfs,mindcfs,eff_lo_priors,model)
    best_system = min(BEST_SCORE_TRANSFORMATIONS,key=lambda x:BEST_SCORE_TRANSFORMATIONS[x]['actdcf'])
    best_system_name = best_system
    best_system = BEST_SCORE_TRANSFORMATIONS[best_system]
    best_system['name'] = best_system_name
    
    # EVALUATION
    (features,classes) = load("project/data/evalData.txt")
    (_,_), (DTE, LTE) = split_db_2to1(features, classes)
    scores = best_system['scores']
    min_dcf = get_min_dcf(scores,LTE,best_system['prior'])
    act_dcf = get_dcf(scores,LTE,best_system['prior'],normalized=True,threshold='optimal')
    print(f'Best system: {best_system} with min dcf: {min_dcf} and act dcf: {act_dcf}')
    # print(f'Best scores transformations: {BEST_SCORE_TRANSFORMATIONS}')
 

