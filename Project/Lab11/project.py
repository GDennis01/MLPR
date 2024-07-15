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
        'gmm':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None,'w':None,'b':None},
        'svm':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None,'w':None,'b':None},
        'logreg':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None,'w':None,'b':None},
        'fused':{'name':None,'prior':None,'actdcf':None,'mindcf':None,'scores':None,'labels':None,'w':None,'b':None}
        }

def extract_train_val_folds_from_ary(X, idx):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]
def calibrate_kfold(model,LTE,prior_cal,fusion=False,evaluation=False,**kwargs):
    """
    Perform a kfold calibration for a given model and a given prior(application)
    Returns:
    lr_scores: The log likelihood ratio scores for each fold
    labels: The labels for each fold
    """
    global BEST_SETUP_SVM,BEST_SETUP_GMM,BEST_SETUP_LOGREG
    if evaluation == False:
        if fusion == False:
            scores = eval("BEST_SETUP_"+model.upper())['scores']
        else:
            scores_svm = BEST_SETUP_SVM['scores']
            scores_gmm = BEST_SETUP_GMM['scores']
            scores_logreg = BEST_SETUP_LOGREG['scores']
    else:
        if fusion == False:
            scores = kwargs.get('scores')
        else:
            scores_svm = kwargs.get('scores_svm')
            scores_gmm = kwargs.get('scores_gmm')
            scores_logreg = kwargs.get('scores_logreg')

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
    #perform a final training on the whole training set
    if fusion == False:
        SCAL = vrow(scores)
    else:
        SCAL = np.vstack([scores_gmm,scores_svm,scores_logreg])    
    LCAL = LTE

    lrc = LogRegClassifier.with_details(SCAL, LCAL, SCAL, LCAL)
    w,b = lrc.train('weighted',l=0,pT=prior_cal)

    return lr_scores, labels,w,b,lrc.llr_scores
def set_best_scores_transformations(LTE,priors,model,fusion=False):
    """
    Compute and set best score transformation for a given model and a given prior(application).
    Best scores are set in the global variable BEST_SCORE_TRANSFORMATIONS
    """
    global BEST_SCORE_TRANSFORMATIONS
    best_act_dcf = np.inf 
    for prior in priors:
        lr_scores,calibrated_labels,w,b,final_scores = calibrate_kfold(model,LTE,prior,fusion)

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
            BEST_SCORE_TRANSFORMATIONS[model]['w'] = w
            BEST_SCORE_TRANSFORMATIONS[model]['b'] = b
    print(f'Best act dcf for {model}: {BEST_SCORE_TRANSFORMATIONS[model]["actdcf"]} with min dcf: {BEST_SCORE_TRANSFORMATIONS[model]["mindcf"]} and training-prior: {BEST_SCORE_TRANSFORMATIONS[model]["prior"]} and target-prior: {TARGET_PRIOR}')
def bayes_cal_plot(cal_dcfs,dcfs,min_dcfs,eff_prior_log_odds,model,cal=True):
    label_pre = '(pre-calibration)' if cal else ''
    label_post = '(post-calibration)' if cal else ''
    if cal and not model == 'fused':
        plt.plot(eff_prior_log_odds,cal_dcfs,label=f'{model} DCF{label_pre}')
    plt.plot(eff_prior_log_odds,dcfs,label=f'{model} DCF{label_post}')
    if min_dcfs is not None:
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

    #computing score transformation for each of the models
    for model in ['svm','gmm','logreg','fused']:
        set_best_scores_transformations(LTE,priors,model,fusion=model=='fused')

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
    best_system = min(filter(lambda x: x!='fused',BEST_SCORE_TRANSFORMATIONS),key=lambda x:BEST_SCORE_TRANSFORMATIONS[x]['actdcf'])
    best_system_name = best_system
    best_system = BEST_SCORE_TRANSFORMATIONS[best_system]
    best_system['name'] = best_system_name
    with open('Project/best_setups/best_setup.json','w') as f:
        # best_system['scores'] = best_system['scores'].tolist()
        # best_system['labels'] = best_system['labels'].tolist()
        # best_system['w'] = best_system['w'].tolist()
        obj = {'name':best_system['name'],'prior':best_system['prior']}
        json.dump(obj,f)
def compute_scores(model,setup,features):
    match model:
        case 'logreg':
            if setup['expanded_feature']:
                features = LogRegClassifier.__quadratic_feature_expansion__(features)
            w,b = setup['w'],setup['b']
            scores = w.T@features+b-np.log(TARGET_PRIOR/(1-TARGET_PRIOR))
        case 'svm':
            w,b = setup['w'],setup['b']
            scores = w.T@features+b
        case 'gmm':
            gmm0 = setup['gmm0']
            gmm1 = setup['gmm1']
            gmm0 = [(w,np.array(m),np.array(C)) for w,m,C in gmm0]
            gmm1 = [(w,np.array(m),np.array(C)) for w,m,C in gmm1]
            scores = gmm_scores(gmm0,gmm1,features)
    return scores
    
def Lab11_eval():
    # EVALUATION
    #data setup
    with open('Project/best_setups/best_setup.json') as f:
        best_system = json.load(f)
    with open('Project/best_setups/best_setup_logreg.json') as f:
        best_setup_logreg_eval = json.load(f)
        w,b = np.array(best_setup_logreg_eval['w']),np.array(best_setup_logreg_eval['b'])
        best_setup_logreg_eval['w'] = w
        best_setup_logreg_eval['b'] = b
    with open('Project/best_setups/best_setup_svm.json') as f:
        best_setup_svm_eval = json.load(f)
        w,b = np.array(best_setup_svm_eval['w']),np.array(best_setup_svm_eval['b'])
        best_setup_svm_eval['w'] = w
        best_setup_svm_eval['b'] = b
    with open('Project/best_setups/best_setup_gmm.json') as f:
        best_setup_gmm_eval = json.load(f)
        gmm0 = best_setup_gmm_eval['gmm0']
        gmm1 = best_setup_gmm_eval['gmm1']
        gmm0 = [(w,np.array(m),np.array(C)) for w,m,C in gmm0]
        gmm1 = [(w,np.array(m),np.array(C)) for w,m,C in gmm1]
        best_setup_gmm_eval['gmm0'] = gmm0
        best_setup_gmm_eval['gmm1'] = gmm1
    with open('Project/best_setups/quadratic_weighted_logreg.json') as f:
        qwl = json.load(f)
        # print(qwl)
        for elem in qwl:
            # print(elem)
            elem['w'] = np.array(elem['w'])


    (features,classes) = load("project/data/evalData.txt")
    (_,_), (DTE, LTE) = split_db_2to1(features, classes)

    best_setups = {'logreg':best_setup_logreg_eval,'svm':best_setup_svm_eval,'gmm':best_setup_gmm_eval} 
    fused_scores =[]
    fused_labels = []

    #bayes error plot for delivered system
    best_min_dcf_bestsystem = np.inf
    #bayes error plot for all systems
    for model in ['logreg','svm','gmm']:
        scores = compute_scores(model,best_setups[model],features)
        fused_scores.append(scores)
        fused_labels.append(classes)
        #calibrating scores
        cal_scores,cal_labels,_,_,_ = calibrate_kfold(model,classes,best_system['prior'],fusion=False,evaluation=True,scores=scores)
        cal_scores = np.hstack(cal_scores)
        cal_labels = np.hstack(cal_labels)

        eff_prior_log_odds,dcfs,mindcfs = bayes_error_plot(cal_scores,cal_labels,left=-4,right=4,n_points=10)
        #showing only actdcf in bayes error plot
        bayes_cal_plot(None,dcfs,None,eff_prior_log_odds,model,cal=False)
        #showing also min dcf in bayes error plot
        bayes_cal_plot(None,dcfs,mindcfs,eff_prior_log_odds,model,cal=False)
        #dcf and min dcf for each model
        act_dcf_cal = get_dcf(cal_scores,cal_labels,TARGET_PRIOR,1.0,1.0,normalized=True,threshold='optimal')
        min_dcf_cal = get_min_dcf(cal_scores,cal_labels,TARGET_PRIOR)
        if model == best_system['name']:
            best_min_dcf_bestsystem = min_dcf_cal
        print(f'{model}: Calibrated MinDCF: {min_dcf_cal} for model {model} ')
        print(f'{model}: Calibrated Actual DCF: {act_dcf_cal} for model {model}')
    
    #retraining the quadratic weighted logreg model to see how it fares against the best system in terms of mindcf
    for elem in qwl:
        w = elem['w']
        b = elem['b']
        l = elem['lambda']
        expanded_features = LogRegClassifier.__quadratic_feature_expansion__(features)
        scores = w.T@expanded_features+b - np.log(TARGET_PRIOR/(1-TARGET_PRIOR))
        min = get_min_dcf(scores,classes,TARGET_PRIOR)
        print(f'Quadratic Weighted LogReg: MinDCF: {min} for lambda: {l}')
    print(f'Best system: {best_system["name"]} with MinDCF: {best_min_dcf_bestsystem}')



 

