from Project.libs.svm import *
from Project.libs.logistic_regression import LogRegClassifier
from Project.libs.gmm import *
from Project.libs.utils import load,split_db_2to1
from Project.libs.bayes_risk import *
import numpy as np

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

def Lab11():
    (features,classes) = load("project/data/trainData.txt")
    _, (DTE, LTE) = split_db_2to1(features, classes)

    #region SVM
    prior_cal = 0.5

    # svm = SVM(features,classes)
    # fscore = svm.train(10,'kernel',kernelFunc=rbfKernel(np.exp(-2)))
    # scores = fscore(svm.DTE)

    ## to avoid retraining
    # np.save("project/data/scores.npy",scores)
    scores = np.load("project/data/scores.npy")

    dcf = get_dcf(scores,LTE,prior_cal,normalized=True,threshold='optimal')
    minDCF = get_min_dcf(scores,LTE,prior_cal)

    lr_scores = []
    labels = []
    for idx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(scores, idx)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, idx)

        lrc = LogRegClassifier.with_details(vrow(SCAL), LCAL, vrow(SVAL), LVAL)
        w,b = lrc.train('weighted',pT=prior_cal)

        lr_score = lrc.llr_scores
        lr_scores.append(lr_score)
        labels.append(LVAL)

    calibrated_scores = np.hstack(lr_scores)
    labels = np.hstack(labels)
    dcf = get_dcf(calibrated_scores,labels,prior_cal,normalized=True,threshold='optimal')
    minDCF = get_min_dcf(calibrated_scores,labels,prior_cal)
    print(f"After calibration: DCF: {dcf}, minDCF: {minDCF}")

    #Training on whole calibration set
    lrc = LogRegClassifier.with_details(vrow(scores), labels, vrow(calibrated_scores), [])
    w,b = lrc.train('weighted',pT=prior_cal)

    calibrated_scores = w.T@vrow(scores) + b- np.log(prior_cal/(1-prior_cal))
    dcf = get_dcf(calibrated_scores,LTE,prior_cal,normalized=True,threshold='optimal')
    minDCF = get_min_dcf(calibrated_scores,LTE,prior_cal)
    print(f"After calibration: DCF: {dcf}, minDCF: {minDCF}")

    #endregion
    pass