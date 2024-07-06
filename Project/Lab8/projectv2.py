import matplotlib.pyplot as plt
import numpy as np
import scipy
from prettytable import PrettyTable
from Project.libs.utils import load,vcol,vrow,split_db_2to1
from Project.libs.bayes_risk import compute_optimal_Bayes_binary_threshold,get_dcf,get_min_dcf,get_confusion_matrix
from Project.libs.logistic_regression import LogRegClassifier

# compute both dcf and min dcf for a given logreg model. it's mainly an utility function lest we repeat code
# prior is needed only for the weighted model
def get_dcf_mindcf_logreg(D,L,lambdas,prior,model="binary",one_fiftieth=False,expaded_feature=False,center_data=False):
    dcfs = []
    min_dcfs = []
    for l in lambdas:
        print(f'Lambda: {l}')
        lrc = LogRegClassifier(D,L,one_fiftieth=one_fiftieth)

        if center_data:
            lrc.DTR = lrc.DTR - lrc.DTR.mean(1,keepdims=True)
            lrc.DTE = lrc.DTE - lrc.DTR.mean(1,keepdims=True)
        lrc.train(model,l=l,expaded_feature=expaded_feature,pT=prior)

        empirical_prior = (lrc.LTR == 1).sum()/lrc.LTR.size

        if model == "binary":
            scores_llr = lrc.logreg_scores - np.log(empirical_prior/(1-empirical_prior))
        elif model == "weighted":
            scores_llr = lrc.logreg_scores - np.log(prior/(1-prior))
        else:
            scores_llr = lrc.logreg_scores - np.log(empirical_prior/(1-empirical_prior))

        dcf = get_dcf(scores_llr,lrc.LTE,prior,1,1,normalized=True,threshold='optimal')
        min_dcf = get_min_dcf(scores_llr,lrc.LTE,prior,1,1)
        print(f'DCF: {dcf}')
        print(f'Min DCF: {min_dcf}\n')

        dcfs.append(dcf)
        min_dcfs.append(min_dcf)
    return dcfs,min_dcfs
    
def plot_dcf_vs_lambda(lambdas,dcfs,min_dcfs):
    plt.figure()
    plt.plot(lambdas,dcfs,label='DCFs')
    plt.plot(lambdas,min_dcfs,label='Min DCFs')
    plt.xscale('log',base=10)
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs Lambda')
    plt.legend()
    plt.show()
    
def Lab8():
    features,classes=load("Project/data/trainData.txt")

    #region DCF for different lambdas
    # lambdas = np.logspace(-4,2,13)
    # prior = 0.1
    # dcfs,min_dcfs = get_dcf_mindcf_logreg(features,classes,lambdas,prior,"binary")
    # plot_dcf_vs_lambda(lambdas,dcfs,min_dcfs)
    #endregion
    
    #region DCF for different lambdas with 1 sample left out
    # prior = 0.1
    # lambdas = np.logspace(-4,2,13)   
    # dcfs,min_dcfs = get_dcf_mindcf_logreg(features,classes,lambdas,prior,"binary",one_fiftieth=True)
    # plot_dcf_vs_lambda(lambdas,dcfs,min_dcfs)
    #endregion

    # region DCF for different lambdas with Weighted Log-reg
    # lambdas = np.logspace(-4,2,13)
    # prior = 0.1
    # dcfs,min_dcfs = get_dcf_mindcf_logreg(features,classes,lambdas,prior,"weighted")
    # plot_dcf_vs_lambda(lambdas,dcfs,min_dcfs)
    #endregion

    #region Quadratic Logistic Regression 
    lambdas = np.logspace(-4,2,13)
    prior = 0.1

    dcfs,min_dcfs = get_dcf_mindcf_logreg(features,classes,lambdas,prior,"binary",expaded_feature=True)
    plot_dcf_vs_lambda(lambdas,dcfs,min_dcfs)
    #endregion

    #region Centering data to see effects of regularization term(lambda)
    lambdas = np.logspace(-4,2,13)
    prior = 0.1

    dcfs,min_dcfs = get_dcf_mindcf_logreg(features,classes,lambdas,prior,"binary",center_data=True)
    plot_dcf_vs_lambda(lambdas,dcfs,min_dcfs)
    #endregion