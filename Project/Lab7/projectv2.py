import matplotlib.pyplot as plt
import numpy as np
import scipy
from prettytable import PrettyTable
from Project.libs.utils import load,vcol,vrow,cov_between_classes,cov_within_classes,split_db_2to1,cov_m
from Project.libs.dim_reduction import PCA
from Project.libs.bayes_risk import get_effective_prior,get_dcf,get_confusion_matrix,get_min_dcf
from Project.libs.gaussian_classifier import GaussianClassifier


def Lab7():
    features,classes=load("Project/data/trainData.txt")

    #region Five applications based on effective prior given the triple (Probabilty,Cost false negative, Cost false positive)
    apps=[[0.5,1.0,1.0],[0.9,1.0,1.0],[0.1,1.0,1.0],[0.5,0.1,0.9],[0.5,0.9,0.1]]
    priors = [get_effective_prior(app[0],app[1],app[2]) for app in apps]
    print(priors)
    #endregion

    #region Optimal bayes decisions for MVG, Tied Cov and Naive Bayes without PCA  for the first three applications
    apps=[[0.5,1.0,1.0],[0.9,1.0,1.0],[0.1,1.0,1.0]]
    gc = GaussianClassifier(features,classes)
    Models = ["mvg","tied","naive"]
    for model in Models:
        print(f'Using model {model}')
        gc.train(model,disable_print=True)
        for app in apps:
            print(f'Application {app}')
            
            optimal_bayes_predictions,_= gc.evaluate(threshold='optimal',prior=app[0],Cfn=app[1],Cfp=app[2],disable_print=True)
            conf_matrix = get_confusion_matrix(optimal_bayes_predictions,gc.LTE)
  
            dcf_norm = get_dcf(gc.LLR,gc.LTE,app[0],app[1],app[2],normalized=True,threshold='optimal')
            min_dcf = get_min_dcf(gc.LLR,gc.LTE,app[0],app[1],app[2])
            
            # print(f'Conf matrix is \n{conf_matrix}')
            print(f'DCF  \n{dcf_norm}')
            print(f'Minimum DCF is \n{min_dcf}\n')
    # #endregion

    # #region Optimal bayes decisions for MVG, Tied Cov and Naive Bayes with PCA  for the first three applications
    my_table = PrettyTable()
    my_table.field_names = ["PCA","Model","Application","Conf Matrix","DCF","Min DCF"]

    apps=[[0.5,1.0,1.0],[0.9,1.0,1.0],[0.1,1.0,1.0]]
    Models = ["mvg","tied","naive"]
    dcfs = np.empty((len(Models),6))
    min_dcfs = np.empty((len(Models),6))

    d = np.inf

    llrs_main_app = {model: {} for model in Models}
    for M in range(1,features.shape[0]+1):
        proj_data = PCA(features,M)@features
        gc = GaussianClassifier(proj_data,classes)

        for model in Models:
            gc.train(model,disable_print=True)
            for app in apps:
                optimal_bayes_predictions,_= gc.evaluate(threshold='optimal',prior=app[0],Cfn=app[1],Cfp=app[2],disable_print=True)
                
                conf_matrix = get_confusion_matrix(optimal_bayes_predictions,gc.LTE)

                dcf_norm = get_dcf(gc.LLR,gc.LTE,app[0],app[1],app[2],normalized=True,threshold='optimal')
                min_dcf = get_min_dcf(gc.LLR,gc.LTE,app[0],app[1],app[2])


                # updating the best PCA setup
                if app == apps[2] and min_dcf < d:
                    d = min_dcf
                    best_pca_setup = (model,M)

                # for main application only
                if app == apps[2]:
                    llrs_main_app[model][M] = gc.LLR 
                    dcfs[Models.index(model),M-1] = dcf_norm
                    min_dcfs[Models.index(model),M-1] = min_dcf

                my_table.add_row([M,model,app,conf_matrix,dcf_norm,min_dcf])

    print(f'Best setup is {best_pca_setup}')
    print("@@@")
    print(my_table)
    for model in Models:
        pyplot = plt.figure()   
        plt.plot(range(1,features.shape[0]+1),dcfs[Models.index(model),:],label=f'DCF {model}')
        plt.plot(range(1,features.shape[0]+1),min_dcfs[Models.index(model),:],label=f'MinDCF {model}')
        plt.xlabel('M')
        plt.xticks(range(1,features.shape[0]+1))
        plt.ylabel('DCF')
        plt.legend()
        plt.show()
    #endregion

    # #region Bayes error plots with best PCA setup for the pi=0.1 application
    # Use the best pca dimension obtained with each model
    (DTR, LTR), (DTE, LTE) = split_db_2to1(features, classes)
    proj_data = PCA(features,best_pca_setup[1])@features
    (DTR, LTR), (DTE, LTE) = split_db_2to1(proj_data, classes)

    llrs = [llrs_main_app[model][best_pca_setup[1]] for model in Models]

    eff_prior_log_odds = np.linspace(-4,4,100)
    priors = [1 / (1 + np.exp(-x)) for x in eff_prior_log_odds]

    pyplot = plt.figure()

    dcfs = np.empty((len(Models),100))
    min_dcfs = np.empty((len(Models),100))
    for model in Models:
        LLR = llrs[Models.index(model)]
        for prior in priors:
            dcf = get_dcf(LLR,LTE,prior,1.0,1.0,normalized=True,threshold='optimal')
            min_dcf = get_min_dcf(LLR,LTE,prior,1.0,1.0)

            dcfs[Models.index(model),priors.index(prior)] = dcf
            min_dcfs[Models.index(model),priors.index(prior)] = min_dcf

    for model in Models:
        dcf = dcfs[Models.index(model),:]
        min_dcf = min_dcfs[Models.index(model),:]
        plt.plot(eff_prior_log_odds,dcf,label=f'{model} DCF')
        plt.plot(eff_prior_log_odds,min_dcf,label=f'{model} MIN DCF') 
        plt.xlabel('Effective Prior Log Odds')
        plt.ylabel('(MIN) DCF')  
        plt.legend()
        plt.show()
    #endregion