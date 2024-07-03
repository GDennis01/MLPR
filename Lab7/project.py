import matplotlib.pyplot as plt
import numpy as np
import scipy
from prettytable import PrettyTable
# Compute log-density for a single sample x (column vector). The result is a 1-D array with 1 element
def logpdf_GAU_ND_singleSample(x, mu, C):
    P = np.linalg.inv(C)
    M = x.shape[0]#number of features
    return -0.5*M*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ P @ (x-mu)).ravel()
def logpdf_GAU_ND(x,mu,C):
    P = np.linalg.inv(C)
    M = x.shape[0]#number of features
    return -0.5*M*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu)*( P @ (x-mu))).sum(0)
def loglikelihood(x,mu,C):
    return logpdf_GAU_ND(x,mu,C).sum()

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def load(filename):
    features=[]
    classes=[]
    with open(filename) as f:
        for l in f:
            fields = l.split(",")
            _features=fields[:-1]
            tmp = np.array([float(i) for i in _features])
            tmp = tmp.reshape(tmp.size,1)
            features.append(tmp)
            classes.append(int(fields[-1].strip()))
        classes=np.array(classes)
        return np.hstack(features),classes

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))
def cov_within_classes(dataset,n_label,classes):
    Sw=0
    for val in range(n_label):
        ds_cl = dataset[:,classes==val]
        mu = ds_cl.mean(1)
        ds_cl_cent = ds_cl-vcol(mu)
        Sw+=ds_cl_cent@ds_cl_cent.T
    # special case when there's only 1 dimension
    if dataset.shape[0] == 1:
        Sw = np.array(Sw,ndmin=2)
    return Sw/dataset.shape[1]
def predict_data(DVAL, LVAL,LLR,threshold):
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[LLR >= threshold] = 2
    PVAL[LLR < threshold] = 1

    myarr = PVAL==LVAL
    correct_pred =  myarr.sum()
    print(f'Correct predictions: {correct_pred}')
    false_pred = myarr.size - correct_pred
    print(f'False predictions: {false_pred}')
    print(f'Error rate: {false_pred/myarr.size*100}% over {myarr.size} samples.')
    return false_pred/myarr.size*100

def get_confusion_matrix(predicted,actual):
    n_labels = len(np.unique(actual))
    confusion_matrix = np.zeros((n_labels,n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            confusion_matrix[i,j] = np.sum((predicted==i)&(actual==j))
    return confusion_matrix
def get_missclass_matrix(confusion_matrix):
    return confusion_matrix / vrow(confusion_matrix.sum(0))
def cov_m(D):
    mu = D.mean(1)
    Dc = D - vcol(mu)
    return Dc@Dc.T/D.shape[1]
def mvg(DTR,LTR,DTE,LTE):
    labels = np.unique(LTE)
    SJoint=np.zeros((len(labels), DTE.shape[1]))
    cov_arr=[]
    for i,l in enumerate(labels):
        class_sample_l = DTR[:,LTR==l]
        # cov = np.cov(class_sample_l,bias=True)
        cov = cov_m(class_sample_l)
        cov_arr.append(cov)
        mu = np.mean(class_sample_l,axis=1,keepdims=True)
        SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/2)
    SMarginal =  vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    predicted_labels = np.argmax(SPost,axis=0)
    return SJoint,np.array(cov_arr),predicted_labels

def tied_cov(DTR,LTR,DTE,LTE):
    classes = np.unique(LTE)
    SJoint=np.zeros((len(classes), DTE.shape[1]))
    cov = cov_within_classes(DTR,2,LTR)
    for i,l in enumerate(classes):
        class_sample_l = DTR[:,LTR==l]
        mu = np.mean(class_sample_l,axis=1,keepdims=True)
        SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/2)
    SMarginal =  vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    predicted_labels = np.argmax(SPost,axis=0)
    return SJoint,predicted_labels

def naive_bayes(DTR,LTR,DTE,LTE):
    classes = np.unique(LTE)
    SJoint=np.zeros((len(classes), DTE.shape[1]))
    for i,l in enumerate(classes):
        class_sample_l = DTR[:,LTR==l]
        # cov = np.cov(class_sample_l,bias=True)
        cov = cov_m(class_sample_l)
        cov = np.diag(np.diag(cov))
        mu = np.mean(class_sample_l,axis=1,keepdims=True)
        SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/2)
    SMarginal =  vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    predicted_labels = np.argmax(SPost,axis=0)
    return SJoint,predicted_labels

def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(np.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return np.exp(logPost)

# Compute optimal Bayes decisions for the matrix of class posterior (each column refers to a sample)
def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return np.argmin(expectedCosts, 0)
def uniform_cost_matrix(nClasses):
    return np.ones((nClasses, nClasses)) - np.eye(nClasses)
    
# this way to compute accuracy yields the *same* exact result as the one in predict_data
def get_accuracy(predicted,actual):
    return (predicted==actual).sum()/len(predicted)*100

def compute_optimal_Bayes_binary_threshold(prior, Cfn, Cfp):
    return -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )   

def get_false_negatives(conf_matrix):
    return conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[1,1])
def get_false_positives(conf_matrix):
    return conf_matrix[1,0]/(conf_matrix[0,0]+conf_matrix[1,0])
def get_true_positives(conf_matrix):
    return 1-get_false_negatives(conf_matrix)
def get_dcf(conf_matrix,prior,Cfn,Cfp,normalized=False):
    _dcf = prior*Cfn*get_false_negatives(conf_matrix) + (1-prior)*Cfp*get_false_positives(conf_matrix)
    if normalized:
        return _dcf/min(prior*Cfn,(1-prior)*Cfp)
    return _dcf
def get_dcf_multiclass(confusion_matrix,cost_matrix,priors,normalized=False):
    miss_class_matrix = get_missclass_matrix(confusion_matrix)
    bayes_error = ((miss_class_matrix*cost_matrix).sum(0)*priors.ravel()).sum()
    if normalized:
        return bayes_error/np.min(cost_matrix@vcol(priors))
    return bayes_error

def get_effective_prior(prior,Cfn,Cfp):
    return (prior*Cfn)/(prior*Cfn+((1-prior)*(Cfp)))

def PCA(dataset,n_dim,specif_dim=None):
    mu_ds = dataset.mean(1)
    centered_dataset = dataset - vcol(mu_ds)
    covar_ds = (centered_dataset@centered_dataset.T)/dataset.shape[1]

    U, s, Vh = np.linalg.svd(covar_ds)
    if specif_dim == None:
        P = U[:,0:n_dim]
    else:
        P = U[:,specif_dim:specif_dim+1]

    return P.T

if __name__ == '__main__':
    features,classes=load("trainData.txt")
    (DTR, LTR), (DTE, LTE) = split_db_2to1(features, classes)

    #region Five applications based on effective prior given the triple (Probabilty,Cost false negative, Cost false positive)
    # apps=[[0.5,1.0,1.0],[0.9,1.0,1.0],[0.1,1.0,1.0],[0.5,0.1,0.9],[0.5,0.9,0.1]]
    # priors = [get_effective_prior(app[0],app[1],app[2]) for app in apps]
    # print(priors)
    #endregion

    #region Optimal bayes decisions for MVG, Tied Cov and Naive Bayes without PCA  for the first three applications
    # apps=[[0.5,1.0,1.0],[0.9,1.0,1.0],[0.1,1.0,1.0]]
    # Models = [mvg,tied_cov,naive_bayes]
    # for model in Models:
    #     print(f'Using model {model.__name__}\n')
    #     if model == mvg:
    #         SJoint,_,predicted = model(DTR,LTR,DTE,LTE)
    #     else:
    #         SJoint,predicted = model(DTR,LTR,DTE,LTE)
            
    #     for app in apps:
    #         print(f'Application {app}')
    #         prior_arr = np.array([1-app[0],app[0]])
    #         costMatrix = [[0,app[1]],[app[2],0]]

    #         binary_threshold = compute_optimal_Bayes_binary_threshold(app[0],app[1],app[2])
            
    #         LLR = np.log(SJoint[1]/SJoint[0])
    #         optimal_bayes_predictions = LLR>binary_threshold
            
    #         conf_matrix = get_confusion_matrix(optimal_bayes_predictions,LTE)
    #         dcf_norm = get_dcf(conf_matrix,app[0],app[1],app[2],normalized=True)

    #         thresholds = np.linspace(LLR.min(),LLR.max(),1000)
    #         min_dcf = np.min([get_dcf(get_confusion_matrix(LLR>t,LTE),app[0],app[1],app[2],normalized=True) for t in thresholds])
            
    #         print(f'Conf matrix is \n{conf_matrix}')
    #         print(f'DCF  \n{dcf_norm}')
    #         print(f'Minimum DCF is \n{min_dcf}\n')
            
    
    # #endregion

    # #region Optimal bayes decisions for MVG, Tied Cov and Naive Bayes with PCA  for the first three applications
    my_table = PrettyTable()
    my_table.field_names = ["PCA","Model","Application","Conf Matrix","DCF","Min DCF"]

    apps=[[0.5,1.0,1.0],[0.9,1.0,1.0],[0.1,1.0,1.0]]
    Models = [mvg,tied_cov,naive_bayes]
    dcfs = np.empty((len(Models),6))
    min_dcfs = np.empty((len(Models),6))

    llrs_main_app = {model.__name__: {} for model in Models}
    for M in range(1,features.shape[0]+1):
        PCA_m = PCA(features,M)
        proj_data = PCA_m@features
        (DTR, LTR), (DTE, LTE) = split_db_2to1(proj_data, classes)

        for model in Models:
            if model == mvg:
                SJoint,_,predicted = model(DTR,LTR,DTE,LTE)
            else:
                SJoint,predicted = model(DTR,LTR,DTE,LTE)
                
            for app in apps:
                prior_arr = np.array([1-app[0],app[0]])
                costMatrix = [[0,app[1]],[app[2],0]]
                binary_threshold = compute_optimal_Bayes_binary_threshold(app[0],app[1],app[2])
                
                LLR = np.log(SJoint[1]/SJoint[0])
                if app == apps[2]:
                    llrs_main_app[model.__name__][M] = LLR 
                optimal_bayes_predictions = LLR>binary_threshold
                
                conf_matrix = get_confusion_matrix(optimal_bayes_predictions,LTE)
                dcf_norm = get_dcf(conf_matrix,app[0],app[1],app[2],normalized=True)
                if app == apps[2]:
                    dcfs[Models.index(model),M-1] = dcf_norm


                thresholds = np.linspace(LLR.min(),LLR.max(),1000)
                min_dcf = np.min([get_dcf(get_confusion_matrix(LLR>t,LTE),app[0],app[1],app[2],normalized=True) for t in thresholds])
                if app == apps[2]:
                    min_dcfs[Models.index(model),M-1] = min_dcf
                my_table.add_row([M,model.__name__,app,conf_matrix,dcf_norm,min_dcf])
    best_PCA = np.argmin(min_dcfs.sum(0))
    print(f'Best PCA is {best_PCA+1}')
    print(my_table)
    for model in Models:
        pyplot = plt.figure()   
        plt.plot(range(1,features.shape[0]+1),dcfs[Models.index(model),:],label=f'{model.__name__}')
        plt.plot(range(1,features.shape[0]+1),min_dcfs[Models.index(model),:],label=f'{model.__name__} MIN')
        plt.xlabel('M')
        plt.xticks(range(1,features.shape[0]+1))
        plt.ylabel('DCF')
        plt.legend()
        plt.show()
    plt.plot(range(1,features.shape[0]+1),dcfs[0,:],label='MVG')
    plt.plot(range(1,features.shape[0]+1),dcfs[1,:],label='Tied Cov')
    plt.plot(range(1,features.shape[0]+1),dcfs[2,:],label='Naive Bayes')

    plt.plot(range(1,features.shape[0]+1),min_dcfs[0,:],label='MVG MIN')
    plt.plot(range(1,features.shape[0]+1),min_dcfs[1,:],label='Tied Cov MIN')
    plt.plot(range(1,features.shape[0]+1),min_dcfs[2,:],label='Naive Bayes MIN')
    plt.xlabel('M')
    plt.xticks(range(1,features.shape[0]+1))
    plt.ylabel('DCF')
    plt.legend()
    plt.show()
    #endregion

    # #region Bayes error plots with best PCA setup for the pi=0.1 application
    (DTR, LTR), (DTE, LTE) = split_db_2to1(features, classes)
    PCA_m = PCA(features,best_PCA+2)
    proj_data = PCA_m@features
    (DTR, LTR), (DTE, LTE) = split_db_2to1(proj_data, classes)
    llrs = [llrs_main_app[model.__name__][best_PCA] for model in Models]
    eff_prior_log_odds = np.linspace(-4,4,100)
    priors = [1 / (1 + np.exp(-x)) for x in eff_prior_log_odds]
    pyplot = plt.figure()

    dcfs = np.empty((len(Models),100))
    min_dcfs = np.empty((len(Models),100))
    for model in Models:
        LLR = llrs[Models.index(model)]
        sorted_llrs = np.sort(LLR)
        thresholds = np.linspace(sorted_llrs[0],sorted_llrs[-1],100)
        for prior in priors:
            threshold = compute_optimal_Bayes_binary_threshold(prior,1,1)

            min_dcf = [get_dcf(get_confusion_matrix(LLR>t,LTE),prior,1.0,1.0,normalized=True) for t in thresholds]
            min_dcf = np.min(min_dcf)
            min_dcfs[Models.index(model),priors.index(prior)] = min_dcf

            dcf = get_dcf(get_confusion_matrix(LLR>threshold,LTE),prior,1.0,1.0,normalized=True)
            dcfs[Models.index(model),priors.index(prior)] = dcf

    for model in Models:
        dcf = dcfs[Models.index(model),:]
        min_dcf = min_dcfs[Models.index(model),:]
        plt.plot(eff_prior_log_odds,dcf,label=f'{model.__name__} DCF')
        plt.plot(eff_prior_log_odds,min_dcf,label=f'{model.__name__} MIN DCF') 
        plt.xlabel('Effective Prior Log Odds')
        plt.ylabel('(MIN) DCF')  
        plt.legend()
        plt.show()
        
    #endregion