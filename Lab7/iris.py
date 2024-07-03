import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import scipy
import sklearn.metrics as skmetrics


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

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L
    

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))
def cov_within_classes(dataset,n_label,classes):
    Sw=0
    for val in range(n_label):
        ds_cl = dataset[:,classes==val]
        swc=np.cov(ds_cl,bias=True)
        Sw+= swc*ds_cl.shape[1]
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

def mvg(DTR,LTR,DTE,LTE):
    labels = np.unique(LTE)
    SJoint=np.zeros((len(labels), DTE.shape[1]))
    cov_arr=[]
    for i,l in enumerate(labels):
        class_sample_l = DTR[:,LTR==l]
        cov = np.cov(class_sample_l,bias=True)
        cov_arr.append(cov)
        mu = np.mean(class_sample_l,axis=1,keepdims=True)
        SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/3)
    SMarginal =  vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    predicted_labels = np.argmax(SPost,axis=0)
    return SJoint,np.array(cov_arr),predicted_labels

def tied_cov(DTR,LTR,DTE,LTE):
    classes = np.unique(LTE)
    SJoint=np.zeros((len(classes), DTE.shape[1]))
    cov = cov_within_classes(DTR,3,LTR)
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
        cov = np.cov(class_sample_l,bias=True)
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
if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    commedia_labels = np.load('data/commedia_labels.npy')
    commedia_labels_eps1 = np.load('data/commedia_labels_eps1.npy')
    commedia_ll = np.load('data/commedia_ll.npy')
    commedia_ll_eps1 = np.load('data/commedia_ll_eps1.npy')

    commedia_labels_infpar = np.load('data/commedia_labels_infpar.npy')
    commedia_labels_infpar_eps1 = np.load('data/commedia_labels_infpar_eps1.npy')

    commedia_llr_infpar = np.load('data/commedia_llr_infpar.npy')
    commedia_llr_infpar_eps1 = np.load('data/commedia_llr_infpar_eps1.npy')
    commedia_llr_infpar_sorted = np.sort(commedia_llr_infpar)
    commedia_llr_infpar_eps1_sorted = np.sort(commedia_llr_infpar_eps1)

    #region MVG Confusion Matrix
    # print("MVG:")
    # SJoint,_,p_labels = mvg(DTR,LTR,DTE,LTE)
    # c_matrix = get_confusion_matrix(p_labels,LTE)
    # print(c_matrix)
    #endregion

    #region Tied Cov Confusion Matrix
    # print("Tied Cov:")
    # SJoint,p_labels = tied_cov(DTR,LTR,DTE,LTE)
    # c_matrix = get_confusion_matrix(p_labels,LTE)
    # print(c_matrix)
    #endregion

    #region Commedia Confusion Matrix
    # commedia_ll = np.load('commedia_ll.npy')
    # commedia_labels = np.load('commedia_labels.npy')

    # commedia_posteriors = compute_posteriors(commedia_ll, np.ones(3)/3.0)
    # commedia_predictions = compute_optimal_Bayes(commedia_posteriors, uniform_cost_matrix(3))

    # c_matrix = get_confusion_matrix(commedia_predictions, commedia_labels)
    # print(c_matrix)
    #endregion

    #region Binary Task
    # params = [(0.5,1,1),(0.8,1,1),(0.5,10,1),(0.8,1,10)]
    # for param in params:
    #     # Optimal Decision
    #     t = compute_optimal_Bayes_binary_threshold(param[0],param[1],param[2])
    #     predictions = commedia_llr_infpar > t
    #     c_matrix = get_confusion_matrix(predictions,commedia_labels_infpar)
    #     print(f'Cost Matrix: {param} ')
    #     print(f'{c_matrix}\n')

    #     # Evaluation
    #     dcf = get_dcf(c_matrix,param[0],param[1],param[2])
    #     print(f'DCF: {dcf}')
    #     print(f'Normalized DCF: {get_dcf(c_matrix,param[0],param[1],param[2],True)}\n')
    #     print(f'Cost Matrix: {param}\n: {skmetrics.confusion_matrix(commedia_labels_infpar,predictions)}')

    #     # Min DCF
    #     thresholds = np.linspace(commedia_llr_infpar_sorted[0],commedia_llr_infpar_sorted[-1],2000)
    #     min_dcf = np.min([get_dcf(get_confusion_matrix(commedia_llr_infpar > t,commedia_labels_infpar),param[0],param[1],param[2],True) for t in thresholds])
    #     print(f'Min DCF: {min_dcf}\n')
    #endregion

    #region ROC Curves
    # thresholds = np.linspace(commedia_llr_infpar_sorted[0],commedia_llr_infpar_sorted[-1],2000)
    # fps_rates = [get_false_positives(get_confusion_matrix(commedia_llr_infpar > t,commedia_labels_infpar)) for t in thresholds]
    # tps_rates = [get_true_positives(get_confusion_matrix(commedia_llr_infpar > t,commedia_labels_infpar)) for t in thresholds]
    # plt.plot(fps_rates,tps_rates)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()
    #endregion

    #region Bayes error plots
    # eff_prior_logs_odds = np.linspace(-3,3,21)
    # priors =[1/(1+np.exp(-prior_log_odds)) for prior_log_odds in eff_prior_logs_odds]
    # dcfs = []
    # min_dcfs = []
    # thresholds = np.linspace(commedia_llr_infpar_sorted[0],commedia_llr_infpar_sorted[-1],2000)
    # for prior in priors:
    #     threshold = compute_optimal_Bayes_binary_threshold(prior,1,1)
    #     dcfs.append(get_dcf(get_confusion_matrix(commedia_llr_infpar > threshold,commedia_labels_infpar),prior,1,1,True))
    #     min_dcf = np.min([get_dcf(get_confusion_matrix(commedia_llr_infpar > t,commedia_labels_infpar),prior,1,1,True) for t in thresholds])
    #     min_dcfs.append(min_dcf)
    
    # plt.plot(eff_prior_logs_odds, dcfs, label='DCF', color='r')
    # plt.plot(eff_prior_logs_odds, min_dcfs, label='min DCF', color='b')
    # plt.ylim([0, 1.1])
    # plt.xlim([-3,3])
    # plt.legend()
    # plt.show()
    #endregion

    #region Comparing recognizer
    # eff_prior_logs_odds = np.linspace(-3,3,21)
    # priors =[1/(1+np.exp(-prior_log_odds)) for prior_log_odds in eff_prior_logs_odds]
    # dcfs_eps1 = []
    # min_dcfs_eps1 = []
    # thresholds = np.linspace(commedia_llr_infpar_eps1[0],commedia_llr_infpar_eps1[-1],1000)

    # for prior in priors:
    #     threshold = compute_optimal_Bayes_binary_threshold(prior,1,1)

    #     dcf_eps1 = get_dcf(get_confusion_matrix(commedia_llr_infpar_eps1 > threshold,commedia_labels_infpar_eps1),prior,1,1,True)
    #     dcfs_eps1.append(dcf_eps1)

    #     min_dcf = np.min([get_dcf(get_confusion_matrix(commedia_llr_infpar_eps1 > t,commedia_labels_infpar_eps1),prior,1,1,True) for t in thresholds])
    #     min_dcfs_eps1.append(min_dcf)

    # plt.plot(eff_prior_logs_odds, dcfs_eps1, label='DCF', color='r')
    # plt.plot(eff_prior_logs_odds, min_dcfs_eps1, label='min DCF', color='b')
    # plt.ylim([0, 1.1])
    # plt.legend()
    # plt.show()
    #endregion

    #region Multiclass evaluation
    # prior = np.array([0.3,0.4,0.3])
    prior = np.ones(3)/3.0
    # costMatrix = np.array([[0,1,2],[1,0,1],[2,1,0]])
    costMatrix = uniform_cost_matrix(3)
    # eps 0.001
    print('EPS 0.001\n')
    commedia_posteriors = compute_posteriors(commedia_ll, prior)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, costMatrix)

    c_matrix = get_confusion_matrix(commedia_predictions,commedia_labels)
    print(get_dcf_multiclass(c_matrix,costMatrix,prior,False))
    print(get_dcf_multiclass(c_matrix,costMatrix,prior,True))
    # eps 1
    print('EPS 1\n')
    commedia_posteriors = compute_posteriors(commedia_ll_eps1, prior)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, costMatrix)

    c_matrix = get_confusion_matrix(commedia_predictions,commedia_labels_eps1)
    print(get_dcf_multiclass(c_matrix,costMatrix,prior,False))
    print(get_dcf_multiclass(c_matrix,costMatrix,prior,True))
    #endregion


 
