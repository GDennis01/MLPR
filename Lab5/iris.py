import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import scipy


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

if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    SJoint_MVG = np.load("Solution/SJoint_MVG.npy")
    logMarginal_MVG = np.load("Solution/logMarginal_MVG.npy")
    logSJoint_MVG = np.load("Solution/logSJoint_MVG.npy")
    logPosterior_MVG = np.load("Solution/logPosterior_MVG.npy")
    
    SJoint_NaiveBayes = np.load("Solution/SJoint_NaiveBayes.npy")
    logMarginal_NaiveBayes = np.load("Solution/logMarginal_NaiveBayes.npy")
    logSJoint_NaiveBayes = np.load("Solution/logSJoint_NaiveBayes.npy")
    logPosterior_NaiveBayes = np.load("Solution/logPosterior_NaiveBayes.npy")
    Posterior_NaiveBayes = np.load("Solution/Posterior_NaiveBayes.npy")

    SJoint_TiedMVG = np.load("Solution/SJoint_TiedMVG.npy")
    logMarginal_TiedMVG = np.load("Solution/logMarginal_TiedMVG.npy")
    logSJoint_TiedMVG = np.load("Solution/logSJoint_TiedMVG.npy")
    logPosterior_TiedMVG = np.load("Solution/logPosterior_TiedMVG.npy")
    Posterior_TiedMVG = np.load("Solution/Posterior_TiedMVG.npy")

    #region ML Estimates parameters
    ## class 0
    # print(DTR.shape)
    # print(DTR[:,LTR==0].mean(1))
    # print(np.cov(DTR[:,LTR==0],bias=True))
    #endregion

    #region Multivariate Gaussian Classifier
    # classes = np.unique(LTE)
    # SJoint=np.zeros((len(classes), DTE.shape[1]))
    # for i,l in enumerate(classes):
    #     class_sample_l = DTR[:,LTR==l]
    #     cov = np.cov(class_sample_l,bias=True)
    #     mu = np.mean(class_sample_l,axis=1,keepdims=True)
    #     SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/3)
    # SMarginal =  vrow(SJoint.sum(0))
    # # each row represents the probability of test samples of being in that class
    # SPost = SJoint / SMarginal
    # predicted_labels = np.argmax(SPost,axis=0)
    
    # accuracy = ((LTE==predicted_labels).sum(0))/len(predicted_labels)*100
    # print(f'accuracy:{accuracy}')
    # err_rate = 1-accuracy

    # logSJoint = np.log(SJoint)
    # logSMarginal = scipy.special.logsumexp(logSJoint,axis=0)

    # logSPost = logSJoint - logSMarginal
    # SPost = np.exp(logSPost)
    #endregion

    #region Naive Bayes Gaussian Classifier
    # classes = np.unique(LTE)
    # SJoint=np.zeros((len(classes), DTE.shape[1]))
    # for i,l in enumerate(classes):
    #     class_sample_l = DTR[:,LTR==l]
    #     cov = np.cov(class_sample_l,bias=True)
    #     cov = np.diag(np.diag(cov))
    #     mu = np.mean(class_sample_l,axis=1,keepdims=True)
    #     SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/3)
    # SMarginal =  vrow(SJoint.sum(0))
    # # each row represents the probability of test samples of being in that class
    # SPost = SJoint / SMarginal
    # predicted_labels = np.argmax(SPost,axis=0)
    
    # accuracy = ((LTE==predicted_labels).sum(0))/len(predicted_labels)*100
    # err_rate = 1-accuracy
    # print(f'accuracy:{accuracy}')

    # logSJoint = np.log(SJoint)
    # logSMarginal = scipy.special.logsumexp(logSJoint,axis=0)

    # logSPost = logSJoint - logSMarginal
    # SPost = np.exp(logSPost)
    # print(SPost[1])
    # print(Posterior_NaiveBayes[1])
    #endregion

    #region Tied Covariance Gaussian Classifier
    # classes = np.unique(LTE)
    # SJoint=np.zeros((len(classes), DTE.shape[1]))
    # cov = cov_within_classes(DTR,3,LTR)
    # for i,l in enumerate(classes):
    #     class_sample_l = DTR[:,LTR==l]
    #     mu = np.mean(class_sample_l,axis=1,keepdims=True)
    #     SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/3)
    # SMarginal =  vrow(SJoint.sum(0))
    # # each row represents the probability of test samples of being in that class
    # SPost = SJoint / SMarginal
    # predicted_labels = np.argmax(SPost,axis=0)
    
    # accuracy = ((LTE==predicted_labels).sum(0))/len(predicted_labels)*100
    # err_rate = 1-accuracy
    # print(f'accuracy:{accuracy}')

    # logSJoint = np.log(SJoint)
    # logSMarginal = scipy.special.logsumexp(logSJoint,axis=0)

    # logSPost = logSJoint - logSMarginal
    # SPost = np.exp(logSPost)
    #endregion

    #region Binary tasks: log-likelihood ratios and MVG
    D, L = load_iris()
    # only class 1 and 2
    D = D[:, L != 0]
    L = L[L != 0]
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    classes = np.unique(LTE)
    SJoint=np.zeros((len(classes), DTE.shape[1]))
    for i,l in enumerate(classes):
        class_sample_l = DTR[:,LTR==l]
        cov = np.cov(class_sample_l,bias=True)
        mu = np.mean(class_sample_l,axis=1,keepdims=True)
        SJoint[i,:] =np.exp(logpdf_GAU_ND(DTE,mu,cov))*(1/3)
    llr = SJoint[1]-SJoint[0]
    predict_data(DTE,LTE,llr,0)
    #endregion


 
