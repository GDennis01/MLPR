import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import scipy
import sklearn.metrics as skmetrics
import scipy.optimize


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

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

# x is an array of shape (2,)
def f(x):
    y,z = x
    return (y+3)**2 + np.sin(y) + (z+1)**2

def fprime(x):
    y,z = x
    return np.array([2*(y+3) + np.cos(y), 2 * (z+1)])

# Optimize the logistic regression loss
def train_logreg_binary(DTR, LTR, l):
    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])
    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]


def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]
  
if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # w,b = train_logreg_binary(DTR,LTR,0.1)
    # print(w)
    # scores = w @ DTE + b
    # predicted_labels = (scores > 0).astype(np.int32)

    for l in [1e-3,1e-1,1.0]:
        # #Binary Logistic Regression
        # w,b = train_logreg_binary(DTR,LTR,l)
        # print(f'Lambda: {l} with w:{w} and b:{b}')
        # scores = np.dot(w.T,DTE) + b
        # #predictions using scores from logistic regression
        # predicted_labels = (scores > 0)*1
        # accuracy = get_accuracy(predicted_labels,LTE)
        # print(f'Err rate:{100-accuracy}')

        # #empirical prior and scores ssr like
        # empirical_prior = (LTR == 1).sum() / LTR.size
        # scores_llr_like = scores - np.log(empirical_prior/(1-empirical_prior))
        # #DCF
        # threshold = compute_optimal_Bayes_binary_threshold(0.5,1,1)
        # predicted_labels = (scores_llr_like > threshold)*1
        # dcf = get_dcf(get_confusion_matrix(predicted_labels,LTE),0.5,1,1,normalized=True)
        # print(f'DCF:{dcf}')
        # #Min DCF
        # thresholds = np.linspace(scores_llr_like.min(),scores_llr_like.max(),100)
        # min_dcf = np.min([get_dcf(get_confusion_matrix((scores_llr_like > t)*1,LTE),0.5,1,1,normalized=True) for t in thresholds])
        # print(f'Min DCF:{min_dcf}\n')

        # #Weighted Logistic Regression
        # empirical_prior = 0.8
        # w,b = train_weighted_logreg_binary(DTR,LTR,l,0.8)
        # # print(f'Lambda: {l} with w:{w} and b:{b}')
        # scores = np.dot(w.T,DTE) + b
        # scores_llr_like = scores - np.log(empirical_prior/(1-empirical_prior))
        # #DCF
        # threshold = compute_optimal_Bayes_binary_threshold(0.8,1,1)
        # predicted_labels = (scores_llr_like > threshold)*1
        # dcf = get_dcf(get_confusion_matrix(predicted_labels,LTE),0.8,1,1,normalized=True)
        # print(f'DCF:{dcf}')
        # #Min DCF
        # thresholds = np.linspace(scores_llr_like.min(),scores_llr_like.max(),100)
        # min_dcf = np.min([get_dcf(get_confusion_matrix((scores_llr_like > t)*1,LTE),0.8,1,1,normalized=True) for t in thresholds])
        # print(f'Min DCF:{min_dcf}\n')

        #Multiclass Logistic Regression
        w,b = train_multiclass_logreg(DTR,LTR,l)
        print(f'Lambda: {l} with w:{w} and b:{b}')
        scores = np.dot(w.T,DTE) + b
        # scores_llr_like = scores - np.log(empirical_prior/(1-empirical_prior))

