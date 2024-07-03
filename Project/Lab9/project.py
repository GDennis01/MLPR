import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
from svm import SVM
from prettytable import PrettyTable
sys.dont_write_bytecode = True

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


def get_confusion_matrix(predicted,actual):
    n_labels = len(np.unique(actual))
    confusion_matrix = np.zeros((n_labels,n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            confusion_matrix[i,j] = np.sum((predicted==i)&(actual==j))
    return confusion_matrix
def get_missclass_matrix(confusion_matrix):
    return confusion_matrix / vrow(confusion_matrix.sum(0))


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

def train_weighted_logreg_binary(DTR, LTR, l, pT):
    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1] 

# needed for the "quadratic" logistic regression, basically apply the expanded feature set to the normal logreg
def quadratic_feature_expansion(X):
    X_T = X.T
    X_expanded = []
    for x in X_T:
        outer_product = np.outer(x, x).flatten()
        expanded_feature = np.concatenate([outer_product, x])
        X_expanded.append(expanded_feature)
    X_expanded = np.array(X_expanded).T
    return X_expanded

def get_min_dcf(SVAL,LVAL,prior,Cfn,Cfp):
    thresholds = np.concatenate([np.array([-np.inf]), SVAL, np.array([np.inf])])
    min_dcf = np.min([get_dcf(get_confusion_matrix((SVAL>t),LVAL),prior,1,1,normalized=True) for t in thresholds])
    return min_dcf

# compute both dcf and min dcf for a given logreg model. it's mainly an utility function lest we repeat code
# prior is needed only for the weighted model
def get_dcf_mindcf_logreg(DTR,LTR,DTE,LTE,lambdas,prior,model=train_logreg_binary):
    dcfs = []
    min_dcfs = []
    for l in lambdas:
        print(f'Lambda: {l}')

        if model == train_logreg_binary:
            w,b = model(DTR, LTR, l)
        elif model == train_weighted_logreg_binary:
            w,b = model(DTR, LTR, l,prior)
        else:
            w,b = model(DTR, LTR, l)
        # w,b = model(DTR, LTR, l)
        
        scores = logreg_scores(w,b,DTE)
        empirical_prior = (LTR==1).sum()/LTR.size
        print("@@@@"+str(empirical_prior))

        if model == train_logreg_binary:
            scores_llr = scores - np.log(empirical_prior/(1-empirical_prior))
        elif model == train_weighted_logreg_binary:
            scores_llr = scores - np.log(prior/(1-prior))
        else:
            scores_llr = scores - np.log(empirical_prior/(1-empirical_prior))

        # scores_llr = scores - np.log(empirical_prior/(1-empirical_prior))

        threshold = compute_optimal_Bayes_binary_threshold(prior, 1, 1)
        predicted_labels = scores_llr > threshold
        confusion_matrix = get_confusion_matrix(predicted_labels,LTE)

        dcf = get_dcf(confusion_matrix,prior,1,1,normalized=True)
        dcfs.append(dcf)
        print(f'DCF: {dcf}')

        thresholds = np.linspace(scores_llr.min(),scores_llr.max(),100)
        min_dcf = np.min([get_dcf(get_confusion_matrix(scores_llr>t,LTE),0.1,1,1,normalized=True) for t in thresholds])
        min_dcfs.append(min_dcf)
        print(f'Min DCF: {min_dcf}\n')
    return dcfs,min_dcfs

def svm_scores(w,b,DTE):
    return (vrow(w) @ DTE + b).ravel()
    
def plot_dcf_vs_c(lambdas,dcfs,min_dcfs):
    plt.figure()
    plt.plot(lambdas,dcfs,label='DCFs')
    plt.plot(lambdas,min_dcfs,label='Min DCFs')
    plt.xscale('log',base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs C')
    plt.legend()
    plt.show()

# Optimize SVM


# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbfKernelFunc


    
if __name__ == '__main__':
    features,classes=load("trainData.txt")
    # take only 1/10 of the data
    features = features[:,:int(features.shape[1]/10)]
    classes = classes[:int(classes.size/10)]
    (DTR, LTR), (DTE, LTE) = split_db_2to1(features, classes)
    prior = 0.1
    svm = SVM(DTR,LTR)

    #region Linear SVM with DCF and MinDCF as C varies
    # dcfs = []
    # min_dcfs = []
    # for C in np.logspace(-5,0,11):
    #     w,b= svm.train(C,'linear',K=1)

    #     scores = svm_scores(w,b,DTE)
    #     threshold = compute_optimal_Bayes_binary_threshold(prior,1,1)
    #     PVAL = (scores > threshold) * 1
    #     conf_matrix = get_confusion_matrix(PVAL, LTE)

    #     min_dcf =  get_min_dcf(scores, LTE, prior, 1.0, 1.0)
    #     min_dcfs.append(min_dcf)

    #     dcf = get_dcf(conf_matrix, prior, 1.0, 1.0,normalized=True)
    #     dcfs.append(dcf)
    # plot_dcf_vs_c(np.logspace(-5,0,11),dcfs,min_dcfs)
    #endregion

    #region Polynomial Kernel SVM with DCF and MinDCF as C varies
    # dcfs = []
    # min_dcfs = []
    # for C in np.logspace(-5,0,11):
    #     fscore= svm.train(C,'kernel',K=1,kernelFunc=polyKernel(2,1))

    #     scores = fscore(DTE)
    #     threshold = compute_optimal_Bayes_binary_threshold(prior,1,1)
    #     PVAL = (scores > threshold) * 1
    #     conf_matrix = get_confusion_matrix(PVAL, LTE)

    #     min_dcf =  get_min_dcf(scores, LTE, prior, 1.0, 1.0)
    #     min_dcfs.append(min_dcf)

    #     dcf = get_dcf(conf_matrix, prior, 1.0, 1.0,normalized=True)
    #     dcfs.append(dcf)
    # plot_dcf_vs_c(np.logspace(-5,0,11),dcfs,min_dcfs)
    #endregion

    #region RBF Kernel SVM with DCF and MinDCF as C varies
    gamma = [("e-4",np.exp(-4)),("e-3",np.exp(-3)),("e-2",np.exp(-2)),("e-1",np.exp(-1))]
    Cs = np.logspace(-3,2,11)
    dcfs = []
    min_dcfs = []
    for l,g in gamma:
        dcfs = []
        min_dcfs = []
        for C in Cs:
            fscore= svm.train(C,'kernel',kernelFunc=rbfKernel(g))

            scores = fscore(DTE)
            threshold = compute_optimal_Bayes_binary_threshold(prior,1,1)
            PVAL = (scores > threshold) * 1
            conf_matrix = get_confusion_matrix(PVAL, LTE)

            min_dcf =  get_min_dcf(scores, LTE, prior, 1.0, 1.0)
            min_dcfs.append(min_dcf)

            dcf = get_dcf(conf_matrix, prior, 1.0, 1.0,normalized=True)
            dcfs.append(dcf)
        plt.plot(Cs,dcfs,label=f'DCF with Gamma: {l}')
        plt.plot(Cs,min_dcfs,label=f'MinDCF with Gamma: {l}')
    plt.xscale('log',base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs C')
    plt.legend()
    plt.show()
    #endregion
        

