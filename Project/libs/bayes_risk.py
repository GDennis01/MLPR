import numpy as np
"""
Confusion Matrix related functions
"""
def get_false_negatives(conf_matrix):
    return conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[1,1])
def get_false_positives(conf_matrix):
    return conf_matrix[1,0]/(conf_matrix[0,0]+conf_matrix[1,0])
def get_true_positives(conf_matrix):
    return 1-get_false_negatives(conf_matrix)
def get_confusion_matrix(predicted,actual):
    n_labels = len(np.unique(actual))
    confusion_matrix = np.zeros((n_labels,n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            confusion_matrix[i,j] = np.sum((predicted==i)&(actual==j))
    return confusion_matrix
def get_missclass_matrix(confusion_matrix):
    return confusion_matrix / vrow(confusion_matrix.sum(0))
def uniform_cost_matrix(nClasses):
    return np.ones((nClasses, nClasses)) - np.eye(nClasses)
    
def get_effective_prior(prior,Cfn,Cfp):
    return (prior*Cfn)/(prior*Cfn+((1-prior)*(Cfp)))

"""
DCF and MinDCF functions
"""
# def get_dcf(conf_matrix,prior,Cfn,Cfp,normalized=False):
#     _dcf = prior*Cfn*get_false_negatives(conf_matrix) + (1-prior)*Cfp*get_false_positives(conf_matrix)
#     if normalized:
#         return _dcf/min(prior*Cfn,(1-prior)*Cfp)
#     return _dcf

def get_dcf(SVAL,LVAL,prior=0.5,Cfn=1,Cfp=1,normalized=False,threshold=0):
    """
    Compute the Detection Cost Function (DCF) for a given threshold
    Args:
        SVAL: The scores of the samples i.e. LLR
        LVAL: The true labels of the samples
        prior: The prior probability of the positive class
        Cfn: The cost of false negatives
        Cfp: The cost of false positives
        normalized: If True, the DCF is normalized by the minimum cost
        threshold: The threshold to use for the classifier. If 'optimal' is passed, the optimal threshold is computed. Default to 0
    Returns:
        The Detection Cost Function (DCF) value
    """
    if threshold == 'optimal':
        threshold = compute_optimal_Bayes_binary_threshold(prior,Cfn,Cfp)
    conf_matrix = get_confusion_matrix((SVAL>threshold),LVAL)
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
def get_min_dcf(SVAL,LVAL,prior=0.5,Cfn=1,Cfp=1):
    # thresholds = np.concatenate([np.array([-np.inf]), SVAL, np.array([np.inf])])
    # thresholds = np.linspace(SVAL.min(),SVAL.max(),1000)
    # sort SVAL
    # thresholds = np.sort(SVAL)
    thresholds = SVAL 
    # min_dcf = np.min([get_dcf(get_confusion_matrix((SVAL>t),LVAL),prior,Cfn,Cfp,normalized=True) for t in thresholds])
    min_dcf = np.min([get_dcf(SVAL,LVAL,prior,Cfn,Cfp,normalized=True,threshold=t) for t in thresholds])
    return min_dcf

""" 
Optimal Threshold functions 
"""
# Compute optimal Bayes decisions for the matrix of class posterior (each column refers to a sample)
def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return np.argmin(expectedCosts, 0)


def compute_optimal_Bayes_binary_threshold(prior, Cfn, Cfp):
    return -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )   


def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(np.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return np.exp(logPost)
