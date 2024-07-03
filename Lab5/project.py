import matplotlib.pyplot as plt
import numpy as np

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
def heatmap(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, '%.2f' % data[i, j],
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5),
                     fontsize=8)
    plt.imshow(data)
    plt.show()

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
# def uni_GAU(x,var,mu):
#     return (1/(np.sqrt(2*np.pi*var))) * np.exp(np.e,-(((x-mu)**2)/2*var))
def uni_GAU(x,var,mu):
    return (1/(np.sqrt(2*np.pi*var))) * np.exp(-(((x-mu)**2)/(2*var)))
    
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

def PCA(dataset,n_dim,specif_dim=None):
    mu_ds = dataset.mean(1)
    centered_dataset = dataset - vcol(mu_ds)
    covar_ds = (centered_dataset@centered_dataset.T)/dataset.shape[1]
    # covar_ds = np.cov(dataset.T,bias=True)

    U, s, Vh = np.linalg.svd(covar_ds)
    if specif_dim == None:
        P = U[:,0:n_dim]
    else:
        P = U[:,specif_dim:specif_dim+1]

    return P.T

def predict_data(DVAL, LVAL,LLR,threshold):
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[LLR >= threshold] = 1
    PVAL[LLR < threshold] = 0

    myarr = PVAL==LVAL
    correct_pred =  myarr.sum()
    print(f'Correct predictions: {correct_pred}')
    false_pred = myarr.size - correct_pred
    print(f'False predictions: {false_pred}')
    print(f'Error rate: {false_pred/myarr.size*100}% over {myarr.size} samples.')
    return false_pred/myarr.size*100

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
    
# this way to compute accuracy yields the *same* exact result as the one in predict_data
def get_accuracy(predicted,actual):
    return (predicted==actual).sum()/len(predicted)*100
    

if __name__ == '__main__':
    features,classes=load("trainData.txt")
    (DTR, LTR), (DTE, LTE) = split_db_2to1(features, classes)

    #region MVG classifier
    print("MVG:")
    SJoint,_,p_labels = mvg(DTR,LTR,DTE,LTE)
    llr = np.log(SJoint[1]/SJoint[0])
    predict_data(DTE,LTE,llr,0)
    acc = get_accuracy(p_labels,LTE)
    print("")
    print(f"Accuracy: {acc}%")
    #endregion

    #region Tied Covariance classifier
    # print("Tied MVG:")
    # SJoint,p_labels = tied_cov(DTR,LTR,DTE,LTE)
    # llr = SJoint[1]-SJoint[0]
    # predict_data(DTE,LTE,llr,0)
    # acc = get_accuracy(p_labels,LTE)
    # print("Accuracy: ",acc)
    # print("")
    #endregion

    #region Naive Bayes classifier
    # print("Naive Bayes:")
    # SJoint,p_labels = naive_bayes(DTR,LTR,DTE,LTE)
    # llr = SJoint[1]-SJoint[0]
    # acc = get_accuracy(p_labels,LTE)
    # predict_data(DTE,LTE,llr,0)
    # print("Accuracy: ",acc)
    #endregion

    #region Pearson Correlation
    # _,covs,_ = mvg(DTR,LTR,DTE,LTE)
    # heatmap(covs[0])
    # heatmap(covs[1])
    # Corr_false = covs[0] / ( vcol(covs[0].diagonal()**0.5) * vrow(covs[0].diagonal()**0.5) )
    # Corr_true = covs[1] / ( vcol(covs[1].diagonal()**0.5) * vrow(covs[1].diagonal()**0.5) )
    # heatmap(Corr_false)
    # heatmap(Corr_true)
    #endregion

    #region Naive bayes features 1-4
    # DTRx = DTR[0:4,:]
    # DTEx = DTE[0:4,:]
    # SJoint,p_labels = naive_bayes(DTRx,LTR,DTEx,LTE)
    # # llr = SJoint[1]-SJoint[0]
    # llr = np.log(SJoint[1]/SJoint[0])
    # print("Naive Bayes:")
    # acc = get_accuracy(p_labels,LTE)
    # print(f"Accuracy: {acc}%")
    # predict_data(DTEx,LTE,llr,0)
    # print("")
    #endregion

    #region Tied Cov features 1-4
    # DTRx = DTR[0:4,:]
    # DTEx = DTE[0:4,:]
    # print("Tied cov:")
    # SJoint,p_labels = tied_cov(DTRx,LTR,DTEx,LTE)
    # # llr = SJoint[1]-SJoint[0]
    # llr = np.log(SJoint[1]/SJoint[0])
    # acc = get_accuracy(p_labels,LTE)
    # print(f"Accuracy: {acc}%")
    # predict_data(DTEx,LTE,llr,0)
    # print("")
    #endregion

    #region MVG features 1-4
    # DTRx = DTR[0:4,:]
    # DTEx = DTE[0:4,:]
    # SJoint,_,p_labels = mvg(DTRx,LTR,DTEx,LTE)
    # # llr = SJoint[1]-SJoint[0]
    # llr = np.log(SJoint[1]/SJoint[0])
    # acc = get_accuracy(p_labels,LTE)
    # print(f"Accuracy: {acc}%")
    # predict_data(DTEx,LTE,llr,0)
    #endregion

    #region MVG and Tied pair 1-2 3-4
    # DTRx = DTR[0:2,:]
    # DTEx = DTE[0:2,:]
    # SJoint,_,p_labels = mvg(DTRx,LTR,DTEx,LTE)
    # llr = SJoint[1]-SJoint[0]
    # print("MVG features 1-2")
    # acc = get_accuracy(p_labels,LTE)
    # print(f"Accuracy: {acc}%")
    # predict_data(DTEx,LTE,llr,0)

    # DTRx = DTR[2:4,:]
    # DTEx = DTE[2:4,:]
    # SJoint,_ ,p_labels= mvg(DTRx,LTR,DTEx,LTE)
    # llr = SJoint[1]-SJoint[0]
    # print("\n\nMVG features 3-4")
    # acc = get_accuracy(p_labels,LTE)
    # print(f"Accuracy: {acc}%")
    # predict_data(DTEx,LTE,llr,0)

    # DTRx = DTR[0:2,:]
    # DTEx = DTE[0:2,:]
    # SJoint,p_labels = tied_cov(DTRx,LTR,DTEx,LTE)
    # llr = SJoint[1]-SJoint[0]
    # print("\n\nTied MVG features 1-2")
    # acc = get_accuracy(p_labels,LTE)
    # print(f"Accuracy: {acc}%")
    # predict_data(DTEx,LTE,llr,0)

    # DTRx = DTR[2:4,:]
    # DTEx = DTE[2:4,:]
    # SJoint,p_labels = tied_cov(DTRx,LTR,DTEx,LTE)
    # llr = SJoint[1]-SJoint[0]
    # print("\n\nTied MVG features 3-4")
    # acc = get_accuracy(p_labels,LTE)
    # print(f"Accuracy: {acc}%")
    # predict_data(DTEx,LTE,llr,0)
    #endregion

    #region PCA pre-processing
    accuracies_mvg = []
    accuracies_tied = []
    accuracies_naive = []
    for i in range(1,6):
        m=i+1
        print(f"PCA with {m} dimensions")
        PCA_m = PCA(features,m)
        proj_data = PCA_m@features
        (DTR, LTR), (DTE, LTE) = split_db_2to1(proj_data, classes)

        print("MVG:")
        SJoint,_,_ = mvg(DTR,LTR,DTE,LTE)
        # llr = SJoint[1]-SJoint[0]
        llr = np.log(SJoint[1]/SJoint[0])
        acc = 100-predict_data(DTE,LTE,llr,0)
        accuracies_mvg.append(acc)

        print("\n\nTied MVG:")
        SJoint,_ = tied_cov(DTR,LTR,DTE,LTE)
        llr = SJoint[1]-SJoint[0]
        acc = 100-predict_data(DTE,LTE,llr,0)
        accuracies_tied.append(acc)

        print("\n\nNaive Bayes:")
        SJoint,_ = naive_bayes(DTR,LTR,DTE,LTE)
        llr = SJoint[1]-SJoint[0]
        acc = 100-predict_data(DTE,LTE,llr,0)
        accuracies_naive.append(acc)
    print(accuracies_mvg)
    print(accuracies_tied)
    print(accuracies_naive) 
    plt.plot(range(2,7),accuracies_mvg,marker="o",label="MVG")
    plt.plot(range(2,7),accuracies_naive,marker="o",label="Naive Bayes")
    plt.plot(range(2,7),accuracies_tied,marker="o",label="Tied MVG")
    plt.xlabel("Number of features")
    plt.xticks(range(2,7))
    plt.ylabel("Accuracy(%)")
    plt.tight_layout()
    plt.legend()
    plt.show()
    #endregion