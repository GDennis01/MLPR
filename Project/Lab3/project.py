import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

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

def _plot_scatter_nested(dataset,classes,n_label,labels):
    for i in range(dataset.shape[0]-1):
        for j in range(dataset.shape[0]-1):
            print("Calling subplot")
            plt.subplot(dataset.shape[0]-1,dataset.shape[0]-1,i+j+1)
            for z in range(n_label):
                print("sublotp")
                plt.scatter(dataset[i][classes==z],dataset[j][classes==z],alpha=0.5,label=labels[z])
                # plt.tight_layout()
                plt.legend()  
    plt.show()

def plot_scatter_nested(dataset,classes,n_label,labels):    
    for i in range(dataset.shape[0]):
        for j in range(i+1,dataset.shape[0]):
            plt.scatter(dataset[i][classes==0],dataset[j][classes==0],alpha=0.5,label=labels[0])
            plt.scatter(dataset[i][classes==1],dataset[j][classes==1],alpha=0.5,label=labels[1])
            plt.xlabel(f'Feature {i+1}')
            plt.ylabel(f'Feature {j+1}')
            plt.legend()
            plt.savefig(f'scatter_plots/feature_{i+1}_vs_feature_{j+1}.png')
            plt.clf()

def plot_scatter(dataset,classes,n_label,invert_xaxis=False,invert_yaxis=False,multiple=-1):
    for i in range(n_label):
        plt.scatter(dataset[0][classes==i],dataset[1][classes==i],alpha=0.5)
    if invert_xaxis:
        plt.gca().invert_xaxis()
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.show()

def plot_hist(dataset,classes,n_label,labels,bins,invert_xaxis=False,invert_yaxis=False,multiple=-1):
    for i in range(n_label):
        if multiple != -1:
            plt.subplot(3,2,multiple+1) 
            plt.xlabel(f'{multiple+1}° PCA Direction')
        plt.hist(dataset[0,classes==i],density=True,bins=bins,alpha=0.5,label=labels[i])
        # plt.xlabel(labels[i])
    if invert_xaxis:
        plt.gca().invert_xaxis()
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.legend()
    if multiple == -1:
        plt.show()



def vcol(dset):
    return dset.reshape((dset.size,1))
def vrow(dset):
    return dset.reshape((1,dset.size))

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


def cov_between_classes(dataset,n_label,classes):
    mu_ds = vcol(dataset.mean(1))
    Sb=0
    # for val in range(1,3):
    for val in range(n_label):
        ds_cl = dataset[:,classes==val]
        mu_c = vcol(ds_cl.mean(1))
        dp_means = mu_c-mu_ds
        Sb=Sb+(dp_means@dp_means.T)*ds_cl.shape[1]
    return Sb/dataset.shape[1]

def cov_within_classes(dataset,n_label,classes):
    Sw=0
    for val in range(n_label):
        ds_cl = dataset[:,classes==val]
        ## Bias=True makes it so that the denominator is N instead of N-1
        swc=np.cov(ds_cl,bias=True)
        Sw+= swc*ds_cl.shape[1]
        ## "Manual" implementation of np.cov. Since np.cov is faster however, I opted for the np lib version
        ## Note: both versions yield the same result!
        # mu = ds_cl.mean(1)
        # ds_cl_cent = ds_cl-vcol(mu)
        # Sw+=ds_cl_cent@ds_cl_cent.T
    # special case when there's only 1 dimension
    if dataset.shape[0] == 1:
        Sw = np.array(Sw,ndmin=2)
    return Sw/dataset.shape[1]
def LDA(cov_wt,cov_bt,m):
    U,s,_ = np.linalg.svd(cov_wt)
    P1 = (U@  np.diag(1.0/(s**0.5)))@ U.T 

    trans_cov_bt = (P1@ cov_bt) @ P1.T 
    #retrieving eigen vec of trans_cov_btw
    eigen_val,eigen_vec = np.linalg.eig(trans_cov_bt)
    #sorting eigen vectors based on eigenvalues
    sorted_index = np.argsort(eigen_val)[::-1]
    eigen_vec = eigen_vec[:, sorted_index]
    #P2= top m eigenvectors
    P2 = eigen_vec[:,0:m]

    LDA_matrix = np.dot(P1.T,P2)
    return LDA_matrix.T
def _LDA(cov_wt,cov_bt,m):
    s, U = scipy.linalg.eigh(cov_bt, cov_wt)
    W = U[:, ::-1][:, 0:m]
  

    return W.T

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)

    idx = np.random.permutation(D.shape[1])

    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)    

def predict_data(DVAL, LVAL,DVAL_lda,threshold):
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    # Se i valori di LDA applicati al validation set sono >= threshold, predico la classe 1
    PVAL[DVAL_lda[0] >= threshold] = 1
    #Altrimenti la classe 0
    PVAL[DVAL_lda[0] < threshold] = 0
    print(f'Threshold:{threshold}\n')

    myarr = PVAL==LVAL
    # True are counted as 1, False as 0 so I can use sum to count the correct predictions
    correct_pred =  myarr.sum()
    print(f'Correct predictions: {correct_pred}')
    # sum false values
    false_pred = myarr.size - correct_pred
    print(f'False predictions: {false_pred}')
    print(f'Error rate: {false_pred/myarr.size*100}% over {myarr.size} samples.')
    return false_pred/myarr.size*100

if __name__ == '__main__':
    features,classes=load('trainData.txt')
    m=2


    #region PCA
    for i in range(6):
        PCA_matrix = PCA(features,1,i)
        proj_data = PCA_matrix@features
        plot_hist(proj_data,classes,2,['False','True'],20,multiple=i)
    plt.show()
    PCA_matrix = PCA(features,2)
    proj_data = PCA_matrix@features
    plot_scatter_nested(proj_data,classes,2,['False','True'])
    #endregion

    #region LDA
    # cov_bt_cl = cov_between_classes(features,2,classes)
    # cov_wt_cl = cov_within_classes(features,2,classes)
    # LDA_matrix = LDA(cov_wt_cl,cov_bt_cl,1)

    # proj_data = LDA_matrix@features
    # plot_hist(proj_data,classes,2,['False','True'],20)
    #endregion

    #region LDA as classifier
    # (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)

    # cov_bt_cl = cov_between_classes(DTR,2,LTR)
    # cov_wt_cl = cov_within_classes(DTR,2,LTR)
    # DTR_lda_m = LDA(cov_wt_cl,cov_bt_cl,m)

    # #Applico LDA al training set
    # DTR_lda = DTR_lda_m@DTR
    # #Applico LDA al validation set
    # DVAL_lda = DTR_lda_m@DVAL

    # # computing threshold
    # mean1 = DTR_lda[0,LTR==0].mean()
    # mean2 = DTR_lda[0,LTR==1].mean()
    # print(f'Mean True{mean1}:\nMean False:{mean2}')
    # #Calcolo una soglia per predirre i valori in base alle medie del training set
    # threshold = (mean1+mean2)/ 2.0
    # predict_data( DVAL, LVAL,DVAL_lda,threshold)
    #endregion

    #region Changing Threshold
    # # Notando il grafico, vediamo il punto con meno overlap delle classi è 0, quindi tramite
    # # bruteforcing scelgo 0.0211 per migliorare l'accuracy
    # (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
   
    # cov_bt_cl = cov_between_classes(DTR,2,LTR)
    # cov_wt_cl = cov_within_classes(DTR,2,LTR)
    # DTR_lda_m = LDA(cov_wt_cl,cov_bt_cl,m)

    # #Applico LDA al training set
    # DTR_lda = DTR_lda_m@DTR
    # #Applico LDA al validation set
    # DVAL_lda = DTR_lda_m@DVAL
    # threshold = 0.0211

    # predict_data( DVAL, LVAL,DVAL_lda,threshold)
    #endregion
    
    #region PCA+LDA
    # err_rates =[]
    # for i in range(6):
    #     PCA_matrix = PCA(features,i+1)
    #     proj_data = PCA_matrix@features
    #     (DTR, LTR), (DVAL, LVAL) = split_db_2to1(proj_data, classes)

    #     cov_bt_cl = cov_between_classes(DTR,2,LTR)
    #     cov_wt_cl = cov_within_classes(DTR,2,LTR)
    #     DTR_lda_m = LDA(cov_wt_cl,cov_bt_cl,1)
        
    #     DTR_lda = DTR_lda_m@DTR
    #     DVAL_lda = DTR_lda_m@DVAL

    #     mean1 = DTR_lda[0,LTR==0].mean()
    #     mean2 = DTR_lda[0,LTR==1].mean()
    #     threshold = (mean1+mean2)/ 2.0
    #     err_rates.append(predict_data( DVAL, LVAL,DVAL_lda,threshold))
    # print(err_rates)
    # plt.plot(range(1,7),err_rates,marker="o")
    # plt.xlabel("Number of dimensions while applying PCA")
    # plt.ylabel("Error rate(%)")
    # plt.tight_layout()
    # plt.show()
    #endregion