import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets

def load(filename):
    records=[[],[],[],[]]
    dic_classes={
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2,
    }
    classes=[]
    with open(filename) as f:
        for l in f:
            fields = l.split(",")
            records[0].append(float(fields[0]))
            records[1].append(float(fields[1]))
            records[2].append(float(fields[2]))
            records[3].append(float(fields[3]))

            classes.append(dic_classes[fields[4].strip()])
        a = np.array(records)
        classes=np.array(classes)
        return a,classes

def vcol(dset):
    return dset.reshape((dset.size,1))
def vrow(dset):
    return dset.reshape((1,dset.size))

def PCA(dataset,n_dim):
    mu_ds = dataset.mean(1)
    centered_dataset = dataset - vcol(mu_ds)
    covar_ds = (centered_dataset@centered_dataset.T)/dataset.shape[1]

    U, s, Vh = np.linalg.svd(covar_ds)
    P = U[:, 0:n_dim]

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
    # for val in range(1,4):
    for val in range(n_label):
        print(val)
        ds_cl = dataset[:,classes==val]
        ## Bias=True makes it so that the denominator is N instead of N-1
        swc=np.cov(ds_cl,bias=True)
        Sw+= swc*ds_cl.shape[1]
        ## "Manual" implementation of np.cov. Since np.cov is faster however, I opted for the np lib version
        ## Note: both versions yield the same result!
        # mu = ds_cl.mean(1)
        # ds_cl_cent = ds_cl-vcol(mu)
        # Sw+=ds_cl_cent@ds_cl_cent.T
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
    # P2 = trans_cov_bt[:,0:m]
    #P2= top m eigenvectors
    P2 = eigen_vec[:,0:m]

    LDA_matrix = np.dot(P1.T,P2)
    
    return LDA_matrix.T
         

def plot_scatter(dataset,classes,n_label,invert_xaxis=False,invert_yaxis=False):
    for i in range(n_label):
        plt.scatter(dataset[0][classes==i],dataset[1][classes==i])
    if invert_xaxis:
        plt.gca().invert_xaxis()
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.show()
def plot_hist(dataset,classes,n_label,bins,invert_xaxis=False,invert_yaxis=False):
    for i in range(n_label):
        plt.hist(dataset[0,classes==i],density=True,bins=bins,alpha=0.5)
    if invert_xaxis:
        plt.gca().invert_xaxis()
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.show()

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

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

if __name__== '__main__':
    m = 2
    sol_eigvec = np.load("IRIS_PCA_matrix_m4.npy")
    sol_lda = np.load("IRIS_LDA_matrix_m2.npy")
    dic_classes={
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2,
    }
    features,classes=load("iris.csv")

    #region PCA
    # cov_bt_cl = cov_between_classes(features,3,classes)
    # cov_wt_cl = cov_within_classes(features,3,classes)
    # print(f'COV BT :\n{cov_bt_cl}\nCOV WT:\n{cov_wt_cl}')
    # PCA_matrix = PCA(features,m)
    # print(PCA_matrix)
    # print(sol_eigvec)
    # proj_data = PCA_matrix@features
    # plot_scatter(proj_data,classes,3)

    # proj_data = PCA(features,m)
    # plot_hist(proj_data,classes,3,20)
    #endregion
    
    #region LDA

    cov_bt_cl = cov_between_classes(features,3,classes)
    cov_wt_cl = cov_within_classes(features,3,classes)
    print(f'COV BT :\n{cov_bt_cl}\nCOV WT:\n{cov_wt_cl}')
    lda_m = LDA(cov_wt_cl,cov_bt_cl,m)
    print(f'LDA MIA:\n{lda_m}\nLDA PROF:\n{sol_lda}')
    proj_data = lda_m@features
    # plot_scatter(proj_data,classes,3,invert_xaxis=True)
    # plot_hist(proj_data,classes,3,15)
    #endregion

    #region PCA+LDA
    # DIris, LIris = load_iris()
    # D = DIris[:, LIris != 0]
    # L = LIris[LIris != 0]
    # (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # cov_bt_cl = cov_between_classes(DTR,2,LTR)
    # cov_wt_cl = cov_within_classes(DTR,2,LTR)

    # print(f'CovB:{cov_bt_cl}\nCovW:{cov_wt_cl}')

    # DTR_lda_m = LDA(cov_wt_cl,cov_bt_cl,m)
    # #Applico LDA al training set
    # DTR_lda = DTR_lda_m@DTR
    # #Applico LDA al validation set
    # DVAL_lda = DTR_lda_m@DVAL
    # # DTR_lda = LDA(cov_within_classes)

    # mean1 = DTR_lda[0,LTR==1].mean()
    # mean2 = DTR_lda[0,LTR==2].mean()
    # #Calcolo una soglia per predirre i valori in base alle medie del training set
    # threshold = (mean1+mean2)/ 2.0
    
    # print(f'{threshold}')
    # #Predicted Values
    # PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    # # Se i valori di LDA applicati al validation set sono >= threshold, predico la classe 2
    # PVAL[DVAL_lda[0] >= threshold] = 2
    # #Altrimenti la classe 1
    # PVAL[DVAL_lda[0] < threshold] = 1

    # print(f'Pval:{PVAL}\nLval:{LVAL}')
    # print(f'{PVAL==LVAL}')


    #endregion


    
    