import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from Project.libs.dim_reduction import PCA,LDA
from Project.libs.utils import cov_m,cov_within_classes,cov_between_classes,vcol,vrow,split_db_2to1,load


def plot_scatter_nested(dataset,classes,n_label,labels):    
    for i in range(dataset.shape[0]):
        for j in range(i+1,dataset.shape[0]):
            plt.scatter(dataset[i][classes==0],dataset[j][classes==0],alpha=0.5,label=labels[0])
            plt.scatter(dataset[i][classes==1],dataset[j][classes==1],alpha=0.5,label=labels[1])
            plt.xlabel(f'Feature {i+1}')
            plt.ylabel(f'Feature {j+1}')
            plt.legend()
            # plt.savefig(f'scatter_plots/feature_{i+1}_vs_feature_{j+1}.png')
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

def Lab3():
    features,classes=load('Project/data/trainData.txt')
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
    cov_bt_cl = cov_between_classes(features,2,classes)
    cov_wt_cl = cov_within_classes(features,2,classes)
    LDA_matrix = LDA(cov_wt_cl,cov_bt_cl,1)

    proj_data = LDA_matrix@features
    plot_hist(proj_data,classes,2,['False','True'],20)
    #endregion

    #region LDA as classifier
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)

    cov_bt_cl = cov_between_classes(DTR,2,LTR)
    cov_wt_cl = cov_within_classes(DTR,2,LTR)
    DTR_lda_m = LDA(cov_wt_cl,cov_bt_cl,m)

    #Applico LDA al training set
    DTR_lda = DTR_lda_m@DTR
    #Applico LDA al validation set
    DVAL_lda = DTR_lda_m@DVAL

    # # computing threshold
    mean1 = DTR_lda[0,LTR==0].mean()
    mean2 = DTR_lda[0,LTR==1].mean()
    print(f'Mean True{mean1}:\nMean False:{mean2}')
    #Calcolo una soglia per predirre i valori in base alle medie del training set
    threshold = (mean1+mean2)/ 2.0
    predict_data( DVAL, LVAL,DVAL_lda,threshold)
    #endregion

    #region Changing Threshold
    # # Notando il grafico, vediamo il punto con meno overlap delle classi è 0, quindi tramite
    # # bruteforcing scelgo 0.0211 per migliorare l'accuracy
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
   
    cov_bt_cl = cov_between_classes(DTR,2,LTR)
    cov_wt_cl = cov_within_classes(DTR,2,LTR)
    DTR_lda_m = LDA(cov_wt_cl,cov_bt_cl,m)

    #Applico LDA al training set
    DTR_lda = DTR_lda_m@DTR
    #Applico LDA al validation set
    DVAL_lda = DTR_lda_m@DVAL
    threshold = 0.0211

    predict_data( DVAL, LVAL,DVAL_lda,threshold)
    #endregion
    
    #region PCA+LDA
    err_rates =[]
    for i in range(6):
        PCA_matrix = PCA(features,i+1)
        proj_data = PCA_matrix@features
        (DTR, LTR), (DVAL, LVAL) = split_db_2to1(proj_data, classes)

        cov_bt_cl = cov_between_classes(DTR,2,LTR)
        cov_wt_cl = cov_within_classes(DTR,2,LTR)
        DTR_lda_m = LDA(cov_wt_cl,cov_bt_cl,1)
        
        DTR_lda = DTR_lda_m@DTR
        DVAL_lda = DTR_lda_m@DVAL

        mean1 = DTR_lda[0,LTR==0].mean()
        mean2 = DTR_lda[0,LTR==1].mean()
        threshold = (mean1+mean2)/ 2.0
        err_rates.append(predict_data( DVAL, LVAL,DVAL_lda,threshold))
    print(err_rates)
    plt.plot(range(1,7),err_rates,marker="o")
    plt.xlabel("Number of dimensions while applying PCA")
    plt.ylabel("Error rate(%)")
    plt.tight_layout()
    plt.show()
    #endregion