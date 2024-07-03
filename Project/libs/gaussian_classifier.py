from __future__ import print_function
from Project.libs.utils import split_db_2to1,cov_m,cov_within_classes,vcol,vrow
from Project.libs.bayes_risk import compute_optimal_Bayes_binary_threshold
import numpy as np
class GaussianClassifier:
    def __init__(self,D,L):
        (self.DTR, self.LTR), (self.DTE, self.LTE) = split_db_2to1(D,L)
        self.covMatrices = []
    def train(self,mode="mvg"):
        # disable print
   

        self.covMatrices = []
        self.predicted_labels = []
        self.SJoint = []
        self.mode = mode
        if mode=="mvg":
            self.__mvg__()
        elif mode=="naive":
            self.__naive__()
        elif mode=="tied":
            self.__tied__()



    def evaluate(self,**kwargs):
        """
        Evaluate the classifier using the test set.
        It also sets both LLRs and the accuracy of the classifier.
        Args:
            threshold: The threshold to use for the classifier. If 'optimal' is passed, the optimal threshold is computed. Default to 0
            prior: If optimal threshold is requested, the prior probability of the positive class. Default to 0.5
            Cfn: If optimal threshold is requested, the cost of false negatives. Default to 1.0
            Cfp: If optimal threshold is requested, the cost of false positives. Default to 1.0
        Returns:
            The predicted labels and the accuracy of the classifier
        """
        threshold = kwargs.get('threshold',0)
        if threshold == 'optimal':
            prior = kwargs.get('prior',0.5)
            Cfn = kwargs.get('Cfn',1.0)
            Cfp = kwargs.get('Cfp',1.0)
            threshold = compute_optimal_Bayes_binary_threshold(prior,Cfn,Cfp)

        self.LLR = np.log(self.SJoint[1]/self.SJoint[0])
        PVAL = np.zeros(shape=self.LTE.shape, dtype=np.int32)
        PVAL[self.LLR >= threshold] = 1
        PVAL[self.LLR < threshold] = 0

        myarr = PVAL==self.LTE
        correct_pred =  myarr.sum()
        print(f'Correct predictions: {correct_pred}')
        false_pred = myarr.size - correct_pred
        self.accuracy = correct_pred/myarr.size*100
        print(f'False predictions: {false_pred}')
        print(f'Error rate: {false_pred/myarr.size*100}% over {myarr.size} samples.')
        return PVAL, self.accuracy

    def get_cov_matrices(self):
        return self.covMatrices
    def __get_llr_mvg__(self):
        return np.log(self.SJoint[1]/self.SJoint[0])
    def __get_llr_tied_naive__(self):
        return self.SJoint[1]-self.SJoint[0]

    def get_accuracy(self):
        return self.accuracy
    def get_err_rate(self):
        return 100-self.accuracy

    def __mvg__(self):
        labels = np.unique(self.LTE)
        SJoint=np.zeros((len(labels), self.DTE.shape[1]))
        cov_arr=[]
        for i,l in enumerate(labels):
            class_sample_l = self.DTR[:,self.LTR==l]
            cov = cov_m(class_sample_l)
            cov_arr.append(cov)
            mu = np.mean(class_sample_l,axis=1,keepdims=True)
            SJoint[i,:] =np.exp(self.__logpdf_GAU_ND__(self.DTE,mu,cov))*(1/2)
        SMarginal =  vrow(SJoint.sum(0))
        SPost = SJoint / SMarginal
        predicted_labels = np.argmax(SPost,axis=0)
        self.SJoint = SJoint
        self.covMatrices = cov_arr
        self.predicted_labels = predicted_labels
        # return SJoint,np.array(cov_arr),predicted_labels


    def __naive__(self):
        classes = np.unique(self.LTE)
        SJoint=np.zeros((len(classes), self.DTE.shape[1]))
        for i,l in enumerate(classes):
            class_sample_l = self.DTR[:,self.LTR==l]
            # cov = np.cov(class_sample_l,bias=True)
            cov = cov_m(class_sample_l)
            cov = np.diag(np.diag(cov))
            mu = np.mean(class_sample_l,axis=1,keepdims=True)
            SJoint[i,:] =np.exp(self.__logpdf_GAU_ND__(self.DTE,mu,cov))*(1/2)
        SMarginal =  vrow(SJoint.sum(0))
        SPost = SJoint / SMarginal
        predicted_labels = np.argmax(SPost,axis=0)
        self.SJoint = SJoint
        self.predicted_labels = predicted_labels
        # return SJoint,predicted_labels
    def __tied__(self):
        classes = np.unique(self.LTE)
        SJoint=np.zeros((len(classes), self.DTE.shape[1]))
        cov = cov_within_classes(self.DTR,2,self.LTR)
        for i,l in enumerate(classes):
            class_sample_l = self.DTR[:,self.LTR==l]
            mu = np.mean(class_sample_l,axis=1,keepdims=True)
            SJoint[i,:] =np.exp(self.__logpdf_GAU_ND__(self.DTE,mu,cov))*(1/2)
        SMarginal =  vrow(SJoint.sum(0))
        SPost = SJoint / SMarginal
        predicted_labels = np.argmax(SPost,axis=0)
        self.SJoint = SJoint
        self.predicted_labels = predicted_labels
        # return SJoint,predicted_labels
    def  __logpdf_GAU_ND__(self,x,mu,C):
        P = np.linalg.inv(C)
        M = x.shape[0]#number of features
        return -0.5*M*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu)*( P @ (x-mu))).sum(0)