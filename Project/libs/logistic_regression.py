from __future__ import print_function
from Project.libs.utils import split_db_2to1,vcol,vrow
import numpy as np
import scipy
class LogRegClassifier:
    def __init__(self,D,L,one_fiftieth=False):
        (self.DTR, self.LTR), (self.DTE, self.LTE) = split_db_2to1(D,L)
        if one_fiftieth:
            self.DTR = self.DTR[:,::50]
            self.LTR = self.LTR[::50]
    
    def with_details(DTR, LTR, DTE, LTE) -> 'LogRegClassifier':
        """
        Custom constructor without the split_db_2to1
        """
        self = LogRegClassifier.__new__(LogRegClassifier)
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        return self
    def train(self,mode="binary",l=1,pT=0.5,expaded_feature=False):
        """
        Train a logistic regression model on a given dataset
        Args:
            mode: The mode of the logistic regression model. Either "binary" or "weighted". Default to "binary"
            l: The regularization parameter(lambda). Default to 1
            pT: For weighted logreg. The probability of the positive class. Default to 0.5
            expaded_feature: Whether to use the expanded feature set. Default to False
        Returns:
            The weights(w) and the bias(b) of the model
        """
        self.pT = pT
        if expaded_feature:
            self.DTR = LogRegClassifier.__quadratic_feature_expansion__(self.DTR)
            self.DTE = LogRegClassifier.__quadratic_feature_expansion__(self.DTE)
        match mode:
            case "binary":
                w,b=self.__logreg_binary__(self.DTR,self.LTR,l)
            case "weighted":
                w,b=self.__train_weighted_logreg_binary__(self.DTR,self.LTR,l,pT)
        self.w = w
        self.b = b
        return w,b

    @property
    def  logreg_scores(self):
        """
        Compute the logreg scores, that is w^Tx + b
        """
        return np.dot(self.w.T,self.DTE) + self.b
    @property
    def llr_scores(self):
        """
        Compute the log-likelihood ratio scores, that is log(p(x|H1)/p(x|H0))
        """
        empirical_prior = (self.LTR==1).sum()/self.LTR.size
        return self.logreg_scores - np.log(empirical_prior/(1-empirical_prior))

    def __logreg_binary__(self,D,L,l):
        """
        Train a binary logistic regression model
        Args:
            D: The training data
            L: The training labels
            l: The regularization parameter(lambda)
        Returns:
            The weights(w) and the bias(b) of the model
        """
        ZTR = self.LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
        def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
            w = v[:-1]
            b = v[-1]
            s = np.dot(vcol(w).T, self.DTR).ravel() + b

            loss = np.logaddexp(0, -ZTR * s)

            G = -ZTR / (1.0 + np.exp(ZTR * s))
            GW = (vrow(G) * self.DTR).mean(1) + l * w.ravel()
            Gb = G.mean()
            return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])
        vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(self.DTR.shape[0]+1))[0]
        # print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
        return vf[:-1], vf[-1]
    def __train_weighted_logreg_binary__(self,DTR, LTR, l, pT):
        """
        Train a binary logistic regression model with weighted classes
        Args:
            DTR: The training data
            LTR: The training labels
            l: The regularization parameter(lambda)
            pT: The probability of the positive class
        Returns:
            The weights(w) and the bias(b) of the model
        """

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
        # print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
        return vf[:-1], vf[-1] 
    def __quadratic_feature_expansion__(X):
        X_T = X.T
        X_expanded = []
        for x in X_T:
            outer_product = np.outer(x, x).flatten()
            expanded_feature = np.concatenate([outer_product, x])
            X_expanded.append(expanded_feature)
        X_expanded = np.array(X_expanded).T
        return X_expanded

