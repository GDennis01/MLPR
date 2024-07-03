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

if __name__ == '__main__':
    features,classes=load("trainData.txt")

    for i in range(features.shape[0]):
        print(i)
        # class 0
        feat_i = features[i,classes == 0]
        plt.figure()
        plt.subplot(1,2,1)
        plt.hist(feat_i,bins=20,density=True)
        feat_i=vcol(feat_i)

        m = feat_i.mean(0)
        m = np.ones((1,1)) * m
        var = np.var(feat_i)
        C = np.ones((1,1)) * var

        x = np.linspace(feat_i.min(),feat_i.max(),100)
        y = logpdf_GAU_ND(vrow(x),m,C)
     
        plt.xlabel(f'Feature {i+1} Class 0')
        mu_ml = feat_i.mean()
        var_ml = np.var(feat_i)
        print(f'Feature {i+1} of class 0: ML mu: {mu_ml} var: {var_ml}')

        plt.plot(x,np.exp(y))

        # class 1
        plt.subplot(1,2,2)
        feat_i = features[i,classes == 1]
        plt.hist(feat_i,bins=20,density=True)
        feat_i=vcol(feat_i)

        m = feat_i.mean(0)
        m = np.ones((1,1)) * m
        var = np.var(feat_i)
        C = np.ones((1,1)) * var

        x = np.linspace(feat_i.min(),feat_i.max(),100)
        y = logpdf_GAU_ND(vrow(x),m,C)
        plt.xlabel(f'Feature {i+1} Class 1')

        mu_ml = feat_i.mean()
        var_ml = np.var(feat_i)
        print(f'Feature {i+1} of class 1: ML mu: {mu_ml} var: {var_ml}')
        plt.plot(x,np.exp(y))

        plt.show()