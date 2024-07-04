import numpy as np
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
def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C
    
def cov_m(D):
    mu = D.mean(1)
    Dc = D - vcol(mu)
    return Dc@Dc.T/D.shape[1]
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