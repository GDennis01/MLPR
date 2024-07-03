from Project.libs.utils import split_db_2to1,cov_m,cov_within_classes,vcol,vrow
import numpy as np
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