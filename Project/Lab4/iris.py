import matplotlib.pyplot as plt
import numpy as np


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
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))


if __name__ == '__main__':
    #region Plotting density
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    print(XPlot.shape)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    print(C)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    plt.show()
    #endregion

    #region checking if density is equal
    # XPlot = np.linspace(-8, 12, 1000)
    # m = np.ones((1,1)) * 1.0
    # C = np.ones((1,1)) * 2.0
    # pdfSol = np.load('llGAU.npy')
    # pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    # print(np.abs(pdfSol - pdfGau).max())
    #endregion

    #region Maximum Likelihood Estimate
    x = np.load('XND.npy')
    print(x.shape)
    m_ML = vcol(x.mean(1))
    C_ML = np.cov(x,bias=True)

    print(loglikelihood(x,m_ML,C_ML))
    #endregion
