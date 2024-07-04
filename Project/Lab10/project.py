from Project.libs.utils import load
from Project.libs.gmm import *
from Project.libs.bayes_risk import get_dcf,get_min_dcf

def gmm_scores(gmm0,gmm1,DTE):
    scores0 = logpdf_GMM(DTE, gmm0)
    scores1 = logpdf_GMM(DTE, gmm1)
    return scores1 - scores0
def Lab10():
    (features,classes) = load("project/data/trainData.txt")
    (DTR, LTR), (DTE, LTE) = split_db_2to1(features, classes)
    prior = 0.5
    #region Full GMM
    #FIXME: with 32 it crashes when trying to invert the covariance matrix,
    for covType in ['full','diagonal']:
        print(f'{covType} GMM')
        for numC in [1,2,8,16]:
            gmm0 = train_GMM_LBG_EM(DTR[:,LTR==0],numC,covType,verbose=False)
            gmm1 = train_GMM_LBG_EM(DTR[:,LTR==1],numC,covType,verbose=False)

            scores = gmm_scores(gmm0,gmm1,DTE)
            dcf = get_dcf(scores,LTE,prior,1.0,1.0,normalized=True,threshold='optimal')
            min_dcf = get_min_dcf(scores,LTE,prior,1.0,1.0)

            print("Number of components: ",numC)
            print("DCF: ",dcf)
            print("MinDCF: ",min_dcf)
            print()
    
