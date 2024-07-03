import matplotlib.pyplot as plt
import numpy as np
import os

# from ..libs.gaussian_classifier import GaussianClassifier
# from .libs.utils import load,vcol,vrow
# from ..libs.dim_reduction import PCA
from Project.libs.gaussian_classifier import GaussianClassifier
from Project.libs.utils import load,vcol,vrow
from Project.libs.dim_reduction import PCA
def heatmap(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, '%.2f' % data[i, j],
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5),
                     fontsize=8)
    plt.imshow(data)
    plt.show()
def Lab5():
    features,classes=load("Project/data/trainData.txt")

    gc = GaussianClassifier(features,classes)

    #region MVG classifier
    print("MVG R:")
    gc.train("mvg")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")
    #endregion

    #region Naive Bayes classifier
    print("Naive R:")
    gc.train("naive")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")
    #endregion

    #region Tied Covariance classifier
    print("Tied MVG R:")
    gc.train("tied")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")
    #endregion

    #region Pearson Correlation
    gc.train("mvg")
    covs = gc.covMatrices
    heatmap(covs[0])
    heatmap(covs[1])
    Corr_false = covs[0] / ( vcol(covs[0].diagonal()**0.5) * vrow(covs[0].diagonal()**0.5) )
    Corr_true = covs[1] / ( vcol(covs[1].diagonal()**0.5) * vrow(covs[1].diagonal()**0.5) )
    heatmap(Corr_false)
    heatmap(Corr_true)
    #endregion

    #region Naive bayes features 1-4
    features_1_4 = features[0:4,:]
    gc = GaussianClassifier(features_1_4,classes)
    gc.train("naive")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")
    #endregion

    #region Tied Covariance features 1-4
    features_1_4 = features[0:4,:]
    gc = GaussianClassifier(features_1_4,classes)
    gc.train("tied")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")
    #endregion

    #region MVG features 1-4
    features_1_4 = features[0:4,:]
    gc = GaussianClassifier(features_1_4,classes)
    gc.train("mvg")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")
    #endregion

    #region MVG and Tied pair 1-2 3-4
    print("MVG 1-2 ")
    features_1_2 = features[0:2,:]
    gc = GaussianClassifier(features_1_2,classes)
    gc.train("mvg")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")

    print("MVG 2-4 ")
    features_2_4 = features[2:4,:]
    gc = GaussianClassifier(features_2_4,classes)
    gc.train("mvg")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")

    print("Tied 1-2 ")
    features_0_2 = features[0:2,:]
    gc = GaussianClassifier(features_0_2,classes)
    gc.train("tied")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")

    print("Tied 2-4 ")
    features_2_4 = features[2:4,:]
    gc = GaussianClassifier(features_2_4,classes)
    gc.train("tied")
    gc.evaluate()
    acc = gc.get_accuracy()
    print(f"Accuracy: {acc}%")
    #endregion

    #region PCA pre-processing
    accuracies_mvg = []
    accuracies_tied = []
    accuracies_naive = []
    for i in range(1,7):
        # m=i+1
        m=i
        print(f"PCA with {m} dimensions")
        proj_data = PCA(features,m)@features

        gc = GaussianClassifier(proj_data,classes)
        print("MVG:")
        gc.train("mvg")
        gc.evaluate()
        acc = gc.get_accuracy()
        accuracies_mvg.append(acc)

        print("\n\nTied MVG:")
        gc.train("tied")
        gc.evaluate()
        acc = gc.get_accuracy()
        accuracies_tied.append(acc)

        print("\n\nNaive Bayes:")
        gc.train("naive")
        gc.evaluate()
        acc = gc.get_accuracy()
        accuracies_naive.append(acc)
    print(accuracies_mvg)
    print(accuracies_tied)
    print(accuracies_naive) 
    plt.plot(range(1,7),accuracies_mvg,marker="o",label="MVG")
    plt.plot(range(1,7),accuracies_naive,marker="o",label="Naive Bayes")
    plt.plot(range(1,7),accuracies_tied,marker="o",label="Tied MVG")
    plt.xlabel("Number of features")
    plt.xticks(range(1,7))
    plt.ylabel("Accuracy(%)")
    plt.tight_layout()
    plt.legend()
    plt.show()
    #endregion
