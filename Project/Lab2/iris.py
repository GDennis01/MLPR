import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load(filename):
    records=[[],[],[],[]]
    dicClasses={
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

            classes.append(dicClasses[fields[4].strip()])
        a = np.array(records)
        classes=np.array(classes)
        return a,classes


if __name__== '__main__':
    features,classes=load("Lab2\iris.csv")
    sepal_len=features[0,:]
    sepal_wdt=features[1,:]
    petal_len=features[2,:]
    petal_wdt=features[3,:]
    for i in range(3):
        plt.hist(sepal_len[classes==i],alpha=0.4,bins=10,density=True)
    plt.show()
    # for i in range(3):
    #     plt.hist(sepal_wdt[classes==1],alpha=0.4,bins=20,density=True)
    # plt.show()
    # for i in range(3):
    #     plt.hist(petal_len[classes==1],alpha=0.4,bins=20,density=True)
    # plt.show()
    # for i in range(3):
    #     plt.hist(petal_wdt[classes==1],alpha=0.4,bins=20,density=True)
    # plt.show()
    
    