import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

def scatter_plot(dataset,classes):
    for i in [0,2,4]:
        
        plt.subplot(1,3,int((i/2)+1))
        plt.xlabel(f'Feature {i+1}')
        plt.ylabel(f'Feature {i+2}')
        plt.scatter(dataset[i,classes==0],dataset[i+1,classes==0],alpha=0.5,label='True')
        plt.scatter(dataset[i,classes==1],dataset[i+1,classes==1],alpha=0.5,label='False')
        plt.tight_layout()
        plt.legend()
    plt.show()
    pass

def histogram(dataset,classes):
     #first I take all rows(first index), then I filter them based on the condition
     #"classes==0" basically maps the whole classes array values to either true or false based on the condition
     #then proceeds to scan both arrays(dataset and classes) in parallel and filter out from dataset where the value of classes is False
    true_feat = dataset[:,classes==0]
    false_feat = dataset[:,classes==1]
    for i in range(6):
    # for i in range(2):
        plt.subplot(3,2,i+1)
        # plt.subplot(1,2,(i+1))
        plt.xlabel(f'Feature {i+1}')
        plt.hist(true_feat[i],density=True,alpha=0.5,bins=10,label='True')
        plt.hist(false_feat[i],density=True,alpha=0.5,bins=10,label='False')
        plt.tight_layout()
        plt.legend()

    plt.show()

def mean(dataset):
    return dataset.mean(1).reshape(dataset.shape[0],1)


if __name__ == '__main__':
    features,classes=load('trainData.txt')

    mu_ds = mean(features)
    dc_ds = features - mu_ds

    features_true_12= features[0:2,classes==1]
    features_false_12= features[0:2,classes==0]

    print(f'Media True per feature 1 e 2:\n{mean(features_true_12)}')
    print(f'Media False per feature 1 e 2:\n{mean(features_false_12)}')
    print(f'Varianza True per feature 1 e 2:\n{features_true_12.var(1)}')
    print(f'Varianza False per feature 1 e 2:\n{features_false_12.var(1)}')
    
    features_true_34= features[2:4,classes==1]
    features_false_34= features[2:4,classes==0]

    print(f'Media True per feature 3 e 4:\n{mean(features_true_34)}')
    print(f'Media False per feature 3 e 4:\n{mean(features_false_34)}')
    print(f'Varianza True per feature 3 e 4:\n{features_true_34.var(1)}')
    print(f'Varianza False per feature 3 e 4:\n{features_false_34.var(1)}')

    histogram(features,classes)
    scatter_plot(features,classes)
    # print(features[0:2,classes==0].mean(1).reshape(features.shape[1],1))
    # print(features[0:2,classes==1].mean(1).reshape(features.shape[1],1))
   
