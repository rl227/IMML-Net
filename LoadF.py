'''
   This program is to load datasetW.mat data

   Code author: Shide Du
   Email: shidedums@163.com
   Date: Dec 4, 2019.

 '''

import scipy.io
import os
from loadMatData import loadData

### Load data with .mat
def loadF(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    F = data['F']
    #print("The size of data matrix is: ", features.shape)
    return F

def loadG(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    G = data['G']
    #print("The size of data matrix is: ", features.shape)
    return G

def loadGW(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    GW = data['YoutubeWG']
    #print("The size of data matrix is: ", features.shape)
    return GW

def loadFW(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    FW = data['Caltech10120WF']
    #print("The size of data matrix is: ", features.shape)
    return FW

def loadMFW(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MFW = data['Caltech101allMFW']
    #print("The size of data matrix is: ", features.shape)
    return MFW

def loadMGW(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MGW = data['scene15MGW']
    #print("The size of data matrix is: ", features.shape)
    return MGW

def loadMGW1(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MGW1 = data['G1']
    #print("The size of data matrix is: ", features.shape)
    return MGW1

def loadMGW2(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MGW2 = data['G2']
    #print("The size of data matrix is: ", features.shape)
    return MGW2

def loadMGW3(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MGW3 = data['G3']
    #print("The size of data matrix is: ", features.shape)
    return MGW3

def loadMGW4(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MGW4 = data['G4']
    #print("The size of data matrix is: ", features.shape)
    return MGW4

def loadMGW5(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MGW5 = data['G5']
    #print("The size of data matrix is: ", features.shape)
    return MGW5

def loadMGW6(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    MGW6 = data['G6']
    #print("The size of data matrix is: ", features.shape)
    return MGW6
if __name__ == '__main__':
    data_dir = "./_multiview datasets"
    dataW_dir = "./datasetW"
    dataGFW_dir = "./datasetGFW"

    features, gnd = loadData(os.path.join(data_dir, "ALOI.mat"))
    # W = loadSIM(os.path.join(dataW_dir, "ORLW.mat"))
    feature1 = features[0][0]
    print("The size of data matrix is: ", feature1.shape)
    F = loadFW(os.path.join(dataGFW_dir, "ALOIWF.mat"))
    F1 = F[0][0]
    G = loadGW(os.path.join(dataGFW_dir, "ALOIWG.mat"))
    G1 = G[0][0]
    print("The size of data matrix is: ", F1.shape)
    print("The size of data matrix is: ", G1.shape)

