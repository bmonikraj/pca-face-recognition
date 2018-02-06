# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 15:04:35 2018

@author: STUDENT1
"""


from PIL import Image
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
import os

#variance covariance matrix with rows as obseravtions and columns as features (mXd)
def covariance(X):
    d = X.shape[1]
    cov = np.zeros((d,d))
    def covar(U,V):
        Umean = np.mean(U)
        Vmean = np.mean(V)
        s = np.dot((U-Umean),(V-Vmean).T)
        s = s / len(U)
        return s
    for i in range(0,d):
        for j in range(0,d):
            cov[i][j] = covar(X[:,i], X[:,j])
    return cov
'''
#p = each person
main_dir = "C:/Users/MONIK/Desktop/ATNT Database/train"
main_dir_test = "C:/Users/MONIK/Desktop/ATNT Database/test"
IMAGES_PER_PERSON = 6
PERSONS = 10
HEIGHT = 112
WIDTH = 92
K = 10

def compute(main_dir, main_dir_test, IMAGES_PER_PERSON, PERSONS, HEIGHT, WIDTH, K):
    trainlabel = []
      
    inp = np.empty((HEIGHT*WIDTH*IMAGES_PER_PERSON,1))
    j=0
    for person in os.listdir(main_dir):
        trainlabel.append(j)
        j=j+1
        p_inp = np.array([])
        foldername = person
        person_dir = os.path.join(main_dir,foldername)
        for file in os.listdir(person_dir):
            file_path = os.path.join(person_dir,file)
            file_path = file_path.replace("\\","/")
            x = np.array(Image.open(file_path)).flatten()
            p_inp = np.append(p_inp,x)
        inp = np.concatenate((inp,p_inp.reshape(HEIGHT*WIDTH*IMAGES_PER_PERSON,1)),axis=1)
    inp = inp[:,1:PERSONS+1]
    
    mean = np.mean(inp,axis=1)
    zero_mean = inp - mean.reshape(mean.shape[0],1)
    sigma = covariance(zero_mean)
    
    eigval, eigvector = np.linalg.eig(sigma)
    idx = eigval.argsort()[::-1]   
    eigvalSort = eigval[idx]
    eigvectorSort = eigvector[:,idx]
    
    eigFaces = np.dot(eigvectorSort[:,0:K].T,zero_mean.T)
    signature = np.dot(eigFaces,zero_mean)
    
    
    #TESTING CASE
    acc=0 
    total=0
    j=0
    for person in os.listdir(main_dir_test):
        #testlabel.append(j)
        foldername = person
        person_dir = os.path.join(main_dir_test,foldername)
        for file in os.listdir(person_dir):
            total = total+1
            file_path = os.path.join(person_dir,file)
            file_path = file_path.replace("\\","/")
            xtest = np.array(Image.open(file_path)).flatten()
            xtest = np.tile(xtest,IMAGES_PER_PERSON)
            zero_meantest = xtest - mean
            psitest = np.dot(eigFaces,zero_meantest.reshape(zero_meantest.shape[0],1))
            
            TEST_PSI = psitest.T
            TEST_SIGNATURE = signature.T
            dis = []
            for i in range(0,PERSONS):
                s = np.power(np.dot((TEST_PSI-TEST_SIGNATURE[i]),(TEST_PSI-TEST_SIGNATURE[i]).T),0.5)
                dis.append(s)
            if j == dis.index(min(dis)):
                acc=acc+1
        j=j+1
    
    return K,float(acc/total)*100

xplot = []
yplot = []
for k in range(0,PERSONS):
    x,y=compute(main_dir, main_dir_test,IMAGES_PER_PERSON,PERSONS,HEIGHT,WIDTH,k)
    xplot.append(x)
    yplot.append(y)
plt.plot(xplot,yplot)
'''
'''
# p = each image
main_dir = "C:/Users/MONIK/Desktop/ATNT Database/train"
main_dir_test = "C:/Users/MONIK/Desktop/ATNT Database/test"
IMAGES_PER_PERSON = 6
PERSONS = 10
HEIGHT = 112
WIDTH = 92
K = 60

def compute(main_dir, main_dir_test, IMAGES_PER_PERSON, PERSONS, HEIGHT, WIDTH, K):  
    trainlabel = []
    MIN = 1000000000000000000000000.0
    inp = np.empty((HEIGHT*WIDTH,1))
    j=0
    for person in os.listdir(main_dir):
        trainlabel.append(j)
        j=j+1
        p_inp = np.array([])
        foldername = person
        person_dir = os.path.join(main_dir,foldername)
        for file in os.listdir(person_dir):
            file_path = os.path.join(person_dir,file)
            file_path = file_path.replace("\\","/")
            x = np.array(Image.open(file_path)).flatten()
            #p_inp = np.append(p_inp,x)
            inp = np.concatenate((inp,x.reshape(x.shape[0],1)),axis=1)
        #inp = np.concatenate((inp,p_inp.reshape(HEIGHT*WIDTH*IMAGES_PER_PERSON,1)),axis=1)
    inp = inp[:,1:IMAGES_PER_PERSON*PERSONS+1]
    
    mean = np.mean(inp,axis=1)
    zero_mean = inp - mean.reshape(mean.shape[0],1)
    sigma = covariance(zero_mean)
    
    eigval, eigvector = np.linalg.eig(sigma)
    idx = eigval.argsort()[::-1]   
    eigvalSort = eigval[idx]
    eigvectorSort = eigvector[:,idx]
    
    eigFaces = np.dot(eigvectorSort[:,0:K].T,zero_mean.T)
    signature = np.dot(eigFaces,zero_mean)
    
    
    #TESTING CASE
    acc=0 
    total=0
    j=0
    for person in os.listdir(main_dir_test):
        foldername = person
        person_dir = os.path.join(main_dir_test,foldername)
        for file in os.listdir(person_dir):
            total = total+1
            file_path = os.path.join(person_dir,file)
            file_path = file_path.replace("\\","/")
            xtest = np.array(Image.open(file_path)).flatten()
            #xtest = np.tile(xtest,IMAGES_PER_PERSON)
            zero_meantest = xtest - mean
            psitest = np.dot(eigFaces,zero_meantest.reshape(zero_meantest.shape[0],1))
            
            TEST_PSI = psitest.T
            TEST_SIGNATURE = signature.T
            dis = []
            for i in range(0,PERSONS*IMAGES_PER_PERSON):
                s = np.power(np.dot((TEST_PSI-TEST_SIGNATURE[i]),(TEST_PSI-TEST_SIGNATURE[i]).T),0.5)
                dis.append(s)
                if s<MIN:
                    MIN = s
            if int(j/4) == int(dis.index(min(dis))/6):
                acc=acc+1
            j=j+1
    
    #print("Accuracy:")
    #print(float(acc/total)*100)
    return MIN,K,float(acc/total)*100

xplot = []
yplot = []
for k in range(0,IMAGES_PER_PERSON*PERSONS):
    m,x,y=compute(main_dir, main_dir_test,IMAGES_PER_PERSON,PERSONS,HEIGHT,WIDTH,k)
    xplot.append(x)
    yplot.append(y)
plt.plot(xplot,yplot)

'''
########################################################################################
#Imposter detection
#m,x,y=compute(main_dir, main_dir_test,IMAGES_PER_PERSON,PERSONS,HEIGHT,WIDTH,7)

main_dir = "C:/Users/MONIK/Desktop/ATNT Database/train"
main_dir_imposter = "C:/Users/MONIK/Desktop/ATNT Database/imposter"
IMAGES_PER_PERSON = 6
PERSONS = 10
HEIGHT = 112
WIDTH = 92
K = 60

def computeImposter(main_dir, main_dir_imposter, IMAGES_PER_PERSON, PERSONS, HEIGHT, WIDTH, K):  
    inp = np.empty((HEIGHT*WIDTH,1))
    for person in os.listdir(main_dir):
        foldername = person
        person_dir = os.path.join(main_dir,foldername)
        for file in os.listdir(person_dir):
            file_path = os.path.join(person_dir,file)
            file_path = file_path.replace("\\","/")
            x = np.array(Image.open(file_path)).flatten()
            #p_inp = np.append(p_inp,x)
            inp = np.concatenate((inp,x.reshape(x.shape[0],1)),axis=1)
        #inp = np.concatenate((inp,p_inp.reshape(HEIGHT*WIDTH*IMAGES_PER_PERSON,1)),axis=1)
    inp = inp[:,1:IMAGES_PER_PERSON*PERSONS+1]
    
    mean = np.mean(inp,axis=1)
    zero_mean = inp - mean.reshape(mean.shape[0],1)
    sigma = covariance(zero_mean)
    
    eigval, eigvector = np.linalg.eig(sigma)
    idx = eigval.argsort()[::-1]   
    eigvalSort = eigval[idx]
    eigvectorSort = eigvector[:,idx]
    
    eigFaces = np.dot(eigvectorSort[:,0:K].T,zero_mean.T)
    signature = np.dot(eigFaces,zero_mean)
    
    yplot=[]
    xplot=[]
    xnum = 1
    #TESTING CASE
    for file in os.listdir(main_dir_imposter):
        file_path = os.path.join(main_dir_imposter,file)
        file_path = file_path.replace("\\","/")
        xtest = np.array(Image.open(file_path)).flatten()
        #xtest = np.tile(xtest,IMAGES_PER_PERSON)
        zero_meantest = xtest - mean
        psitest = np.dot(eigFaces,zero_meantest.reshape(zero_meantest.shape[0],1))
            
        TEST_PSI = psitest.T
        TEST_SIGNATURE = signature.T
        dis = []
        for i in range(0,PERSONS*IMAGES_PER_PERSON):
            s = np.power(np.dot((TEST_PSI-TEST_SIGNATURE[i]),(TEST_PSI-TEST_SIGNATURE[i]).T),0.5)
            dis.append(s) 
        yplot.append(min(dis)[0][0])
        xplot.append(xnum)
        xnum = xnum + 1
        
    #print("Accuracy:")
    #print(float(acc/total)*100)
    return xplot,yplot


xplot,yplot=computeImposter(main_dir, main_dir_imposter,IMAGES_PER_PERSON,PERSONS,HEIGHT,WIDTH,60)
plt.plot(xplot,yplot,'ro')