import pylab as pb
import numpy as np
from math import pi
from scipy . spatial . distance import cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math

#Prior
#Create a GP-prior with a squared exponential co-variance function.
xdata=[]
x=np.arange(-math.pi,math.pi+0.1,0.05)
x=np.array(x)

priorMu=np.zeros(len(x))

def kernel(xi,xj,sigma,lengthscale):
    return (np.power(sigma,2)*np.exp(-np.power(xi-xj,2)/np.power(lengthscale,2)))

def plotSample(lengthscale,sigma):
    priorCov=np.mat(np.zeros((len(x), len(x))))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            priorCov[i,j]=kernel(x[i],x[j],sigma,lengthscale)
    priorsample = np.random.multivariate_normal(priorMu,priorCov,3)
    for prior in priorsample:
        for xi in x:
            plt.plot(x,prior)
            plt.plot(x,prior)
            plt.plot(x,prior)
    plt.show()

def createkernel(lengthscale, sigma, xi, xj):
    k = np.zeros((len(xi), len(xj)))
    
    for i in range(len(xi)):
        for j in range(len(xj)):
            k[i][j] = kernel(xi[i],xj[j],sigma,lengthscale)

    return k

def plotforinterval(mini,maxi,step,sigma,l,doplot=True):
    xnewList=[]
    postSampleList=[]
    postCovList=[]
    xwide=np.arange(mini,maxi,step)
    for xnew in xwide:
        xnew=[xnew]
        knewold=createkernel(l,sigma,xnew,x)
        koldnew=createkernel(l,sigma,x,xnew)
        knewnew=createkernel(l,sigma,xnew,xnew)
        koldold=createkernel(l,sigma,x,x)+np.power(sigma,2)*np.identity(len(x))

        postMu=np.dot(knewold,np.dot(np.linalg.inv(koldold),y))
        postCov=knewnew-np.dot(knewold,np.dot(np.linalg.inv((koldold)),koldnew))


        postSample = np.random.normal(postMu,postCov)
        xnewList.append(xnew)
        postSampleList.append(postSample)
        postCovList.append(postCov[0][0])
        if(doplot):
            plt.plot(xnew,postSample,'xg',xnew,postSample+postCov,"_g",xnew,postSample-postCov,"_g")
            plt.plot(x,y,'or')
    return xnewList,postSampleList,postCovList


def getPostSample(xnew,sigma,l):
    knewold=createkernel(l,sigma,xnew,x)
    koldnew=createkernel(l,sigma,x,xnew)
    knewnew=createkernel(l,sigma,xnew,xnew)
    koldold=createkernel(l,sigma,x,x)+np.power(sigma,2)*np.identity(len(x))

    postMu=np.dot(knewold,np.dot(np.linalg.inv(koldold),y))
    postCov=knewnew-np.dot(knewold,np.dot(np.linalg.inv((koldold)),koldnew))

    return np.random.multivariate_normal(postMu,postCov,7)

#Sample from this prior and visualise the samples
#Show samples using different length-scale for the squared exponential
#plotSample(0.1,1)
plotSample(0.5,1)
#plotSample(1,1)
#plotSample(1.5,1)

#Generate data
evec=[]
for i in range(0,len(x)):
    evec.append(np.random.normal(0, 0.5))
evec=np.array(evec)
y=np.sin(x)+evec

#Show distribution mean and std for points
sigma=1
l=1
xnewList,postSampleList,postCovList=plotforinterval(-5,5,0.2,1,2)
plt.show()

#Show samples of functions fitting the data
xnew=np.arange(-5,5,0.05)
postSample=getPostSample(xnew,1,2)
for sample in postSample:
    plt.plot(xnew,sample)
    plt.plot(x,y,'or')
plt.show()