# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:14:52 2017

@author: khot
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y=np.array([[1],[1],[0]])

def sigmoid (x):
    return 1/(1 + np.exp(-x))
    
#Variable initialization
epoch=5000 #Setting training iterations
inputlayer_neurons = x.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))





for i in range(epoch):
    #forwrd propogation
    #step1 : z2 = xw(1)
    z2 = np.dot(x,wh)
    z2 = z2+bh
     #step2 : a2= f(z2)
    a2 = sigmoid(z2)
     #step3 : z3 = a2w(2)
    z3=np.dot(a2 ,wout)
    z3 = z3+bout
     #step4 : yHat = f(z3)
    yHat = sigmoid(z3)
    

print(yHat)


plt.bar([0,1,2] , y , alpha=0.8 , width=0.35)
plt.bar([0.35,1.35,2.35] , yHat , alpha=0.8 , width=0.35 , color='red')
plt.show()



