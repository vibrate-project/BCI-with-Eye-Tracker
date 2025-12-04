# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:20:31 2024

@author: teodo
"""

# import tensorflow as tf
import ReadData as r
import Frequency as freq
import SegmentsPositions as seg
import pickle
import os
import math
import numpy as np
import HelpFunctions as h
from multiprocessing import Pool
from functools import partial
from itertools import repeat


modes=['negative','positive']
NLayers=3
NSegments=21
NameData='Pickle/ParData_LearningPeople'+'.pkl'
# NameData='Pickle/data'+'.pkl'  
# NameData='Pickle/data_TestPeople'+'.pkl' 


 
# # participant / mode / segment / XY
# with open(NameData, 'rb') as f:
# # with open('Pickle/data'+'.pkl', 'rb') as f:
#     Data=pickle.load(f)
# NPeople=len(Data)
# [Negative,Positive]=GetStates()
# N=len(Negative[0])



def GetStates():
    with open(NameData, 'rb') as f:
        Data=pickle.load(f)
    NPeople=len(Data)
    Negative=[]
    Positive=[]
    
    for p in range(NPeople):
        neg=[]
        pos=[]
        for s in range(len(Data[p][0])):
            neg=neg+Data[p][0][s]
            pos=pos+Data[p][1][s]
        Negative.append(neg)
        Positive.append(pos)
    return [Negative,Positive]

GetStates()
def sigm(x):
    y=1/(1+np.exp(-x))
    return y

def Model(Input,Weights):
    # NW=len(weights)
    L0=len(Input)
    L1=int(L0/3)
    Connect0=20
    step0=int((L0-Connect0)/L1)
    
    
    
    
    WCounter=0
    Layer1=[]
    for i in range(L1):
        value=0
        for j in range(Connect0):
            value=value+(Weights[WCounter]*Input[j+i*step0])
            WCounter=WCounter+1
        value=sigm(value)
        Layer1.append(value)
    L2=int(L1/6)
    Connect1=12
    step1=int((L1-Connect1)/L2)
    # print('i:', i)
    Layer2=[]
    for i in range(L2):
        value=0
        for j in range(Connect1):
            value=value+Weights[WCounter]*Layer1[j+i*step1]
            WCounter=WCounter+1
        value=sigm(value)
        Layer2.append(value)
    # print('i:', i)



    L3=1
    Connect2=L2
    step2=int((L2-Connect2)/L3)
    
    Layer3=[]
    for i in range(L3):
        value=0
        for j in range(Connect2):
            value=value+Weights[WCounter]*Layer2[j+i*step2]
            WCounter=WCounter+1
        value=sigm(value)
        Layer3.append(value)
    # print('i:', i)    

    Output=Layer3[0]
    return Output



def Err(Weights,segment=1):
    
    [Negative,Positive]=GetStates()

    N=len(Negative[0])
    

    
    PosRes=[]
    NegRes=[]
    N=len(Negative)
    for i in range(N):
        NegRes.append(Model(Negative[i],Weights))
        PosRes.append(Model(Positive[i],Weights))
    # with Pool() as pool:
    #     par = pool.starmap(Model, zip(Negative + Positive, repeat(Weights)))
        
    # NegRes = par[:N]
    # PosRes = par[N:]

    Err=0
    for i in range(N):
        Err=Err+NegRes[i]-PosRes[i]
        
    return Err



def Grad(Weights,segment):
    
    eps=0.000001
    N=len(Weights)
    # def func(i):
    #     return GradErrHelp(Weights,i,eps)
    
    with Pool() as pool:        
         # func = partial(GradHelp, Weights=Weights, eps=eps)
         Grad = pool.starmap(GradHelp, zip(repeat(Weights), range(N), repeat(eps),repeat(segment)))
    return Grad
    
    


def GradHelp(Weights,i,eps,segment):
    Weights[i]=Weights[i]+eps
    right=Err(Weights,segment)
    Weights[i]=Weights[i]-2*eps
    left=Err(Weights,segment)
    gra=(right-left)/(2*eps)
    return gra

# Input=Negative[0]

# L0=len(Input)
# L1=int(L0/20)
# Connect0=120
# L2=int(L1/6)
# Connect1=12
# L3=1
# Connect2=L2
# Weights=[0.5]*(L1*Connect0+L2*Connect1+L3*Connect2)



def FindWeights(segment=0):
    
    LearningRateStart=10**(5)
    LearningRateEnd=10**(-10)
                         
  
    NPowersteps=100
    NSteps=50
    

    
    [Negative,Positive]=GetStates()
    
    
    Input=Negative[0]
    L0=len(Input)
    L1=int(L0/3)
    Connect0=20
    L2=int(L1/6)
    Connect1=12
    L3=1
    Connect2=L2
    Weights=[0.5]*(L1*Connect0+L2*Connect1+L3*Connect2)
    Weights=h.Normalise(Weights)
    
    LRStep=(LearningRateStart-LearningRateEnd)/NSteps
    LearningRate=LearningRateStart+LRStep
    for i in range(NSteps):
        LearningRate=LearningRate-LRStep
        grad=Grad(Weights,segment)
        
        
        
        DirName='MiddleAdaptivePar'
        MainDir=os.getcwd()    
        os.makedirs(DirName,exist_ok=True)
        os.chdir('./'+DirName)
        with open('grad'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(grad, f)
        os.chdir(MainDir)
        
        DirName='MiddleAdaptivePar'
        MainDir=os.getcwd()    
        os.makedirs(DirName,exist_ok=True)
        os.chdir('./'+DirName)
        with open('Weights'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(Weights, f)
        os.chdir(MainDir)
        
        for j in range(len(grad)):
            grad[j]=-grad[j]*LearningRate
        
        
        
        ErrH=10**10
        
        errOld=10**10
        errNew=10**10
        for k in range(NPowersteps):
            NewWeights=Weights.copy()
            for j in range(len(grad)):
                NewWeights[j]=Weights[j]+grad[j]*(1/2)**(k)
            NewWeights=h.Normalise(NewWeights)
            errNew=Err(NewWeights,segment)
            
            if errNew>errOld:
                for j in range(len(grad)):
                    Weights[j]=Weights[j]+grad[j]*(1/2)**(k-1)
                Weights=h.Normalise(Weights)   
                Error=errOld
                WeightsH=Weights.copy()
                break
            else:
                WeightsH=NewWeights.copy()
                errOld=errNew       
        print(errNew)    
        if ErrH-errNew<0.00000001:
            print("Good News")
            break
        ErrH=errNew
        
    return Weights





[Negative,Positive]=GetStates()
N=len(Negative[0])
DirName='MiddleAdaptivePar'
MainDir=os.getcwd()    
os.chdir('./'+DirName)
i=40
with open('Weights'+str(i)+'.pkl', 'rb') as f:
    WWW=pickle.load(f)
os.chdir(MainDir)
sum=0
N=len(Negative)
for p in range(N):
    sum=sum+Model(Positive[p],WWW)+Model(Negative[p],WWW)
sum=sum/N/2

for p in range(N):
    print(Model(Positive[p],WWW)-sum>0,Model(Negative[p],WWW)-sum<0)

for p in range(N):
    print(Model(Positive[p],WWW)-Model(Negative[p],WWW))

for i in range(50):
    MainDir=os.getcwd()    
    os.chdir(DirName)
    with open('Weights'+str(i)+'.pkl', 'rb') as f:
        WWW=pickle.load(f)
    os.chdir(MainDir)
    print(Err(WWW))
    


# with open(NameData, 'rb') as f:
#     Data=pickle.load(f)    
# Data[p][0][s][0]+Data[p][0][s][1]

