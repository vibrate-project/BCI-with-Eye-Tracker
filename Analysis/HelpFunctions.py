import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy



def Float(X):
    New=[]
    for el in X:
        New.append(float(el))
    return New
       

    
    
def Sort(X):
    # IncreasingOrder
    N=len(X)
    Xsorted=X.copy()
    for i in range(N):
        for j in range(i+1,N):
            if Xsorted[i]>Xsorted[j]:
                swap=Xsorted[i]
                Xsorted[i]=Xsorted[j]
                Xsorted[j]=swap
                
    return Xsorted
            
def Normalise(X):
    N=len(X)
    sum=0
    for i in range(N):
        sum=sum+X[i]**2
    sum=sum**(1/2)
    for i in range(N):
        X[i]=X[i]/sum
    return X
    
    
    
    
    
    
def f1(a,b):
    return a+b

def f2(x,y):
    r=5
    def f3(c):
        return c+r
    return f3(x)+y
    
