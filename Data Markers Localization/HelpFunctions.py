import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
import Parameters as parm



def Float(X):
    New=[]
    for el in X:
        New.append(float(el))
    return New
       
def IsBlackList(participant,mode='none'):
    #This function could work with both arguments or only with participant argument
    N=len(parm.BlackListPar)
    bool=1
    for i in range(N):
        if parm.BlackListPar[i]==participant and (parm.BlackListMode[i]==mode or mode=='none'):
            bool=0
            
    return bool
    

    
    
    
    
    
    