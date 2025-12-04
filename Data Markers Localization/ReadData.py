import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
import HelpFunctions as h




def ReadCoordinates(participant,mode):
    
    filenameImport='Data/'+'Participant'+str(participant)+'_'+mode+'.csv'
    FullData= list(csv.reader(open(filenameImport)))
    FullData.pop(0)
    N=len(FullData)
    FullData=np.transpose(FullData)
    X=FullData[6-1].copy()
    X=h.Float(X)
    Y=FullData[7-1].copy()
    Y=h.Float(Y)
    T=FullData[4-1].copy()
    T=h.Float(T)
    
    
    [X,Y,T]=MakeSignalEvenFrequency(X,Y,T,120)
    [X,Y]=PutIntoFrame(X,Y)
    
    
    
    return[X,Y,T] 
        



# Takes all variables above 1 and below 0 and puts them into the interval
def PutIntoFrame(X_original,Y_original):
    # Gives the rate of outliers
    OutlierRate=0.07
    
    N=len(X_original)
    Out=(N*OutlierRate) // 1
    Out=int(Out)
    
   
    
    X=X_original.copy()
    Y=Y_original.copy()
    
    
    
    # Nh=N-1000
    X[N-1]=0.5
    Y[N-1]=0.5
    Nh=N
    
    
    #Finds the points when the participant is blinking and approximates the movement there linearly
    for i in range(Nh):
        if X[i]==0:
            l=0
            while X[i+l]==0:
                l=l+1
            step=(X[i+l]-X[i-1])/l
            for j in range(l):
                X[i+j]=X[i-1]+step*j
            
    for i in range(Nh):
        if Y[i]==0:
            l=0
            while Y[i+l]==0:
                l=l+1
            step=(Y[i+l]-Y[i-1])/l
            for j in range(1,l+1):
                Y[i+j]=Y[i-1]+step*j
            
    
    #Corrects the orientation
    for i in range(N):
        Y[i]=1-Y[i]
        
    
    
    
    
    # Clean the outliers
    
    Operations=10
    portion=(Out/Operations) // 1
    portion=int(portion)
    

    for oper in range(Operations):
        
        Xmean=0
        for i in range(N):
            Xmean=Xmean+X[i]**2
            
        Xmean=(Xmean/N)**(1/2)
        Xdeviation=X.copy() 
        for i in range(N):
            Xdeviation[i]=abs(Xdeviation[i]-Xmean)
        XdeviationSort=Xdeviation.copy()
        list.sort(XdeviationSort)
        thresshold=XdeviationSort[N-portion]    
        del XdeviationSort
        for i in range(N):
            if Xdeviation[i]>thresshold:
                l=0
                while Xdeviation[i+l]>thresshold:
                    l=l+1
                step=(X[i+l]-X[i-1])/l
                for j in range(l):
                    X[i+j]=X[i-1]+step*j 
        
     
        
     
        
        
        
        
        
        Ymean=0
        for i in range(N):
            Ymean=Ymean+X[i]**2
            
        Ymean=(Ymean/N)**(1/2)
        Ydeviation=Y.copy() 
        for i in range(N):
            Ydeviation[i]=abs(Ydeviation[i]-Ymean)
        YdeviationSort=Ydeviation.copy()
        list.sort(YdeviationSort)
        thresshold=YdeviationSort[N-portion]    
        del YdeviationSort
        for i in range(N):
            if Ydeviation[i]>thresshold:
                l=0
                while Ydeviation[i+l]>thresshold:
                    l=l+1
                step=(Y[i+l]-Y[i-1])/l
                for j in range(l):
                    Y[i+j]=Y[i-1]+step*j
    
    
    
    
    
    

    
    #Adjust the interval
    Xmin=X[0]
    Xmax=X[0]
    for i in range(N):
        if Xmin>X[i]:
            Xmin=X[i]
        if Xmax<X[i]:
            Xmax=X[i]
    Xinterval=Xmax-Xmin
    for i in range(N):
        X[i]=X[i]/Xinterval
    
    Xmin=Xmin/Xinterval
    for i in range(N):
        X[i]=X[i]-Xmin
        
        
    
    Ymin=Y[0]
    Ymax=Y[0]
    for i in range(N):
        if Ymin>Y[i]:
            Ymin=Y[i]
        if Ymax<Y[i]:
            Ymax=Y[i]
    Yinterval=Ymax-Ymin
    for i in range(N):
        Y[i]=Y[i]/Yinterval
    
    Ymin=Ymin/Yinterval
    for i in range(N):
        Y[i]=Y[i]-Ymin
        
        
    #Translate the interval into [0;1]
    
    
        
        
    # for i in range(len(X)):
    #     if X[i]>1:
    #         X[i]=1    
    # for i in range(len(Y)):
    #     if Y[i]>1:
    #         Y[i]=1
    # for i in range(len(X)):
    #     if X[i]<0:
    #         X[i]=0    
    # for i in range(len(Y)):
    #     if Y[i]<0:
    #         Y[i]=0
            
   
    
    return [X,Y]


def MakeSignalEvenFrequency(X,Y,T,Frequency=120):
    Duration=T[-1]
    
    X_e=MakeSignalEvenHelp(X,T,Duration,Frequency)
    Y_e=MakeSignalEvenHelp(Y,T,Duration,Frequency)
    
    N=math.floor(Duration*Frequency)
    T_step=1/Frequency
    
    T_e=[]
    for i in range(N):
        T_e.append(i*T_step)
    
    return [X_e,Y_e,T_e]



    
def MakeSignalEvenHelp(X,T,Dur,Freq):
    N=math.floor(Dur*Freq)
    T_step=1/Freq
    
    X_e=[]
    time=0
    state=1
    for i in range(N):
        
        leftDist=time-(T[state-1]-T[0])
        rightDist=(T[state]-T[0])-time
        
        p=X[state-1]*rightDist/(leftDist+rightDist)+X[state]*leftDist/(leftDist+rightDist)
        X_e.append(p)
        
        time=time+T_step
        
        for j in range(state-1,len(T)):
            if T[j]-T[0]>time:
                state=j
                break
            
        
    return X_e




def ReadFile(Name):
    
    fileName='Data/'+Name+'.csv'
    FullData= list(csv.reader(open(fileName)))
    
    
    
    return FullData
    