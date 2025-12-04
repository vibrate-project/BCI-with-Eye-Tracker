import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
import HelpFunctions as h
import Parameters as parm
import Frequency as freq
import ReadDataSynthetic as rs






def ReadSegments(participant,mode):
    filenameImport=filenameImport='Data/MovingPoints/'+'points_'+str(participant)+str(mode)+'.csv'
    Marks= list(csv.reader(open(filenameImport)))
    
    [X,Y,T]=rs.ReadCoordinateSynthetic(participant,mode)
    
    Durations=[parm.DurationFirstPoints,parm.DurationFirstPoints]
    for t in parm.SquareTimes:
        Durations.append(t)
    
    N=len(Durations)
    Segments=[]
    
    for i in range(N):
        Xp=[]
        Yp=[]
        Tp=[]
        startTime=Marks[i][parm.Time_index]
        start=int(Marks[i][parm.iIndex])
        maxDur=Durations[i]
        dur=0
        j=0
        while dur<maxDur:
            Xp.append(X[start+j])
            Yp.append(Y[start+j])
            Tp.append(T[start+j])
            dur=Tp[j]-T[start]
            j=j+1
            
        Segments.append([Xp,Yp,Tp])
        
    return Segments
            
    



def ReadMovingPoints(participant,mode):
    
    filenameImport='Data/MovingPoints/'+'points_'+str(participant)+mode+'.csv'
    FullData= list(csv.reader(open(filenameImport)))
    N=len(FullData)
    
    Points=[]
    # [StartPoint, EndPoint, n, Time]
    for i in range(N):
        
        data=FullData[i]
        # Horizontal movement
        if i%2==0:
            StartPoint=[data[parm.FirstCoordinateIndex],data[parm.StaticCoordinateIndex]]
            EndPoint=[data[parm.SecondCoordinateIndex],data[parm.StaticCoordinateIndex]]

        # Vertical movement
        if i%2==1:
            StartPoint=[data[parm.StaticCoordinateIndex],data[parm.FirstCoordinateIndex]]
            EndPoint=[data[parm.StaticCoordinateIndex],data[parm.SecondCoordinateIndex]]   
            
        n=data[parm.iIndex]
        time=data[parm.Time_index]
        Points.append([StartPoint,EndPoint,n,time])
    
    return Points
    
# ppp=ReadMovingPoints(1,'negative')    
    

    
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
    

    
    # [X,Y]=PutIntoFrame(X,Y)
    [X,Y]=PutIntoFrame(X,Y)
    
    
    
    return[X,Y,T] 
        


def PutIntoFrame(X_original,Y_original):
# Takes all variables above 1 and below 0 and puts them into the interval

    # Gives the rate of outliers
    OutlierRate=0.05
    
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



def ReadImportantParts(participant,mode):
    
    [X,Y,T]=rs.ReadCoordinatesSynthetic(participant,mode)
    Mpoints=ReadMovingPoints(participant,mode)
    
    durations=[12,15,15,15,15,15,7,10,10,7,5,7,5,7,30,7,30,7,20,7,40]
    
    index=int(Mpoints[2][2])
    ImportantIndecies=[index]
    zeroTime=T[index]
    begin=9
    b=begin-1
    for i in range(begin):
        t=zeroTime
        for j in range(0,i+1):
            t=t-durations[b-j]
        ind=ImportantIndecies[0]
        while ind>=0:
            if T[ind]<=t:
                ImportantIndecies.insert(0,ind)
                break
            ind=ind-1
            
            
    # Second Part        
    for i in range(len(durations)-begin):
        t=zeroTime
        for j in range(0,i+1):
            t=t+durations[begin+j]
        ind=ImportantIndecies[begin]
        while ind<len(T):
            if T[ind]>t:
                ImportantIndecies.append(ind-1)
                break
            ind=ind+1
    
    
    
    ImportantIndecies.pop()    
    return ImportantIndecies,durations


def ReadSegments(participant,mode):
    [X,Y,T]=rs.ReadCoordinatesSynthetic(participant,mode)
    [ImportantIndecies,durations]=ReadImportantParts(participant,mode)
    N=len(ImportantIndecies)
    Segments=[]
    for i in range(N):
        index=ImportantIndecies[i]
        segmentX=[]
        segmentY=[]
        segmentT=[]
        j=index
        while T[j]-T[index]<durations[i]:
            segmentX.append(X[j])
            segmentY.append(Y[j])
            segmentT.append(T[j])
            j=j+1
        Segments.append([segmentX,segmentY,segmentT])
        
    return Segments


def ReadSegmentsSynthetic(participant,mode):
    [X,Y,T]=rs.ReadCoordinatesSynthetic(participant,mode)
    [X,Y,T]=freq.MakeSignalEven(X,Y,T,T[-1],120)
    [ImportantIndecies,durations]=ReadImportantParts(participant,mode)
    N=len(ImportantIndecies)
    Segments=[]
    for i in range(N):
        index=ImportantIndecies[i]
        segmentX=[]
        segmentY=[]
        segmentT=[]
        j=index
        while T[j]-T[index]<durations[i]:
            segmentX.append(X[j])
            segmentY.append(Y[j])
            segmentT.append(T[j])
            j=j+1
        Segments.append([segmentX,segmentY,segmentT])
        
    return Segments
    
    
def ReadCoordinatesSynthetic(participant,mode):
    [X,Y,T]=ReadCoordinates(participant,mode)    
    [X,Y,T]=freq.MakeSignalEven(X,Y,T,T[-1],120)
    return X,Y,T


def MergeSegmentsFull(participant,mode):
    FullData=ReadCoordinatesSynthetic(participant,mode)
    Segments=ReadSegmentsSynthetic(participant,mode)
    
    a=4


# MergeSegmentsFull(2,'negative')



# sss=ReadSegmentsSynthetic(7,'negative')
# sss=ReadSegmentsSynthetic(7,'negative')


# sss=ReadSegmentsSynthetic(7,'negative')