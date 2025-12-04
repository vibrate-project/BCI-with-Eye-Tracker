import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
import HelpFunctions as h
import Parameters as parm
import ReadData as read
import pickle
import os





def FindSegments(participant,mode):
    [X,Y,T]=read.ReadSyntheticCoordinates(participant,mode)
    index=int(read.ReadMovingPoints(participant,mode)[2][2])
    Time=float(read.ReadMovingPoints(participant,mode)[2][3])
    # Starting from the closest
    timesBefore=[10,10,7,15,15,15,15,15,12]
    # Starting from the closest
    timesAfter=[7,5,7,5,7,30,7,30,7,20,7,40]
    
    
    Indecies=[]
    
    time=Time-timesBefore[0]
    i=index
    t=Time
    j=0
    while i>0:
        if t<=time:
            Indecies.insert(0,i)
            j=j+1
            if j==len(timesBefore):
                break
            time=time-timesBefore[j]
        i=i-1
        t=T[i]
    
        
    Indecies.append(index)
    
    time=Time+timesAfter[0]
    i=index
    t=Time
    j=0
    while i<len(T):
        if t>=time:
            Indecies.append(i)
            j=j+1
            if j==len(timesAfter)-1:
                break
            time=time+timesAfter[j]
        i=i+1
        t=T[i]
          
        
    Times=timesBefore+timesAfter
    return [Indecies,Times]



def ReadSegments(NPeople):
    Data=[]
    modes=['negative','positive']
    
    for p in range(1,NPeople+1):
        DataModes=[]
        print(p)
        for m in modes:
            if p!=12 or m!='negative':
                print(m)
                DataSegments=[]
                [X,Y,T]=read.ReadSyntheticCoordinates(p,m)
                SegmentsCoord=FindSegments(p,m)
                NSegments=len(SegmentsCoord[0])
                for s in range(NSegments):
                    index=SegmentsCoord[0][s]
                    dur=SegmentsCoord[1][s]
                    segmentX=[]
                    segmentY=[]
                    # segmentT=[]
                    i=0
                    
                    freq=120
                    LenSeg=freq*dur
                    
                    # while T[index+i]-T[index]<dur:
                    while i<LenSeg:
                        segmentX.append(X[index+i])
                        segmentY.append(Y[index+i])
                        i=i+1
                    DataSegments.append([segmentX,segmentY])
                DataModes.append(DataSegments)
        Data.append(DataModes)
    return Data


def ReadSegmentsSomePeople(People):
    Data=[]
    modes=['negative','positive']
    
    for p in People:
        DataModes=[]
        print(p)
        for m in modes:
            if p!=12 or m!='negative':
                print(m)
                DataSegments=[]
                [X,Y,T]=read.ReadSyntheticCoordinates(p,m)
                SegmentsCoord=FindSegments(p,m)
                NSegments=len(SegmentsCoord[0])
                for s in range(NSegments):
                    index=SegmentsCoord[0][s]
                    dur=SegmentsCoord[1][s]
                    segmentX=[]
                    segmentY=[]
                    # segmentT=[]
                    i=0
                    
                    freq=120
                    LenSeg=freq*dur
                    
                    # while T[index+i]-T[index]<dur:
                    while i<LenSeg:
                        segmentX.append(X[index+i])
                        segmentY.append(Y[index+i])
                        i=i+1
                    DataSegments.append([segmentX,segmentY])
                DataModes.append(DataSegments)
        Data.append(DataModes)
    return Data




# LearningPeople=[1,3,4,6,8,10,11,13,14,15,16,17,18,20]
# data=ReadSegmentsSomePeople(LearningPeople)
# DirName='Pickle'
# MainDir=os.getcwd()    
# os.makedirs(DirName,exist_ok=True)
# os.chdir('./'+DirName)
# with open('data_LearningPeople' + '.pkl', 'wb') as f:
#     pickle.dump(data, f)
# os.chdir(MainDir)


# TestPeople=[2,5,7,9,19]   
# data=ReadSegmentsSomePeople(TestPeople)
# DirName='Pickle'
# MainDir=os.getcwd()    
# os.makedirs(DirName,exist_ok=True)
# os.chdir('./'+DirName)
# with open('data_TestPeople' + '.pkl', 'wb') as f:
#     pickle.dump(data, f)
# os.chdir(MainDir)




        
     







# b=FindSegments(1,'negative')





