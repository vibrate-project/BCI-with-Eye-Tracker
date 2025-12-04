import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
from pathos.multiprocessing import ProcessingPool as Pool
import Parameters as parm






def FindMovingPointVerticalDU(X, Y, T, Duration_Segment):
    


    N=len(X)
    def func(i):
        [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, Duration_Segment)
        [start,end,Xcoord]=FindMovingPointGivenSegmentVerticalDU(X_s, Y_s, T_s, 0.3, 0.3)
        err=ErrMovingPointVerticalDU(X_s,Y_s,T_s,start,end)
        mins=T[i]//60
        secs=T[i]%60
        h=[err,mins,secs,start,end,T[i],i,Xcoord]
        if end-start>0.4:
            return h
        else:
            return None
        

    with Pool() as pool:
        results=pool.map(func, range(1,N-parm.SaveLast,parm.NStepSegments))
        
    Result = [r for r in results if r is not None] 
    Result=Sort(Result)
    
    return Result


def FindMovingGivenBeginingDU(X, Y, T, begin, Time_Segment):
    
    [X_s,Y_s,T_s]=CutSegment(X, Y, T, begin,Time_Segment)
    return FindMovingPointGivenSegmentVerticalDU(X_s,Y_s,T_s)
    
    
    

def ErrMovingPointVerticalDU(X_s,Y_s,T_s,start,end):
    N=len(X_s)
    time=T_s[N-1]
    I=end-start
    speed=I/time
    
    
    X_center=0
    for x in X_s:
        X_center+=x**2
    X_center=(X_center/N)**(1/2)
    
    errY=0
    for i in range(N):
        errY+=(Y_s[i]-(start+T_s[i]*speed))**2
    errY=(errY/N)**(1/2)
    
    errX=0
    for i in range(N):
        errX+=(X_s[i]-X_center)**2
    errX=(errX/N)**(1/2)
    
    err=(errX**2+errY**2)**(1/2)
    
    return err

        
    
def FindMovingPointGivenSegmentVerticalDU(X_s,Y_s,T_s,rand_start,rand_end):

    MaxSteps=5
    eps=0.0001
    err_tollerance=0.0000001
    N=len(X_s)
    start=rand_start
    end=rand_end
    def Err(start,end):
        return ErrMovingPointVerticalDU(X_s,Y_s,T_s,start,end)
               
    grad=[Err(start+eps,end)-Err(start,end),Err(start,end+eps)-Err(start,end)]
    norm=(grad[0]**2+grad[1]**2)**(1/2)
    grad[0]=grad[0]/norm
    grad[1]=grad[1]/norm
    
    err_new=Err(start,end)
    err_old=err_new+10
    
    while err_old-err_new>err_tollerance:
        grad=[Err(start+eps,end)-Err(start,end),Err(start,end+eps)-Err(start,end)]
        norm=(grad[0]**2+grad[1]**2)**(1/2)
        grad[0]=grad[0]/norm
        grad[1]=grad[1]/norm
        
        err_old=err_new
        for i in range(MaxSteps):
            start_temp=start-grad[0]/(2**i)
            end_temp=end-grad[1]/(2**i)
            err_temp=Err(start_temp,end_temp)
            if(err_temp<err_new and 0<start_temp and start_temp<1 and 0<end_temp and end_temp<1):
                start=start_temp
                end=end_temp
                err_new=err_temp
                break
            
    X_center=0
    for x in X_s:
        X_center+=x**2
    X_center=(X_center/N)**(1/2)
    Xcoord=X_center
    
    return [start,end,Xcoord]

    


#Here starts the Up-Down movement

def FindMovingPointVerticalUD(X, Y, T, Duration_Segment):
    

     N=len(X)
     def func(i):
         [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, Duration_Segment)
         [start,end,Xcoord]=FindMovingPointGivenSegmentVerticalUD(X_s, Y_s, T_s, 0.3, 0.3)
         err=ErrMovingPointVerticalUD(X_s,Y_s,T_s,start,end)
         mins=T[i]//60
         secs=T[i]%60
         h=[err,mins,secs,start,end,T[i],i,Xcoord]
         if end-start<-0.4:
             return h
         else:
             return None
         

     with Pool() as pool:
         results=pool.map(func, range(1,N-parm.SaveLast,parm.NStepSegments))
         # results=pool.map(func, range(10000,22000,50))
     Result = [r for r in results if r is not None] 
     Result=Sort(Result)
     
     return Result





    



def FindMovingGivenBeginingUD(X, Y, T, begin, Time_Segment):
    
    [X_s,Y_s,T_s]=CutSegment(X, Y, T, begin,Time_Segment)
    return FindMovingPointGivenSegmentVerticalDU(X_s,Y_s,T_s)
    
    
    

def ErrMovingPointVerticalUD(X_s,Y_s,T_s,start,end):
    N=len(X_s)
    time=T_s[N-1]
    I=end-start
    speed=I/time
    
    
    X_center=0
    for x in X_s:
        X_center+=x**2
    X_center=(X_center/N)**(1/2)
    
    errY=0
    for i in range(N):
        errY+=(Y_s[i]-(start+T_s[i]*speed))**2
    errY=(errY/N)**(1/2)
    
    errX=0
    for i in range(N):
        errX+=(X_s[i]-X_center)**2
    errX=(errX/N)**(1/2)
    
    err=(errX**2+errY**2)**(1/2)
    
    return err

        
    
def FindMovingPointGivenSegmentVerticalUD(X_s,Y_s,T_s,rand_start,rand_end):

    MaxSteps=10
    eps=0.0001
    err_tollerance=0.0000001
    N=len(X_s)
    start=rand_start
    end=rand_end
    def Err(start,end):
        return ErrMovingPointVerticalUD(X_s,Y_s,T_s,start,end)
               
    grad=[Err(start+eps,end)-Err(start,end),Err(start,end+eps)-Err(start,end)]
    norm=(grad[0]**2+grad[1]**2)**(1/2)
    grad[0]=grad[0]/norm
    grad[1]=grad[1]/norm
    
    err_new=Err(start,end)
    err_old=err_new+10
    
    while err_old-err_new>err_tollerance:
        grad=[Err(start+eps,end)-Err(start,end),Err(start,end+eps)-Err(start,end)]
        norm=(grad[0]**2+grad[1]**2)**(1/2)
        grad[0]=grad[0]/norm
        grad[1]=grad[1]/norm
        
        err_old=err_new
        for i in range(MaxSteps):
            start_temp=start-grad[0]/(2**i)
            end_temp=end-grad[1]/(2**i)
            err_temp=Err(start_temp,end_temp)
            if(err_temp<err_new and 0<start_temp and start_temp<1 and 0<end_temp and end_temp<1):
                start=start_temp
                end=end_temp
                err_new=err_temp
                break
            
    X_center=0
    for x in X_s:
        X_center+=x**2
    X_center=(X_center/N)**(1/2)
    Xcoord=X_center
    
    return [start,end,Xcoord]


#Here ends the Up-Down movement











def CutSegment(X, Y, T, begin,Duration_Segment):
    X_segment=[]
    Y_segment=[]
    T_segment=[]
    t0=T[begin]
    for i in range(begin,len(X)):
        if T[i]-T[begin]<=Duration_Segment:
            X_segment.append(X[i])
            Y_segment.append(Y[i])
            T_segment.append(T[i]-T[begin])
        else:
            break
    return(X_segment,Y_segment,T_segment)
    


    
    
    
    
def Sort(M):
    A=M.copy()
    N=len(A)
    for i in range(N):
        for j in range(i,N):
            if A[i][0]>A[j][0]:
                h=A[i]
                A[i]=A[j]
                A[j]=h
    return A   
    
    
    
    
    

    