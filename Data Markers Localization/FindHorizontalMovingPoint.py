import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
from pathos.multiprocessing import ProcessingPool as Pool
import Parameters as parm
        
        
        
def FindMovingPointHorizontalLR(X, Y, T, Duration_Segment):
        
    N=len(X)
    def func(i):
        print(i)
        [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, Duration_Segment)
        [start,end,Ycoord]=FindMovingPointGivenSegmentHorizontalLR(X_s, Y_s, T_s, 0.3, 0.3)
        err=ErrMovingPointHorizontalLR(X_s,Y_s,T_s,start,end)
        
        mins=T[i]//60
        secs=T[i]%60
        
        
           
        
        h=[err,mins,secs,start,end,T[i],i,Ycoord]
        if end-start>0.4:
            return h
        else:
            return None
                
        
    with Pool() as pool:
        results=pool.map(func, range(1,N-parm.SaveLast,parm.NStepSegments))
        
    # results=[]
    # for i in range(1,N-10,10):
    #     results.append(func(i))
        
    Result = [r for r in results if r is not None] 
    Result=Sort(Result)
        
    return Result
        
def FindMovingGivenBeginingHorizontalLR(X, Y, T, begin, Time_Segment):
        
    [X_s,Y_s,T_s]=CutSegment(X, Y, T, begin,Time_Segment)
    return FindMovingPointGivenSegmentHorizontalLR(X_s,Y_s,T_s)
        
        
def ErrMovingPointHorizontalLR(X_s,Y_s,T_s,start,end):
    
    
    N=len(X_s)
    time=T_s[N-1]
    I=end-start
    speed=I/time
    
    
    Y_center=0
    for y in Y_s:
        Y_center+=y**2
    Y_center=(Y_center/N)**(1/2)
    
    
    errX=0
    for i in range(N):
        errX+=(X_s[i]-(start+T_s[i]*speed))**2
    errX=(errX/N)**(1/2)
    
    
    errY=0
    for i in range(N):
        errY+=(Y_s[i]-Y_center)**2
    errY=(errY/N)**(1/2)
    
    
    err=(errX**2+errY**2)**(1/2)
    
    return err
    
    
    
    # Radius=0.05
    # Outlaier=40 #procents
    # N=len(X_s)
    # good=N*(100-Outlaier)//100
    # time=T_s[N-1]
    # I=end-start
    # speed=I/time
    
    
    # Y_center=0
    # for y in Y_s:
    #     Y_center+=y**2
    # Y_center=(Y_center/N)**(1/2)
    
    
    
    
    
    # ErrX=[]
    # for i in range(N):
    #     ErrX.append((X_s[i]-(start+T_s[i]*speed))**2)
    
    # ErrY=[]
    # for i in range(N):
    #     ErrY.append((Y_s[i]-Y_center)**2)
    
    # Err=[]
    # for i in range(N):
    #     Err.append((ErrX[i]**2+ErrY[i]**2)**(1/2))
    
    
    # for i in range(N):
    #     for j in range(i+1,N):
    #         if(Err[i]>Err[j]):
    #             h=Err[i]
    #             Err[i]=Err[j]
    #             Err[j]=h
    # err=0
    # for i in range(good):
    #     # if Err[i]>Radius:
    #     err+=Err[i]
    # err/=good
    # return err

        
    
def FindMovingPointGivenSegmentHorizontalLR(X_s,Y_s,T_s,rand_start,rand_end):

    MaxSteps=5
    eps=0.0001
    err_tollerance=0.0000001
    N=len(X_s)
    start=rand_start
    end=rand_end
    def Err(start,end):
        return ErrMovingPointHorizontalLR(X_s,Y_s,T_s,start,end)
               
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
    Y_center=0
    for y in Y_s:
        Y_center+=y**2
    Y_center=(Y_center/N)**(1/2)
    Ycoord=Y_center
    
    return [start,end,Ycoord]

    
#Here starts the Right-Left Movement

def FindMovingPointHorizontalRL(X, Y, T, Duration_Segment):

    rand_X=0.3
    rand_Y=0.3

    N=len(X)
    N=len(X)
    def func(i):
        [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, Duration_Segment)
        [start,end,Ycoord]=FindMovingPointGivenSegmentHorizontalRL(X_s, Y_s, T_s, rand_X, rand_Y)
        err=ErrMovingPointHorizontalRL(X_s,Y_s,T_s,start,end)
        mins=T[i]//60
        secs=T[i]%60
        h=[err,mins,secs,start,end,T[i],i,Ycoord]
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






def FindMovingPointHorizontalRL_SKIP(X, Y, T, Duration_Segment):

    rand_X=0.1
    rand_Y=0.9

    N=len(X)
    Result=[]
    for i in range(1,N-10,5):
        [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, Duration_Segment)
        [start,end]=FindMovingPointGivenSegmentHorizontalRL(X_s, Y_s, T_s, rand_X, rand_Y)
        err=ErrMovingPointHorizontalRL(X_s,Y_s,T_s,start,end)
        mins=T[i]//60
        secs=T[i]%60
        h=[err,mins,secs,start,end,T[i],i]
        if end-start<-0.5:
            Result.append(h)
            
    Result=Sort(Result) 

            
    return Result


def FindMovingGivenBeginingHorizontalRL(X, Y, T, begin, Time_Segment):
    
    [X_s,Y_s,T_s]=CutSegment(X, Y, T, begin,Time_Segment)
    return FindMovingPointGivenSegmentHorizontalRL(X_s,Y_s,T_s)
    
    
    

def ErrMovingPointHorizontalRL(X_s,Y_s,T_s,start,end):
    # N=len(X_s)
    # time=T_s[N-1]
    # I=end-start
    # speed=-I/time
    
    
    # Y_center=0
    # for y in Y_s:
    #     Y_center+=y**2
    # Y_center=(Y_center/N)**(1/2)
    
    
    # errX=0
    # for i in range(N):
    #     errX+=(X_s[i]-(start+T_s[i]*speed))**2
    # errX=(errX/N)**(1/2)
    
    
    # errY=0
    # for i in range(N):
    #     errY+=(Y_s[i]-Y_center)**2
    # errY=(errY/N)**(1/2)
    
    
    # err=(errX**2+errY**2)**(1/2)
    
    # return err







  # Radius=0.005
  Outlaier=10 #procents
  N=len(X_s)
  good=N*(100-Outlaier)//100
  
  time=T_s[N-1]
  I=end-start
  speed=I/time
 
 
  Y_center=0
  for y in Y_s:
      Y_center+=y
  Y_center=Y_center/N
 
 
 
 
 
  ErrX=[]
  for i in range(N):
      ErrX.append((X_s[i]-(start+T_s[i]*speed))**2)
 
  ErrY=[]
  for i in range(N):
      ErrY.append((Y_s[i]-Y_center)**2)
 
  Err=[]
  for i in range(N):
      Err.append((ErrX[i]**2+ErrY[i]**2)**(1/2))
 
 
  for i in range(N):
      for j in range(i+1,N):
          if(Err[i]>Err[j]):
              h=Err[i]
              Err[i]=Err[j]
              Err[j]=h
  err=0
  # good=N
  for i in range(good):
       # if Err[i]>Radius:
      err+=Err[i]
  err/=good
  return err

        
    
def FindMovingPointGivenSegmentHorizontalRL(X_s,Y_s,T_s,rand_start,rand_end):

    MaxSteps=5
    eps=0.000001
    err_tollerance=0.001
    N=len(X_s)
    start=rand_start
    end=rand_end
    def Err(start,end):
        return ErrMovingPointHorizontalRL(X_s,Y_s,T_s,start,end)
               
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
    Y_center=0
    for y in Y_s:
        Y_center+=y**2
    Y_center=(Y_center/N)**(1/2)
    Ycoord=Y_center
    
    return [start,end,Ycoord]
#Here Finishes the Right-Left Movement

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





