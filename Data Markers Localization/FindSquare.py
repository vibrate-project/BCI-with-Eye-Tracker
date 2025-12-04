import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
import ReadData as r
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool



# global helper
# helper=[]
# global helper2
# helper2=[]


#times are the durations of each side
def FindSquare(X, Y, T, times):
    
    
    N=len(X)

    Result=[]
    for i in range(1,N-10,5):
        print(N-i)
        duration=0
        for t in times:
            duration=duration+t
            
        [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, duration)
        boundary=FindSquareGivenSegment(X_s,Y_s,T_s,times)
        err=ErrSquare(X_s,Y_s,T_s,boundary,times)
        mins=T[i]//60
        secs=T[i]%60
        h=[err,mins,secs,T[i],i]
        
        
        totalLength=LengthBoundary(boundary)
        h.append(totalLength)
        h.append(boundary)
        
        # for b in boundary:
        #     h.append(b[0])
        #     h.append(b[1])
            
        if totalLength>0.8:
            Result.append(h)
            
        # Result.append(h)
        
    Result=Sort(Result)  
    
    
    
    return Result


def FindSquarePar(X, Y, T, times):
    
    
    N=len(X)

    Result=[]
    
    def parallel(i):
        duration=0
        for t in times:
            duration=duration+t
            
        [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, duration)
        boundary=FindSquareGivenSegment(X_s,Y_s,T_s,times)
        err=ErrSquare(X_s,Y_s,T_s,boundary,times)
        mins=T[i]//60
        secs=T[i]%60
        h=[err,mins,secs,T[i],i]
        
        
        totalLength=LengthBoundary(boundary)
        h.append(totalLength)
        h.append(boundary)
        if totalLength>0.8:
            # Result.append(h)
            return h
        return None
        
        
    if __name__ == "__main__":
        with Pool() as pool:
            results=pool.map(parallel, range(1,N-10,15))  
    

    Result = [r for r in results if r is not None]
    Result=Sort(Result) 
    return Result

    
    
    
def FindSquareGivenBegining(X, Y, T, begin, times):
    
    duration=0
    for t in times:
        duration=duration+t
        
    [X_s,Y_s,T_s]=CutSegment(X, Y, T, begin,duration)
    return FindSquareGivenSegment(X_s,Y_s,T_s,times)
    
    
    
#boundary [first point, second point, third point, fourth point]
#times [first time, second time, third time, fourth time]

def ErrSquare(X_s,Y_s,T_s,boundary,times):
    N=len(X_s)
    S=len(times)
    # time=T_s[N-1]
    # I=end-start
    # speed=I/time
    
    #Segments marks marks the i-s of each of the segments
    SegmentsMarks=[]
    previous=0
    
    
    # for i in range(N):
    #     for j in range(S):
    #         if T_s[i]>=previous+times[j]:
    #             SegmentsMarks.append([previous,i])
    #             previous=i+1
                
    
    for j in range(S):
        for i in range(previous,N):
            if T_s[i]>=T_s[previous]+times[j]:
                SegmentsMarks.append([previous,i])
                previous=i+1
                break
                   
                   
    if len(SegmentsMarks)<S:
        SegmentsMarks.append([previous,N-1])
            
        
        
    SegmentsPoints=[]
    for i in range(S):
        interval=[boundary[i],boundary[i+1]]
        SegmentsPoints.append(interval)
        
    if len(SegmentsMarks)<4:
        return 999
    Err=0
    for s in range(S):
        if s>=len(SegmentsMarks):
            print(s)
            print(SegmentsMarks)
        startNumber=SegmentsMarks[s][0]
        endNumber=SegmentsMarks[s][1]
        [startX,startY]=SegmentsPoints[s][0]
        [endX,endY]=SegmentsPoints[s][1]
        
        # time=times[s]
        startTime=T_s[startNumber]
        endTime=T_s[endNumber]
        time=endTime-startTime
        
        IntervalX=endX-startX
        IntervalY=endY-startY
        SpeedX=IntervalX/time
        SpeedY=IntervalY/time
        print(SpeedX)
        
        for i in range(startNumber,endNumber+1):
            MovingPointX=startX+SpeedX*(T_s[i]-T_s[startNumber])
            MovingPointY=startY+SpeedY*(T_s[i]-T_s[startNumber])
            # helper.append(MovingPointX)
            # helper2.append(X_s[i])
            err=(MovingPointX-X_s[i])**2+(MovingPointY-Y_s[i])**2
            Err=Err+err
    Err=Err/N
    Err=Err**(1/2)
    return Err
        
    
    
        
    
def FindSquareGivenSegment(X_s,Y_s,T_s,times):

    MaxSteps=50
    eps=0.00001
    err_tollerance=0.00001
    
    
    N=len(X_s)
    NPoints=len(times)+1
    # start=rand_start
    # end=rand_end
    def Err(boundary):
        return ErrSquare(X_s,Y_s,T_s,boundary,times)
        
    #boundary=4points x 2coordinate = matrix
    def grad(boundary):
        
       gr=[]
       for point in range(NPoints):
           part=[]
           #X and Y:
           for coord in range(2):
               arg=copy.deepcopy(boundary)
               arg[point][coord]+=eps
               el=Err(arg)-Err(boundary)
               el=el/eps
               part.append(el)
           gr.append(part)
       return gr
   
    def norm(gr):
        norm=0
        for point in range(NPoints):
            for coord in range(2):
                norm=norm+gr[point][coord]**2
        norm=norm**(1/2)
        return norm
    def normalise(gr):
        norm=0
        for point in range(NPoints):
            for coord in range(2):
                norm=norm+gr[point][coord]**2
        norm=norm**(1/2)
        
        for point in range(NPoints):
            for coord in range(2):
                gr[point][coord]/=norm
                
    def move(boundary,gr,i):
        res=[]
        for point in range(NPoints):
            part=[]
            for coord in range(2):
                el=boundary[point][coord]-gr[point][coord]/(2**i)
                part.append(el)
            res.append(part)
        return res
    
    def isInDomain(boundary):
        bo=1
        for point in range(NPoints):
            for coord in range(2):
                if 0>boundary[point][coord] or boundary[point][coord]>1:
                    bo=0
        return bo
        
        
   
    boundary=[]
    for ponit in range(NPoints):
        part=[]
        for coord in range(2):
            part.append(0.3)
        boundary.append(part)
                
    gr=grad(boundary)

    norm=norm(gr)
    if norm==0:
        return boundary        
    for point in range(NPoints):
        for coord in range(2):
            gr[point][coord]/=norm       
            
   
    
    err_new=Err(boundary)
    if err_new==999:
        return boundary
    err_old=err_new+10
    
    # while err_old-err_new>err_tollerance and (err_old-err_new)/err_new>err_tollerance:
    while err_old-err_new>err_tollerance:
        # print([err_old,err_new])
        gr=grad(boundary)
        normalise(gr)
        
        err_old=err_new
        for i in range(MaxSteps):
            boundary_temp=move(boundary,gr,i)
            
            
            # start_temp=start-gr[0]/(2**i)
            # end_temp=end-gr[1]/(2**i)
            
            
            err_temp=Err(boundary_temp)
            if(err_temp<err_new and isInDomain(boundary_temp)):
                boundary=copy.deepcopy(boundary_temp)
                err_new=err_temp
                break
            
    
    
    return boundary







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

def LengthBoundary(boundary):
    
    totalLength=0
    for j in range(len(boundary)-1):
        totalLength=totalLength+(boundary[j+1][0]-boundary[j][0])**2
        totalLength=totalLength+(boundary[j+1][1]-boundary[j][1])**2
    totalLength=totalLength+(boundary[j+1][0]-boundary[0][0])**2
    totalLength=totalLength+(boundary[j+1][1]-boundary[0][1])**2
    totalLength=totalLength**(1/2)
    return totalLength


# times=[7,5,7,5]
# [X,Y,T]=r.ReadCoordinates(1,'negative')
# [X_s,Y_s,T_s]= CutSegment(X, Y, T, 32026, 24)
# b=FindSquareGivenSegment(X_s,Y_s,T_s,times)
# # helper=[]
# # helper2=[]
# err=ErrSquare(X_s,Y_s,T_s,b,times)




if __name__ == "__main__":
       
    [X,Y,T]=r.ReadCoordinates(1,'negative')
    times=[7,5,7,5]
    res=FindSquarePar(X, Y, T, times)






