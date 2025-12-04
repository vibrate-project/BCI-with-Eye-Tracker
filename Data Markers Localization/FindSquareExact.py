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
import Parameters as parm
import logging



global helper
helper=[]
# global helper2
# helper2=[]


#times are the durations of each side



def FindSquarePar(X, Y, T, times):
    
    
    N=len(X)  
    
    
    # DirName='Pickle444'
    # NNNamesJu=list(locals().keys())
    # NNNamesJu.append('NNNamesJu')
    # import pickle
    # import os
    # MainDir=os.getcwd()    
    # os.makedirs(DirName,exist_ok=True)
    # os.chdir('./'+DirName)

    # for name in NNNamesJu:
    #     value=locals()[name]
        
    #     try:
    #         with open(name + '.pkl', 'wb') as f:
    #             pickle.dump(value, f)      
                
                
                
                
    #     except (TypeError, pickle.PicklingError):
    #         print(f"Warning: Could not pickle {name} (type: {type(value)})")
    # os.chdir(MainDir)
    
    
    def CutSegmentHelp(X, Y, T, i, duration):
        return CutSegment(X, Y, T, i, duration)
    
    def parallel(i):
        duration=0
        for t in times:
            duration=duration+t
        # deleteee(3)    
        [X_s,Y_s,T_s]=CutSegmentHelp(X, Y, T, i, duration)
        boundary=FindSquareGivenSegment(X_s,Y_s,T_s,times)
        err=ErrSquare(X_s,Y_s,T_s,boundary,times)
        mins=T[i]//60
        secs=T[i]%60
        h=[err,mins,secs,T[i],i]
        
        # print(boundary)
        totalLength=LengthBoundary(boundary)
        h.append(totalLength)
        h.append(boundary)
        
        
        # HERE
        # print(i)
        # return h
    
    
        if IsSquare(boundary)==1:
            return h
        return None
    
    # results=[]
    # for i in range(19000,21000,50):
    #     print(i)
    #     results.append(parallel(i))
        
    with Pool() as pool:
        results=pool.map(parallel, range(1,N-parm.SaveLast,parm.NStepSegments))
        
   
        

    Result = [r for r in results if r is not None]
    Result=Sort(Result) 
    
    # if len(Result)==0:
    #     return [10000, 0, 0, 0, 0, 0, [[0, 0], [0, 0], [0, 0],[0, 0],[0, 0]]]
    
    # HERE
    # return Result
    return Result[0]

    
def deleteee(a):
    print(a)
    
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
        # if s>=len(SegmentsMarks):
            # print(s)
            # print(SegmentsMarks)
        startNumber=SegmentsMarks[s][0]
        endNumber=SegmentsMarks[s][1]
        [startX,startY]=SegmentsPoints[s][0]
        [endX,endY]=SegmentsPoints[s][1]
        
        # time=times[s]
        
        
        
        
        # DirName='Pickle108'
        # NNNamesJu=list(locals().keys())
        # NNNamesJu.append('NNNamesJu')
        # import pickle
        # import os
        # MainDir=os.getcwd()    
        # os.makedirs(DirName,exist_ok=True)
        # os.chdir('./'+DirName)

        # for name in NNNamesJu:
        #     value=locals()[name]
            
        #     try:
        #         with open(name + '.pkl', 'wb') as f:
        #             pickle.dump(value, f)      
                    
                    
                    
                    
        #     except (TypeError, pickle.PicklingError):
        #         print(f"Warning: Could not pickle {name} (type: {type(value)})")
        # os.chdir(MainDir) 
        
        
        
        
        startTime=T_s[startNumber]
        endTime=T_s[endNumber]
        time=endTime-startTime
        
        IntervalX=endX-startX
        IntervalY=endY-startY
        SpeedX=IntervalX/time
        SpeedY=IntervalY/time
        # print(SpeedX)
        
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
    Ncoord=6
    
    # coord=[X1,Y1,X2,y1,x2,Y3,X4,y3,x4,Y5]
    # coord=[X1,Y1,X2,Y3,X4,Y5] / 6 arguments
    index=[0, 1, 2, 1, 2, 3, 4, 3, 4, 5]
    def Err(coord):
        # helper=coord
        boundary=[]
        for i in range(5):
            point=[coord[index[2*i]],coord[index[2*i+1]]]
            boundary.append(point)
        return ErrSquare(X_s,Y_s,T_s,boundary,times)
        
    #boundary=4points x 2coordinate = matrix
    def grad(coord):
        
       gr=[]
       for i in range(Ncoord):
            arg=copy.deepcopy(coord)
            arg[i]+=eps
            derriv=Err(arg)-Err(coord)
            derriv=derriv/eps
            gr.append(derriv)
       return gr
   
    def norm(coord):
        norm=0
        for i in range(Ncoord):
            norm=norm+coord[i]**2
        norm=norm**(1/2)
        return norm
    def normalise(coord):
        norm=0
        for i in range(Ncoord):
            norm=norm+coord[i]**2
        norm=norm**(1/2)
        for i in range(Ncoord):
            coord[i]=coord[i]/norm
        
                
    def move(coord,gr,small):
        res=[]
        for i in range(Ncoord):
            el=coord[i]-gr[i]/(2**small)
            res.append(el)
        return res
    
    def isInDomain(coord):
        bo=1
        for i in range(Ncoord):
            if 0>coord[i] or coord[i]>1:
                bo=0
        return bo
        
        
   
    coord=[]
    for i in range(Ncoord):
        coord.append(0.3)
    helper=coord            
    gr=grad(coord)

    norm=norm(gr)
    if norm==0:
        boundary=[]
        for i in range(5):
            point=[coord[index[2*i]],coord[index[2*i+1]]]
            boundary.append(point)
        return boundary        
    for i in range(Ncoord):
        gr[i]/=norm       
    
    err_new=Err(coord)
    if err_new==999:
        boundary=[]
        for i in range(5):
            point=[coord[index[2*i]],coord[index[2*i+1]]]
            boundary.append(point)
        return boundary 
    err_old=err_new+10
    
    # while err_old-err_new>err_tollerance and (err_old-err_new)/err_new>err_tollerance:
    while err_old-err_new>err_tollerance:
        # print([err_old,err_new])
        gr=grad(coord)
        normalise(gr)
        
        err_old=err_new
        for i in range(MaxSteps):
            coord_temp=move(coord,gr,i)
            err_temp=Err(coord_temp)
            if(err_temp<err_new and isInDomain(coord_temp)):
                coord=copy.deepcopy(coord_temp)
                err_new=err_temp
                break
            
    
    boundary=[]
    for i in range(5):
        point=[coord[index[2*i]],coord[index[2*i+1]]]
        boundary.append(point)
    return boundary



def IsSquare(boundary):
    
    MinSide=0.3
    MinPerimeter=2
    boo=1
    for p in range(len(boundary)-1):
        side=0
        side=side+(boundary[p][0]-boundary[p+1][0])**2
        side=side+(boundary[p][1]-boundary[p+1][1])**2
        side=side**(1/2)
        # print("side: ", side)
        if side<MinSide:
            boo=0

    # print("bound: ",LengthBoundary(boundary))
    if LengthBoundary(boundary)<MinPerimeter:
        boo=0
    
    return boo


def CutSegment(X, Y, T, begin, Duration_Segment):
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
        side=0
        side=side+(boundary[j+1][0]-boundary[j][0])**2
        side=side+(boundary[j+1][1]-boundary[j][1])**2
        side=side**(1/2)
        # print(side)
        totalLength=totalLength+side
    # totalLength=totalLength+(boundary[j+1][0]-boundary[0][0])**2
    # totalLength=totalLength+(boundary[j+1][1]-boundary[0][1])**2
    
    return totalLength

# def FindSquare(X, Y, T, times):
    
    
#     N=len(X)

#     Result=[]
#     for i in range(1,N-10,5):
#         # print(N-i)
#         duration=0
#         for t in times:
#             duration=duration+t
            
#         [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, duration)
#         boundary=FindSquareGivenSegment(X_s,Y_s,T_s,times)
#         err=ErrSquare(X_s,Y_s,T_s,boundary,times)
#         mins=T[i]//60
#         secs=T[i]%60
#         h=[err,mins,secs,T[i],i]
        
        
#         totalLength=LengthBoundary(boundary)
#         h.append(totalLength)
#         h.append(boundary)
        
#         # for b in boundary:
#         #     h.append(b[0])
#         #     h.append(b[1])
            
#         if totalLength>0.8:
#             Result.append(h)
            
#         # Result.append(h)
        
#     Result=Sort(Result)  
    
    
    
#     return Result




def FindFourSegmentsThroughSquare(X,Y,T,times):

    res=FindSquarePar(X, Y, T, times)
    boundary=res[6]
    dur=0
    for t in times:
        dur=dur+t
    
    
    
    
    # DirName='Pickle108'
    # NNNamesJu=list(locals().keys())
    # NNNamesJu.append('NNNamesJu')
    # import pickle
    # import os
    # MainDir=os.getcwd()    
    # os.makedirs(DirName,exist_ok=True)
    # os.chdir('./'+DirName)

    # for name in NNNamesJu:
    #     value=locals()[name]
        
    #     try:
    #         with open(name + '.pkl', 'wb') as f:
    #             pickle.dump(value, f)      
                
                
                
                
    #     except (TypeError, pickle.PicklingError):
    #         print(f"Warning: Could not pickle {name} (type: {type(value)})")
    # os.chdir(MainDir) 
    
    
    
    
    
    Start=res[4]
    Segments=[]
    segment=[Start]
    i=0
    dur=0
    
    
    
    
    
    
    
    
    
    
    
    for t in times:
        dur=dur+t
        while T[Start+i]-T[Start]<dur:
            i=i+1
        segment.append(Start+i-1)
        Segments.append(segment)
        segment=[Start+i]
    
    
    
    return [Segments,boundary] 




if __name__ == "__main__":
       
    [X,Y,T]=r.ReadCoordinates(10,'negative')
    times=[7,5,7,5]
    res=FindSquarePar(X, Y, T, times)
    
#     filenameExport=str('RESULT.csv')
#     with open(filenameExport, 'w') as f:
#         writer = csv.writer(f)
#         for row in res:
#             writer.writerow(row)





# if __name__ == "__main__":
#     [X,Y,T]=r.ReadCoordinates(7,'positive')
#     times=[7,5,7,5]
#     res2=FindFourSegmentsThroughSquare(X,Y,T,times)
    
    
    
    
    
    
    
    
    # res=FindSquarePar(X, Y, T, times)

# if __name__ == "__main__":
#     [X,Y,T]=r.ReadCoordinates(3,'negative')
#     times=[7,5,7,5]
#     best=FindSquarePar(X, Y, T, times)

# if __name__ == "__main__":
#     times=[7,5,7,5]
#     [X,Y,T]=r.ReadCoordinates(4,'negative')
#     FindSquarePar(X, Y, T, times)
    
    
    

# if __name__ == "__main__":
#     [X,Y,T]=r.ReadCoordinates(4,'negative')
#     times=[7,5,7,5]
#     i=35331
#     duration=0
    

    
#     for t in times:
#         duration=duration+t
    
#     [X_s,Y_s,T_s]=CutSegment(X, Y, T, i, duration)
    
#     FindSquareGivenSegment(X_s,Y_s,T_s,times)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    