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
import SegmentsPositions as sp
import os











def VariablesPerson(participant,mode):
    Points=read.ReadMovingPoints(participant,mode)
    times=[7,7,7,5,7,5]
    [X,Y,T]=read.ReadSyntheticCoordinates(participant,mode)
    N=len(X)
    
    
    Following=[]
    FollowingTangent=[]
    FollowingPerpendicular=[]
    FollowingOnlyX=[]
    FollowingOnlyY=[]
    TotalDistance=[]
    
    SpeedError=[]
    SpeedErrorX=[]
    SpeedErrorY=[]
    SpeedErrorDirTangentMovement=[]
    SpeedErrorDirPerpendicularMovement=[]
    AverageSpeed=[]
    AverageSpeedX=[]
    AverageSpeedY=[]
    AverageSpeedDirTangentMovement=[]
    AverageSpeedDirPerpendicularMovement=[]
    
    AverageAcceleration=[]
    AverageAccelerationX=[]
    AverageAccelerationY=[]
    AverageAccelerationDirTangentMovement=[]
    AverageAccelerationDirPerpendicularMovement=[]
    
    NStops=[]
    Stops=[]
    DurationStops=[]
    
    
    for i in range(len(Points)):
        p=Points[i]        
        [start,end,n,ti]=p
        
        n=int(n)
        start=h.Float(start)
        end=h.Float(end)
        
        Xseg=[]
        Yseg=[]
        Tseg=[]
        for j in range(n,N):
            if T[j]-T[n]>times[i]:
                break
            Xseg.append(X[j])
            Yseg.append(Y[j])
            Tseg.append(T[j])
        segment=[Xseg,Yseg,Tseg]
        Following.append(FindFollowing(segment,start,end))
        Dir=(i+1)%2
        FollowingTangent.append(FindFollowingTangent(segment,start,end,Dir))
        FollowingPerpendicular.append(FindFollowingPerpendicular(segment,start,end,Dir))
        FollowingOnlyX.append(FindFollowingOnlyX(segment,start,end))
        FollowingOnlyY.append(FindFollowingOnlyY(segment,start,end))
        TotalDistance.append(FindTotalDistance(segment))
        
        PointSpeedX=(end[0]-start[0])/len(Xseg)
        PointSpeedY=(end[1]-start[1])/len(Xseg)
        PointSpeed=((PointSpeedX)**2+(PointSpeedY)**2)**(1/2)
        
        
        Speed=[]
        for j in range(len(Xseg)-1):
            speed=(Xseg[j+1]-Xseg[j])**2+(Yseg[j+1]-Yseg[j])**2
            speed=speed**(1/2)
            Speed.append(speed)
            
        SpeedX=[]
        for j in range(len(Xseg)-1):
            speed=Xseg[j+1]-Xseg[j]
            # speed=speed**(1/2)
            SpeedX.append(speed)
            
        SpeedY=[]
        for j in range(len(Yseg)-1):
            speed=Yseg[j+1]-Yseg[j]
            # speed=speed**(1/2)
            SpeedY.append(speed)
            
        Acceleration=[]
        for j in range(1,len(Xseg)-1):
            accel=(Xseg[j+1]-Xseg[j-1])**2+(Yseg[j+1]-Yseg[j-1])**2
            accel=accel**(1/2)
            Acceleration.append(accel)
            
        AccelerationX=[]
        for j in range(1,len(Xseg)-1):
            accel=Xseg[j+1]-Xseg[j-1]
            # accel=accel**(1/2)
            AccelerationX.append(accel)
            
        AccelerationY=[]
        for j in range(1,len(Xseg)-1):
            accel=Yseg[j+1]-Yseg[j-1]
            # accel=accel**(1/2)
            AccelerationY.append(accel)
            
        
        
        SpeedError.append(FindSpeedError(Speed,PointSpeed))
        SpeedErrorX.append(FindSpeedError(SpeedX,PointSpeedX))
        SpeedErrorY.append(FindSpeedError(SpeedY,PointSpeedY))
        SpeedErrorDirTangentMovement.append(FindSpeedErrorTangent(SpeedX,SpeedY,PointSpeedX,PointSpeedY,Dir))
        SpeedErrorDirPerpendicularMovement.append(FindSpeedErrorPerpendicular(SpeedX,SpeedY,PointSpeedX,PointSpeedY,Dir))
        AverageSpeed.append(FindSpeedError(Speed,0))
        AverageSpeedX.append(FindSpeedError(SpeedX,0))
        AverageSpeedY.append(FindSpeedError(SpeedY,0))
        AverageSpeedDirTangentMovement.append(FindSpeedErrorTangent(SpeedX,SpeedY,0,0,Dir))
        AverageSpeedDirPerpendicularMovement.append(FindSpeedErrorPerpendicular(SpeedX,SpeedY,0,0,Dir))
        
        AverageAcceleration.append(FindSpeedError(Acceleration,0))
        AverageAccelerationX.append(FindSpeedError(AccelerationX,0))
        AverageAccelerationY.append(FindSpeedError(AccelerationY,0))
        AverageAccelerationDirTangentMovement.append(FindSpeedErrorTangent(AccelerationX,AccelerationY,0,0,Dir))
        AverageAccelerationDirPerpendicularMovement.append(FindSpeedErrorPerpendicular(AccelerationX,AccelerationY,0,0,Dir))
        
        NStops.append(len(FindStops(Speed)))
        Stops.append(FindStops(Speed))
        stops=Stops[-1]
        dur=0
        for st in stops:
            dur=dur+st[1]
        DurationStops.append(dur)
            
        
        
        
        
    return [Following,FollowingTangent,FollowingPerpendicular,FollowingOnlyX,FollowingOnlyY,
            # !!!!!!!!!!!!!!!!!!
            TotalDistance,
            # !!!!!!!!!!!!!!!!!!
            SpeedError,SpeedErrorX,SpeedErrorY,SpeedErrorDirTangentMovement,SpeedErrorDirPerpendicularMovement,
            AverageSpeed,AverageSpeedX,AverageSpeedY,
            AverageSpeedDirTangentMovement,AverageSpeedDirPerpendicularMovement,
            AverageAcceleration,AverageAccelerationX,AverageAccelerationY,
            AverageAccelerationDirTangentMovement,AverageAccelerationDirPerpendicularMovement,
            NStops,
            # Stops,
            DurationStops
            
            ]


'''





def VariablesSegment(Segment):
    Points=read.ReadMovingPoints(participant,mode)
    times=[7,7,7,5,7,5]
    
    
    
    [X,Y,T]=read.ReadCoordinates(participant,mode)
    N=len(X)
    
    
    Following=[]
    FollowingTangent=[]
    FollowingPerpendicular=[]
    FollowingOnlyX=[]
    FollowingOnlyY=[]
    TotalDistance=[]
    
    SpeedError=[]
    SpeedErrorX=[]
    SpeedErrorY=[]
    SpeedErrorDirTangentMovement=[]
    SpeedErrorDirPerpendicularMovement=[]
    AverageSpeed=[]
    AverageSpeedX=[]
    AverageSpeedY=[]
    AverageSpeedDirTangentMovement=[]
    AverageSpeedDirPerpendicularMovement=[]
    
    AverageAcceleration=[]
    AverageAccelerationX=[]
    AverageAccelerationY=[]
    AverageAccelerationDirTangentMovement=[]
    AverageAccelerationDirPerpendicularMovement=[]
    
    NStops=[]
    Stops=[]
    
    
    for i in range(len(Points)):
        p=Points[i]        
        [start,end,n,ti]=p
        
        n=int(n)
        start=h.Float(start)
        end=h.Float(end)
        
        Xseg=[]
        Yseg=[]
        Tseg=[]
        for j in range(n,N):
            if T[j]-T[n]>times[i]:
                break
            Xseg.append(X[j])
            Yseg.append(Y[j])
            Tseg.append(T[j])
        segment=[Xseg,Yseg,Tseg]
        Following.append(FindFollowing(segment,start,end))
        Dir=(i+1)%2
        FollowingTangent.append(FindFollowingTangent(segment,start,end,Dir))
        FollowingPerpendicular.append(FindFollowingPerpendicular(segment,start,end,Dir))
        FollowingOnlyX.append(FindFollowingOnlyX(segment,start,end))
        FollowingOnlyY.append(FindFollowingOnlyY(segment,start,end))
        TotalDistance.append(FindTotalDistance(segment))
        
        PointSpeedX=(end[0]-start[0])/len(Xseg)
        PointSpeedY=(end[1]-start[1])/len(Xseg)
        PointSpeed=((PointSpeedX)**2+(PointSpeedY)**2)**(1/2)
        
        
        Speed=[]
        for j in range(len(Xseg)-1):
            speed=(Xseg[j+1]-Xseg[j])**2+(Yseg[j+1]-Yseg[j])**2
            speed=speed**(1/2)
            Speed.append(speed)
            
        SpeedX=[]
        for j in range(len(Xseg)-1):
            speed=Xseg[j+1]-Xseg[j]
            # speed=speed**(1/2)
            SpeedX.append(speed)
            
        SpeedY=[]
        for j in range(len(Yseg)-1):
            speed=Yseg[j+1]-Yseg[j]
            # speed=speed**(1/2)
            SpeedY.append(speed)
            
        Acceleration=[]
        for j in range(1,len(Xseg)-1):
            accel=(Xseg[j+1]-Xseg[j-1])**2+(Yseg[j+1]-Yseg[j-1])**2
            accel=accel**(1/2)
            Acceleration.append(accel)
            
        AccelerationX=[]
        for j in range(1,len(Xseg)-1):
            accel=Xseg[j+1]-Xseg[j-1]
            # accel=accel**(1/2)
            AccelerationX.append(accel)
            
        AccelerationY=[]
        for j in range(1,len(Xseg)-1):
            accel=Yseg[j+1]-Yseg[j-1]
            # accel=accel**(1/2)
            AccelerationY.append(accel)
            
        
        
        SpeedError.append(FindSpeedError(Speed,PointSpeed))
        SpeedErrorX.append(FindSpeedError(SpeedX,PointSpeedX))
        SpeedErrorY.append(FindSpeedError(SpeedY,PointSpeedY))
        SpeedErrorDirTangentMovement.append(FindSpeedErrorTangent(SpeedX,SpeedY,PointSpeedX,PointSpeedY,Dir))
        SpeedErrorDirPerpendicularMovement.append(FindSpeedErrorPerpendicular(SpeedX,SpeedY,PointSpeedX,PointSpeedY,Dir))
        AverageSpeed.append(FindSpeedError(Speed,0))
        AverageSpeedX.append(FindSpeedError(SpeedX,0))
        AverageSpeedY.append(FindSpeedError(SpeedY,0))
        AverageSpeedDirTangentMovement.append(FindSpeedErrorTangent(SpeedX,SpeedY,0,0,Dir))
        AverageSpeedDirPerpendicularMovement.append(FindSpeedErrorPerpendicular(SpeedX,SpeedY,0,0,Dir))
        
        AverageAcceleration.append(FindSpeedError(Acceleration,0))
        AverageAccelerationX.append(FindSpeedError(AccelerationX,0))
        AverageAccelerationY.append(FindSpeedError(AccelerationY,0))
        AverageAccelerationDirTangentMovement.append(FindSpeedErrorTangent(AccelerationX,AccelerationY,0,0,Dir))
        AverageAccelerationDirPerpendicularMovement.append(FindSpeedErrorPerpendicular(AccelerationX,AccelerationY,0,0,Dir))
        
        NStops.append(len(FindStops(Speed)))
        Stops.append(FindStops(Speed))
        
        
        
    return [Following,FollowingTangent,FollowingPerpendicular,FollowingOnlyX,FollowingOnlyY,
            # !!!!!!!!!!!!!!!!!!
            TotalDistance,
            # !!!!!!!!!!!!!!!!!!
            SpeedError,SpeedErrorX,SpeedErrorY,SpeedErrorDirTangentMovement,SpeedErrorDirPerpendicularMovement,
            AverageSpeed,AverageSpeedX,AverageSpeedY,
            AverageSpeedDirTangentMovement,AverageSpeedDirPerpendicularMovement,
            AverageAcceleration,AverageAccelerationX,AverageAccelerationY,
            AverageAccelerationDirTangentMovement,AverageAccelerationDirPerpendicularMovement,
            
            
            
            
            
            
            NStops,Stops
            
            ]

'''
def FindFollowing(segment,start,end):
    [X,Y,T]=segment
    N=len(T)
    grad=[(end[0]-start[0])/(N-1),(end[1]-start[1])/(N-1)]
    point=start
    Err=0
    for i in range(N):
        time=T[i]-T[0]
        err=(X[i]-point[0])**2+(Y[i]-point[1])**2
        Err=Err+err
        point=[point[0]+time*grad[0],point[1]+time*grad[1]]
    Err=Err/N
    Err=(Err)**(1/2)
    return Err



def FindFollowingTangent(segment,start,end,Dir):
    # Dir Shows the direction of the movement. 1 means the direction is X and 0 means the direction is Y
    [X,Y,T]=segment
    N=len(T)
    grad=[(end[0]-start[0])/(N-1),(end[1]-start[1])/(N-1)]
    point=start
    Err=0
    for i in range(N):
        time=T[i]-T[0]
        err=Dir*(X[i]-point[0])**2+(1-Dir)*(Y[i]-point[1])**2
        Err=Err+err
        point=[point[0]+time*grad[0],point[1]+time*grad[1]]
    Err=Err/N
    Err=(Err)**(1/2)
    return Err


def FindFollowingPerpendicular(segment,start,end,Dir):
    # Dir Shows the direction of the movement. 1 means the direction is X and 0 means the direction is Y
    [X,Y,T]=segment
    N=len(T)
    grad=[(end[0]-start[0])/(N-1),(end[1]-start[1])/(N-1)]
    point=start
    Err=0
    for i in range(N):
        time=T[i]-T[0]
        err=(1-Dir)*(X[i]-point[0])**2+Dir*(Y[i]-point[1])**2
        Err=Err+err
        point=[point[0]+time*grad[0],point[1]+time*grad[1]]
    Err=Err/N
    Err=(Err)**(1/2)
    return Err


def FindFollowingOnlyX(segment,start,end):
    [X,Y,T]=segment
    N=len(T)
    grad=[(end[0]-start[0])/(N-1),(end[1]-start[1])/(N-1)]
    point=start
    Err=0
    for i in range(N):
        time=T[i]-T[0]
        err=(X[i]-point[0])**2
        Err=Err+err
        point=[point[0]+time*grad[0],point[1]+time*grad[1]]
    Err=Err/N
    Err=(Err)**(1/2)
    return Err



def FindFollowingOnlyY(segment,start,end):
    [X,Y,T]=segment
    N=len(T)
    grad=[(end[0]-start[0])/(N-1),(end[1]-start[1])/(N-1)]
    point=start
    Err=0
    for i in range(N):
        time=T[i]-T[0]
        err=(Y[i]-point[1])**2
        Err=Err+err
        point=[point[0]+time*grad[0],point[1]+time*grad[1]]
    Err=Err/N
    Err=(Err)**(1/2)
    return Err

def FindTotalDistance(segment):
    [X,Y,T]=segment
    N=len(T)
    
    Distance=0
    for i in range(1,N):
        dist=((X[i]-X[i-1])**2+(Y[i]-Y[i-1])**2)**(1/2)
        Distance=Distance+dist
        
    return Distance


def FindSpeedError(SpeedSeg,speedPoint):
    Err=0
    N=len(SpeedSeg)
    for i in range(N):
        err=(SpeedSeg[i]-speedPoint)**2
        Err=Err+err
    Err=Err/N
    Err=Err**(1/2)
    return Err

def FindSpeedErrorTangent(SpeedSegX,SpeedSegY,speedPointX,speedPointY,Dir):
    Err=0
    N=len(SpeedSegX)
    for i in range(N):
        err=Dir*(SpeedSegX[i]-speedPointX)**2+(1-Dir)*(SpeedSegY[i]-speedPointY)**2
        Err=Err+err
    Err=Err/N
    Err=Err**(1/2)
    return Err

def FindSpeedErrorPerpendicular(SpeedSegX,SpeedSegY,speedPointX,speedPointY,Dir):
    Err=0
    N=len(SpeedSegX)
    for i in range(N):
        err=(1-Dir)*(SpeedSegX[i]-speedPointX)**2+Dir*(SpeedSegY[i]-speedPointY)**2
        Err=Err+err
    Err=Err/N
    Err=Err**(1/2)
    return Err



def FindStops(SpeedSeg):
    Percentage=40
    MinimalLength=10
    
    AverageSpeed=0
    N=len(SpeedSeg)
    for i in range(N):
        AverageSpeed=AverageSpeed+SpeedSeg[i]
    AverageSpeed=AverageSpeed/N
    SpeedSegSorted=h.Sort(SpeedSeg)
    mark=int(Percentage/100*N)
    Thresshold=SpeedSegSorted[mark]
    
    Stops=[]
    end=-1
    for i in range(N):
        if SpeedSeg[i]<Thresshold and i>end:
            end=0
            j=i
            while j<N and SpeedSeg[j]<Thresshold:
                j=j+1
            length=j-i
            
            if length>MinimalLength:
                stop=[i,length]
                Stops.append(stop)
            end=j
                
    return Stops
            
            
         
    

def SaveVariables():      
    modes=['negative','positive']
    # participant / mode / segment 
    LearningPeople=[1,3,4,6,8,10,11,13,14,15,16,17,18,20]
    data=[]
    for p in LearningPeople:
        per=[]
        for m in modes:
            per.append(VariablesPerson(p,m))
        data.append(per)
    DirName='Pickle'
    MainDir=os.getcwd()    
    os.makedirs(DirName,exist_ok=True)
    os.chdir('./'+DirName)
    with open('ParData_LearningPeople' + '.pkl', 'wb') as f:
        pickle.dump(data, f)
    os.chdir(MainDir)
    
    
    modes=['negative','positive']
    TestPeople=[2,5,7,9,19]   
    data=[]
    for p in TestPeople:
        per=[]
        for m in modes:
            per.append(VariablesPerson(p,m))
        data.append(per)
    DirName='Pickle'
    MainDir=os.getcwd()    
    os.makedirs(DirName,exist_ok=True)
    os.chdir('./'+DirName)
    with open('ParData_TestPeople' + '.pkl', 'wb') as f:
        pickle.dump(data, f)
    os.chdir(MainDir)

def SaveVariablesAll():      
    modes=['negative','positive']
    # participant / mode / segment 
    LearningPeople=[i for i in range(1,21) if i!=12]
    data=[]
    for p in LearningPeople:
        per=[]
        for m in modes:
            per.append(VariablesPerson(p,m))
        data.append(per)
    DirName='Pickle'
    MainDir=os.getcwd()    
    os.makedirs(DirName,exist_ok=True)
    os.chdir('./'+DirName)
    with open('ParData_AllPeople' + '.pkl', 'wb') as f:
        pickle.dump(data, f)
    os.chdir(MainDir)
    
    


# SaveVariablesAll()

















