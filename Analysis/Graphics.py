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
import seaborn as sns
 







def SaveFile(participant,mode):
    Points=read.ReadMovingPoints(participant,mode)
    times=[7,7,7,5,7,5]
    [X,Y,T]=read.ReadCoordinates(participant,mode)
    N=len(X)
    filenameExport='Kircho/Data_participant'+str(participant)+'_'+str(mode)+'.csv'
    with open(filenameExport, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(X)
        writer.writerow(Y)
        writer.writerow(T) 


def Histogram(participant,mode):
    
    
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
            
        Distances=[]
        for j in range(1,len(Xseg)):
            dist=((Xseg[j]-Xseg[j-1])**2+(Yseg[j]-Yseg[j-1])**2)**(1/2)
            Distances.append(dist)
            
            
        
            
        
        NameHistogram="Distance"
        Interval=[0,0.05]
        Nbars=100
        step=(Interval[1]-Interval[0])/Nbars
        Histo=[0]*Nbars
        for j in range(len(Distances)):
            value=Distances[j]
            state=Interval[0]
            for b in range(1,Nbars):
                if state<=value and value<state+step:
                    Histo[b]=Histo[b]+1
                if Interval[1]<=value:
                    Histo[-1]=Histo[-1]+1
                    break
                state=state+step
        
        labels=[]
        left=Interval[0]
        for b in range(0,Nbars):
            # right=left+step
            labels.append(b+1)
            
        
        name="Participant"+str(participant)+"_"+mode+"_Segment"+str(i+1)
        
        plt.ylim(0,150)
        plt.bar(labels,Histo)
        
        plt.title(label=name, fontweight=10, pad='2.0')
        plt.savefig("Images/"+name+".svg")
        plt.ylim(0,150)
        plt.close()
        
        
    return 0

def PlotPeople():
    modes=['positive','negative']
    Nparticipants=10
    for p in range(1,Nparticipants+1):
        for m in modes:
            Histogram(p,m)
            


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
            
            
         
        
        
    
    
    
    
    
    
# participant=5
# mode='positive'
# vp=Histogram(participant,mode)
# participant=6
# mode='negative'
# vn=Histogram(participant,mode)

# PlotPeople()






