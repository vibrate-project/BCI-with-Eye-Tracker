import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
import FindHorizontalMovingPoint as fh
import FindVerticalMovingPoint as fv
import FindSquareExact as sq
import ReadData as r
import ReadDataSynthetic as rs
import os
import HelpFunctions as h
# import Pickle as p



    
    
def FindAllMovingPointsHelp(X, Y, T, Duration_Short_Segment,Duration_Long_Segment):
    ShortHorizontaLR=fh.FindMovingPointHorizontalLR(X, Y, T, Duration_Short_Segment)
    
    LongHorizontalLR=fh.FindMovingPointHorizontalLR(X, Y, T, Duration_Long_Segment)
    LongHorizontalRL=fh.FindMovingPointHorizontalRL(X, Y, T, Duration_Long_Segment)
    
    
    ShortVerticalDU=fv.FindMovingPointVerticalDU(X, Y, T, Duration_Short_Segment)
    ShortVerticalUD=fv.FindMovingPointVerticalUD(X, Y, T, Duration_Short_Segment)
    
    return ShortHorizontaLR, LongHorizontalLR, LongHorizontalRL, ShortVerticalDU, ShortVerticalUD
    # return  fh.FindMovingPointHorizontalRL(X, Y, T, Duration_Long_Segment)
    
def FindAllMovingPoints(X, Y, T, Duration_Short_Segment,Duration_Long_Segment):
    
   
    # P1- Mono Left-Right
    # P2- Mono Up-Down
    # P3- Circular Left-Right
    # P4- Circular Up-Down
    # P5- Circular Right-Left
    # P6- Circular Down-Up
    # We find the points in the following order: P3,P5,P1,P2,P4,P6
    # The pionts contain: err,mins,secs,start,end,T[i],i
    
    ShortHorizontalLR=fh.FindMovingPointHorizontalLR(X, Y, T, Duration_Short_Segment)
    
    LongHorizontalLR=fh.FindMovingPointHorizontalLR(X, Y, T, Duration_Long_Segment)
    LongHorizontalRL=fh.FindMovingPointHorizontalRL(X, Y, T, Duration_Long_Segment)
    
    ShortVerticalDU=fv.FindMovingPointVerticalDU(X, Y, T, Duration_Short_Segment)
    ShortVerticalUD=fv.FindMovingPointVerticalUD(X, Y, T, Duration_Short_Segment)
    
    [P1,P2,P3,P4,P5,P6]=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    P3=LongHorizontalLR[0]
    P5=LongHorizontalRL[0]
    
    for el in ShortHorizontalLR:
        if P3[6]-el[6]>12:
            P1=el
            break
    
    for el in ShortVerticalUD:
        if P1[6]<el[6] and el[6]<P3[6]:
            P2=el
            break
        
        
    for el in ShortVerticalUD:
        if P3[6]<el[6] and el[6]<P5[6]:
            P4=el
            break
        
        
    for el in ShortVerticalDU:
        if el[6]>P5[6]:
            P6=el
            break   
        
    return [P1,P2,P3,P4,P5,P6]
    
    
    
    
    
def FindAllMovingPointsSquare(X, Y, T, Duration_Short_Segment,Duration_Long_Segment,times):
   
    # P1- Mono Left-Right
    # P2- Mono Up-Down
    # P3- Circular Left-Right
    # P4- Circular Up-Down
    # P5- Circular Right-Left
    # P6- Circular Down-Up
    # We find the points in the following order: P3,P5,P1,P2,P4,P6
    # The points contain: err,mins,secs,start,end,T[i],i
    
    ppp=[0,0,0,0,0]
    
    ShortHorizontaLR=fh.FindMovingPointHorizontalLR(X, Y, T, Duration_Short_Segment)
    
    LongHorizontalLR=fh.FindMovingPointHorizontalLR(X, Y, T, Duration_Long_Segment)
    LongHorizontalRL=fh.FindMovingPointHorizontalRL(X, Y, T, Duration_Long_Segment)

    
    ShortVerticalDU=fv.FindMovingPointVerticalDU(X, Y, T, Duration_Short_Segment)
    ShortVerticalUD=fv.FindMovingPointVerticalUD(X, Y, T, Duration_Short_Segment)
    
    [SquareSegments,boundary]=sq.FindFourSegmentsThroughSquare(X,Y,T,times)
    
    
    
    [P1,P2,P3,P4,P5,P6]=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    
    
    
    #START
    DirName='Pickle23'
    NNNamesJu=list(locals().keys())
    NNNamesJu.append('NNNamesJu')
    import pickle
    import os
    MainDir=os.getcwd()    
    os.makedirs(DirName,exist_ok=True)
    os.chdir('./'+DirName)

    for name in NNNamesJu:
        value=locals()[name]
        
        try:
            with open(name + '.pkl', 'wb') as f:
                pickle.dump(value, f)      
        except (TypeError, pickle.PicklingError):
            print(f"Warning: Could not pickle {name} (type: {type(value)})")
    os.chdir(MainDir)
    #END             
    
    
    
    
    for j in range(len(LongHorizontalLR)):
        start=LongHorizontalLR[j][6]
        if abs(T[SquareSegments[0][0]]-T[start])<2:         
            P3=LongHorizontalLR[j]
            break
    
    
    for j in range(len(LongHorizontalRL)):
        start=LongHorizontalRL[j][6]
        if abs(T[SquareSegments[2][0]]-T[start])<2:         
            P5=LongHorizontalRL[j]
            break
    
    
    
    #START
    DirName='Pickle2'
    NNNamesJu=list(locals().keys())
    NNNamesJu.append('NNNamesJu')
    import pickle
    import os
    MainDir=os.getcwd()    
    os.makedirs(DirName,exist_ok=True)
    os.chdir('./'+DirName)

    for name in NNNamesJu:
        value=locals()[name]
        
        try:
            with open(name + '.pkl', 'wb') as f:
                pickle.dump(value, f)      
        except (TypeError, pickle.PicklingError):
            print(f"Warning: Could not pickle {name} (type: {type(value)})")
    os.chdir(MainDir)
    #END      
    
    
    
    
    
    
    for el in ShortHorizontaLR:
        if P3[5]-el[5]>14 and P3[5]-el[5]<20:
            P1=el
            break
    # print("P1: ",P1)
    
    for el in ShortVerticalUD:
        if P1[5]<el[5] and el[5]<P3[5]:
            P2=el
            break
    # print("P2: ",P2)
        
    
    for el in ShortVerticalUD:
        if P3[5]<el[5] and el[5]<P5[5]:
            P4=el
            break
    # print("P4: ",P4)
        
        
    for el in ShortVerticalDU:
        if el[5]>P5[5] and T[el[6]]-T[P5[6]]<Duration_Long_Segment+2:
            P6=el
            break   
    # print("P6: ",P6)
        
    return [P1,P2,P3,P4,P5,P6]    
    
    
    
def FindAllMovingPointsSquarePeople(NPeople,short,long):
    Duration_Long_Segment=long
    Duration_Short_Segment=short
    mode=['negative','positive']  
    
    for np in range(1,NPeople+1):

        for m in mode:
            if h.IsBlackList(np,m)==1:
                print(h.IsBlackList(np,mode))
                [X,Y,T]=rs.ReadCoordinatesSynthetic(np,m)
                
                Points=FindAllMovingPointsSquare(X, Y, T, Duration_Short_Segment,Duration_Long_Segment,[Duration_Long_Segment, Duration_Short_Segment,Duration_Long_Segment,Duration_Short_Segment])
                times=[]
                # for p in P:
                #     times.append([p[1],p[2]])
                    
                MainDir=os.getcwd()
                os.chdir('./Result')
                os.makedirs('MovingPoints',exist_ok=True)
                os.chdir('./MovingPoints')  
                filenameExport=str('points_'+str(np)+m+'.csv')
                with open(filenameExport, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in Points:
                        writer.writerow(row) 
                        
                os.chdir(MainDir)         
           
    return 0           








# if __name__ == "__main__":      
#     [X,Y,T]=r.ReadCoordinates(7,'positive')    
#     ShortHorizontaLR=fh.FindMovingPointHorizontalLR(X, Y, T, 5)


if __name__ == "__main__":      
    times=[7,5,7,5]
    [X,Y,T]=rs.ReadCoordinatesSynthetic(7,'positive')    
    [SquareSegments,boundary]=sq.FindFourSegmentsThroughSquare(X,Y,T,times)

    
    