import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy
import FindHorizontalMovingPoint as hor
import FindVerticalMovingPoint as ver
import HelpFunctions as h
import ReadData as r
import FindAllMovingPoints as fmp
import multiprocessing as mp
import os


#Global parameters
Time_Tollerance=0.2  # seconds
Time_Short=5 #seconds
Time_Long=7 #seconds
NumParticipants=6
Duration_Short_Segment=Time_Short-Time_Tollerance
Duration_Long_Segment=Time_Long-Time_Tollerance     

Mode=['positive','negative']


def writePoints(arg):    
    participant=arg[0]
    mode=arg[1]
    [X,Y,T]=r.ReadCoordinates(participant,mode)
    Points6=fmp.FindAllMovingPoints(X, Y, T, Duration_Short_Segment,Duration_Long_Segment)
    
    parentDir=os.getcwd()
    os.chdir('./Result')
    os.mkdir('/Points')
    os.chdir(parentDir)
      
    filenameExport='ResultSynthetic/Points/'+'POINTS'+str(participant)+mode+'.csv'
    with open(filenameExport, 'w') as f:
        writer = csv.writer(f)
        for row in Points6:
            writer.writerow(row)    
        
        
if __name__ == "__main__":
    pool = mp.Pool() 
    pool.map(writePoints, [[par,mode] for par in range(1,NumParticipants+1) for mode in Mode])
    pool.close()
    pool.joint()






    
    
