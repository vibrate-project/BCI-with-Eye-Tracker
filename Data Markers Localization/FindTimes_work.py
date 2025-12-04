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


if __name__ == "__main__":

    #Global parameters
    Time_Tollerance=0  # seconds
    Time_Short=5 #seconds
    Time_Long=7 #seconds
    NumParticipants=6
    
    Duration_Short_Segment=Time_Short-Time_Tollerance
    Duration_Long_Segment=Time_Long-Time_Tollerance     
    
    Mode=['positive','negative']
    
    MainDir=os.getcwd()
    os.chdir('./Result')
    os.makedirs('ElementsWork',exist_ok=True)
    os.chdir(MainDir)



def writeElementsWork(arg):    
    participant=arg[0]
    mode=arg[1]
    Duration_Short_Segment=arg[2]
    Duration_Long_Segment=arg[3]
    [X,Y,T]=r.ReadCoordinates(participant,mode)
    Elements=fmp.FindAllMovingPointsHelp(X, Y, T, Duration_Short_Segment,Duration_Long_Segment)
    names=['ShortHorizontaLR', 'LongHorizontalLR', 'LongHorizontalRL', 'ShortVerticalDU', 'ShortVerticalUD']
    
    parentDir=os.getcwd()
    os.chdir('./Result/ElementsWork/')
    DirName='Participant'+str(participant)+'_'+mode
    os.makedirs(DirName,exist_ok=True)
    os.chdir(DirName)
    
    
    
    
    # DirName='Participant'+str(participant)+'_'
    
    # path = os.getcwd()
    # parent = os.path.dirname(path)
    
    for i in range(len(names)):
        filenameExport=names[i]+'.csv'
        with open(filenameExport, 'w') as f:
            writer = csv.writer(f)
            for row in Elements[i]:
                writer.writerow(row)  
    os.chdir(parentDir)
    return 0
        
    
    
        
if __name__ == "__main__":
    pool = mp.Pool() 
    pool.map(writeElementsWork, [[par,mode,Duration_Short_Segment,Duration_Long_Segment] for par in range(1,NumParticipants+1) for mode in Mode])
    pool.close()
    pool.join()






    
    
