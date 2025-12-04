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
import FindAllMovingPointsSynthetic as fmps
import Parameters as parm

#Global parameters
Time_Tollerance=0  # seconds
Time_Short=5 #seconds
Time_Long=7 #seconds
global Duration_Short_Segment
global Duration_Long_Segment
Duration_Short_Segment=Time_Short-Time_Tollerance
Duration_Long_Segment=Time_Long-Time_Tollerance     

       
global START
global END
global STEP
START=10000
END=22000
STEP=50
    







    
# [X,Y,T]=r.ReadCoordinates(2,'negative')





# Best=hor.FindHorizontalMovingPoint(X, Y, T,Duration_Segment)
# Best=ver.FindVerticalMovingPoint(X, Y, T,Duration_Segment)
# result=mp.FindMovingPointHorizontal(X, Y, T, Duration_Short_Segment)




# [ShortHorizontalLR, LongHorizontalLR, LongHorizontalRL, ShortVerticalDU, ShortVerticalUD]=fmp.FindAllMovingPointsHelp(X, Y, T, Duration_Short_Segment,Duration_Long_Segment)
# result=[ShortHorizontalLR, LongHorizontalLR, LongHorizontalRL, ShortVerticalDU, ShortVerticalUD]
# names=['ShortHorizontalLR', 'LongHorizontalLR', 'LongHorizontalRL', 'ShortVerticalDU', 'ShortVerticalUD']
# for numb in range(5):
#     filenameExport=str(names[numb]+'.csv')
#     with open(filenameExport, 'w') as f:
#         writer = csv.writer(f)
#         for row in result[numb]:
#             writer.writerow(row)    








# [P1,P2,P3,P4,P5,P6]=fmp.FindAllMovingPointsSquare(X, Y, T, Duration_Short_Segment,Duration_Long_Segment,[Duration_Long_Segment, Duration_Short_Segment,Duration_Long_Segment,Duration_Short_Segment])
# P=[P1,P2,P3,P4,P5,P6]
# times=[]
# for p in P:
#     times.append([p[1],p[2]])
    
# filenameExport=str('POINTS'+'.csv')
# with open(filenameExport, 'w') as f:
#     writer = csv.writer(f)
#     for row in P:
#         writer.writerow(row)    
    
    
    
    
    
    
    
    
    
   
# ver.FindMovingPointVerticalUD(X, Y, T, Duration_Short_Segment)
    
    
# LongHorizontalRL=mp.FindAllMovingPointsHelp(X, Y, T, Duration_Short_Segment,Duration_Long_Segment)
    
    
    
    

if __name__ == "__main__": 
    fmps.FindAllMovingPointsSquarePeople(parm.NPeople,Duration_Short_Segment,Duration_Long_Segment)
    
    
    
    
