# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:07:15 2024

@author: teodo
"""

import os
import pickle









import pickle
import os
DirName='Middle5'
MainDir=os.getcwd()    
os.chdir('./'+DirName)
i=15
with open('Weights'+str(i)+'.pkl', 'rb') as f:
    valueW=pickle.load(f)
# with open('grad'+str(i)+'.pkl', 'rb') as f:
#     valueG=pickle.load(f)
os.chdir(MainDir)







# #START
# DirName='Pickle4'
# MainDir=os.getcwd()    
# os.chdir('./'+DirName)

# with open('NNNamesJu.pkl', 'rb') as f:
#     NNNamesJu=pickle.load(f)

# for name in NNNamesJu:
#     with open(name+'.pkl', 'rb') as f:
#         value=pickle.load(f)
#     globals()[name]=value

# os.chdir(MainDir)
# #END
        

        
# for j in range(len(LongHorizontalLR)):
#     start=LongHorizontalLR[j][6]
#     print(abs(T[SquareSegments[0][0]]-T[start]))
#     if abs(T[SquareSegments[0][0]]-T[start])<2:         
#         P3=LongHorizontalLR[j]
#         break
        
        
        
        
        
        