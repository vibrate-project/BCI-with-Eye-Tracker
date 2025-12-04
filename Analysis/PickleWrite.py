# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:07:15 2024

@author: teodo
"""




# #START
# DirName='Pickle'
# NNNamesJu=list(locals().keys())
# NNNamesJu.append('NNNamesJu')
# import pickle
# import os
# MainDir=os.getcwd()    
# os.makedirs(DirName,exist_ok=True)
# os.chdir('./'+DirName)

# for name in NNNamesJu:
#     value=locals()[name]
#     with open(name+'.pkl', 'wb') as f:
#         pickle.dump(value,f)    
# os.chdir(MainDir)
# #END         
        
        
        
        
#START
DirName='Pickle4'
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
        
        
        
        
        
        
        
        
        