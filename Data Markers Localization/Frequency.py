# import ReadData as r
import math

def MakeSignalEven(X,Y,T,Duration,Frequency=60):
    
    X_e=MakeSignalEvenHelp(X,T,Duration,Frequency)
    Y_e=MakeSignalEvenHelp(Y,T,Duration,Frequency)
    
    N=math.floor(Duration*Frequency)
    T_step=1/Frequency
    
    T_e=[]
    for i in range(N):
        T_e.append(i*T_step)
    
    return X_e,Y_e,T_e
    
    
    
    
    
    
def MakeSignalEvenHelp(X,T,Dur,Freq):
    N=math.floor(Dur*Freq)
    T_step=1/Freq
    
    X_e=[]
    time=0
    state=1
    for i in range(N):
        
        leftDist=time-(T[state-1]-T[0])
        rightDist=(T[state]-T[0])-time
        
        p=X[state-1]*rightDist/(leftDist+rightDist)+X[state]*leftDist/(leftDist+rightDist)
        X_e.append(p)
        
        time=time+T_step
        
        for j in range(state-1,len(T)):
            if T[j]-T[0]>time:
                state=j
                break
            
        
    return X_e




participant=1
mode='positive'
# X,Y,T=r.ReadSegments(participant,mode)[2]

# X_e,Y_e,T_e=MakeSignalEven(X,Y,T,7,Frequency=300)