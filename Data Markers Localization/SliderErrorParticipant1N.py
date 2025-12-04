import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import random
import copy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from matplotlib.widgets import Slider, Button, RadioButtons
import time
import ReadData as r
import FindHorizontalMovingPoint as hor
import HelpFunctions as h




#Short
FixStart=0.089447805
FixEnd=0.906706166
Start=18236
duration=6.8

#Long
FixStart=0.089447805
FixEnd=0.906706166
Start=18236
duration=6.8









fig, ax = plt.subplots()

plt.subplots_adjust(bottom=0.6)
ax.set_xlim(0,1)
ax.set_ylim(0,1)


def ReadCoordinates():
    
    filenameImport='Data/Participant1_negative.csv'
    FullData= list(csv.reader(open(filenameImport)))
    FullData.pop(0)
    N=len(FullData)
    FullData=np.transpose(FullData)
    X=FullData[6-1].copy()
    X=h.Float(X)
    Y=FullData[7-1].copy()
    Y=h.Float(Y)
    T=FullData[4-1].copy()
    T=h.Float(T)
    
    [X,Y]=r.PutIntoFrame(X,Y)
    
    
    
    return[X,Y,T]         
    
    
[X,Y,T]=ReadCoordinates()


i=Start
while T[i]-T[Start]<=duration:
    i=i+1
End=i-1

EyeX=[]
EyeY=[]
Time=[]
for i in range(Start,End+1):
    EyeX.append(X[i])
    EyeY.append(Y[i])
    Time.append(T[i])
    
MovingY=0
for y in EyeY:
    MovingY+=y**2
MovingY=MovingY**(1/2)/len(EyeY)

start=10
end=100
length=end-start+1

N=len(EyeX)
time_interval=Time[N-1]-Time[0]
step=(FixEnd-FixStart)/time_interval





eyeX=[]
eyeY=[]
time=[]
for i in range(length):
    eyeX.append(EyeX[start+i])
    eyeY.append(EyeY[start+i])
        



axcolor = 'lightgoldenrodyellow'
axStart = plt.axes([0.1, 0.25, 0.75, 0.03], facecolor=axcolor)
axEnd = plt.axes([0.1, 0.2, 0.75, 0.03], facecolor=axcolor)

axFrameLeft = plt.axes([0.1, 0.3, 0.75, 0.03], facecolor=axcolor)
axFrameRight = plt.axes([0.1, 0.35, 0.75, 0.03], facecolor=axcolor)
axFrameDown = plt.axes([0.1, 0.4, 0.75, 0.03], facecolor=axcolor)
axFrameUp = plt.axes([0.1, 0.45, 0.75, 0.03], facecolor=axcolor)

sStart = Slider(axStart, 'Start', 1, End-Start, valinit=start, valstep=1)
sEnd = Slider(axEnd, 'End', 1, End-Start, valinit=end, valstep=1)

sLeft = Slider(axFrameLeft, 'Left', -2,3, valinit=0, valstep=0.1)
sRight = Slider(axFrameRight, 'Right', -2,3, valinit=1, valstep=0.1)
sDown = Slider(axFrameDown, 'Down', -2,3, valinit=0, valstep=0.1)
sUp = Slider(axFrameUp, 'Up', -2,3, valinit=1, valstep=0.1)

ax.scatter(np.array(eyeX),np.array(eyeY),s=1.6)

def update(val):
    start = int(sStart.val)
    end = int(sEnd.val)
    
    ax.clear()
    
    length=end-start+1
    global eyeX
    global eyeY
    eyeX=[]
    eyeY=[]
    time=[]
    for i in range(length):
        eyeX.append(EyeX[start+i])
        eyeY.append(EyeY[start+i])
        time.append(Time[start+i])
     
        
    Left=sLeft.val
    Right=sRight.val
    Down=sDown.val
    Up=sUp.val
    ax.set_xlim(Left,Right)
    ax.set_ylim(Down,Up)
    ax.scatter(np.array(eyeX),np.array(eyeY),s=1.6)
    # plt.xlim(7,9)
    # axes.Axes.set_xlim(4,7)
    
    global axPoint
    axPoint.clear()
    # axPoint = plt.axes([0.1, 0.5, 0.75, 0.03], facecolor=axcolor)
    global sPoint
    sPoint = Slider(axPoint, 'Point', 1, length, valinit=10, valstep=1)
    sPoint.on_changed(update2)
    
    global MovingX
    global MovingY
    MovingX=FixStart+(time[10]-Time[0])*step    
    ax.scatter(MovingX,MovingY,s=20.6,color="orange")
    
    plt.show()


sStart.on_changed(update)
sEnd.on_changed(update)
sLeft.on_changed(update)
sRight.on_changed(update)
sDown.on_changed(update)
sUp.on_changed(update)






StartLeftAx = plt.axes([0.1, 0.13, 0.1, 0.04])
buttonStartLeft = Button(StartLeftAx, 'Start-', color=axcolor, hovercolor='0.975')
def StartLeft(d):
    sStart.set_val(sStart.val-sStep.val)
buttonStartLeft.on_clicked(StartLeft)


StartRightAx = plt.axes([0.8, 0.13, 0.1, 0.04])
buttonStartRight = Button(StartRightAx, 'Start+', color=axcolor, hovercolor='0.975')
def StartRight(d):
    sStart.set_val(sStart.val+sStep.val)
buttonStartRight.on_clicked(StartRight)


EndLeftAx = plt.axes([0.1, 0.07, 0.1, 0.04])
buttonEndLeft = Button(EndLeftAx, 'End-', color=axcolor, hovercolor='0.975')
def EndLeft(d):
    sEnd.set_val(sEnd.val-sStep.val)
buttonEndLeft.on_clicked(EndLeft)


EndRightAx = plt.axes([0.8, 0.07, 0.1, 0.04])
buttonEndRight = Button(EndRightAx, 'End+', color=axcolor, hovercolor='0.975')
def EndRight(d):
    sEnd.set_val(sEnd.val+sStep.val)
buttonEndRight.on_clicked(EndRight)






StepAx = plt.axes([0.1, 0.01, 0.75, 0.04])
sStep = Slider(StepAx, 'Step', 1, 20, valinit=1, valstep=1)
sStep.on_changed(update)
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# def reset(event):
#     sStart.reset()
#     sEnd.reset()
# button.on_clicked(reset)




# def colorfunc(label):
#     l.set_color(label)
#     fig.canvas.draw_idle()






def update2(val):
    N=sPoint.val
    ax.scatter(eyeX[N],eyeY[N],s=20.6,color="red")
    

axPoint = plt.axes([0.1, 0.5, 0.75, 0.03], facecolor=axcolor)
sPoint = Slider(axPoint, 'Point', 1, End-Start, valinit=start, valstep=1)
sPoint.on_changed(update2)

# hor.ErrMovingPointHorizontalRL(EyeX,EyeY,Time,FixStart,FixEnd)
plt.show()