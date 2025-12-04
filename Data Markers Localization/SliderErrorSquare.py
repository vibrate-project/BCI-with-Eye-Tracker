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
import FindSquare as fsq





boundary=[[0.5053621843616481, 0.6663840608800004], [0.2970420815733442, 0.16476041446737255], [0.5598388551731813, 0.70077254612142], [0.813292724104087, 0.5550217139868824], [0.5864938870848676, 0.4256215068168123]]
Start=32026

boundary=[[0.49329815560543894, 0.004838335138205508], [0.4880472033793369, 0.26596706663460806], [0.49448702855021737, 0.5334341489247426], [0.5250161685624373, 0.47231525150329584], [0.3232230394172002, 0.6557046056481124]]
Start=15541

boundary=[[0.8198843053112322, 0.44974029311369274], [0.7675663517558304, 0.40184542182969074], [0.835067035769561, 0.13111368244424368], [0.4267719035706081, 0.1514057730869488], [0.23588890947149377, 0.7593313453143099]]
Start=33586


boundary=[[0.440240941405016, 0.788133680261112], [0.529239952822696, 0.9999990813819809], [0.5246901918353664, 0.10826017759616968], [0.5286717067074181, 0.04483365252117796], [0.3968992020840788, 0.19939732061314697]]
Start=14476


boundary=[[0.9006967493798559, 0.9447103002087699], [0.9376956559743694, 0.013694069117870938], [0.008711800805170285, 0.07635254411256218], [0.19161570129757138, 0.8252594343422517], [0.5499729278752984, 0.1569796720781314]]
Start=19741

boundary=[[0.1923465834445436, 0.7174512805892715], [0.8206092604412092, 0.9754936895372371], [0.9999976578788575, 0.06708430118713378], [0.03935864109393435, 0.10494205019113662], [0.04021223454616636, 0.6378444488706403]]
Start=19351

boundary=[[0.3492223991138054, 0.8357410324215169], [0.9358194906670125, 0.7521199925390962], [0.767479581851255, 0.016926504144175528], [9.380152751552956e-07, 0.27536456864595305], [0.19718345536574045, 0.612717325194361]]
Start=19441

boundary=[[0.33979740948041753, 0.384030571650233], [0.6141301092662258, 0.977850539803849], [0.9999977458466183, 0.3323729553162708], [0.3470561317794622, 0.04371224968568695], [0.11172272431176132, 0.30878601284562557]]
Start=19246

Start=14500
boundary=[[0.49144212032985934, 0.9599083757863648], [0.4932046216523383, 0.9599083757863648], [0.4932046216523383, 0.040701621754491504], [0.48906410182883236, 0.040701621754491504], [0.48906410182883236, 0.14267681496711962]]

Start=19400
boundary=[[0.24508015735667682, 0.9010980521099855], [0.922374028734559, 0.9010980521099855], [0.922374028734559, 0.0961752334868301], [1.7107095625174013e-06, 0.0961752334868301], [1.7107095625174013e-06, 0.5916155768624227]]

Start=19381
boundary=[[0.07617879069285242, 0.8990101953748141], [0.9393681754419634, 0.8990101953748141], [0.9393681754419634, 0.06060999060059337], [0.0004883043840631603, 0.06060999060059337], [0.0004883043840631603, 0.9479029235433253]]

Start=19381
boundary=[[0.07617879069285242, 0.8990101953748141], [0.9393681754419634, 0.8990101953748141], [0.9393681754419634, 0.06060999060059337], [0.0004883043840631603, 0.06060999060059337], [0.0004883043840631603, 0.9479029235433253]]

boundary=[[0.47056842915624914, 0.6321639212800655], [0.7215466599883799, 0.6321639212800655], [0.7215466599883799, 0.4356938974326029], [0.7834645647852427, 0.4356938974326029], [0.7834645647852427, 0.3320159533888849]]
Start=32650




[X,Y,T]=r.ReadCoordinates(4,'negative')
times=[7,5,7,5]

duration=0
for t in times:
    duration+=t
[EyeX,EyeY,Time]=fsq.CutSegment(X,Y,T,Start,duration)



SegmentsMarks=[]
previous=0
S=len(times)
N=len(EyeX)

for j in range(S):
    for i in range(previous,N):
        if Time[i]>=Time[previous]+times[j]:
            SegmentsMarks.append([previous,i])
            previous=i+1
            break
               
if len(SegmentsMarks)<S:
    SegmentsMarks.append([previous,N-1])
    
SegmentsPoints=[]
for i in range(S):
    interval=[boundary[i],boundary[i+1]]
    SegmentsPoints.append(interval)








fig, ax = plt.subplots()

plt.subplots_adjust(bottom=0.6)
ax.set_xlim(0,1)
ax.set_ylim(0,1)





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

global start
global end
start=10
end=100
length=end-start+1

N=len(EyeX)
time_interval=Time[N-1]-Time[0]
FixStart=SegmentsMarks[0][0]
FixEnd=SegmentsMarks[3][1]

step=(FixEnd-FixStart)/time_interval



global eyeX
global eyeY
global time

eyeX=[]
eyeY=[]
time=[]
for i in range(length):
    eyeX.append(EyeX[start+i])
    eyeY.append(EyeY[start+i])
        



axcolor = 'lightgoldenrodyellow'
axStart = plt.axes([0.1, 0.25, 0.75, 0.03], facecolor=axcolor)
axEnd = plt.axes([0.1, 0.2, 0.75, 0.03], facecolor=axcolor)

axFrameLeft = plt.axes([0.1, 0.30, 0.75, 0.03], facecolor=axcolor)
axFrameRight = plt.axes([0.1, 0.35, 0.75, 0.03], facecolor=axcolor)
axFrameDown = plt.axes([0.1, 0.40, 0.75, 0.03], facecolor=axcolor)
axFrameUp = plt.axes([0.1, 0.45, 0.75, 0.03], facecolor=axcolor)
# axFrameFix = plt.axes([0.1, 0.50, 0.75, 0.03], facecolor=axcolor)

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
    global time
    eyeX=[]
    eyeY=[]
    time=[]
    for i in range(length):
        eyeX.append(EyeX[start+i])
        eyeY.append(EyeY[start+i])
        time.append(Time[start+i])
     
    global Left
    global Right
    global Down
    global Up   
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
    sPoint = Slider(axPoint, 'Point', 0, length-1, valinit=10, valstep=1)
    sPoint.on_changed(update2)
    
    global MovingX
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
    ax.clear()
    ax.set_xlim(Left,Right)
    ax.set_ylim(Down,Up)
    ax.scatter(np.array(eyeX),np.array(eyeY),s=1.6)
    number=sPoint.val
    globNumber=number+start
    ax.scatter(eyeX[number],eyeY[number],s=20.6,color="red")
    NSegment=0
    for i in range(len(SegmentsMarks)):
        l=SegmentsMarks[i][0]
        r=SegmentsMarks[i][1]
        if l<=globNumber and globNumber<=r:
            NSegment=i
            
    localNumber=globNumber-SegmentsMarks[NSegment][0]
    
    LPointX=SegmentsPoints[NSegment][0][0]
    LPointY=SegmentsPoints[NSegment][0][1]
    RPointX=SegmentsPoints[NSegment][1][0]
    RPointY=SegmentsPoints[NSegment][1][1]
    distanceX=RPointX-LPointX
    distanceY=RPointY-LPointY
    duration=times[NSegment]
    speedX=distanceX/duration
    speedY=distanceY/duration
    
    # print(speedX)
    
    ti=time[number]-Time[SegmentsMarks[NSegment][0]]
    # print(ti)
    
    
    
    PoX=LPointX+speedX*ti
    PoY=LPointY+speedY*ti
    
    ax.scatter(PoX,PoY,s=20.6,color="Purple")

axPoint = plt.axes([0.1, 0.5, 0.75, 0.03], facecolor=axcolor)
sPoint = Slider(axPoint, 'Point', 0, End-Start, valinit=start, valstep=1)
sPoint.on_changed(update2)
# sPoint.on_changed(update)

# hor.ErrMovingPointHorizontalRL(EyeX,EyeY,Time,FixStart,FixEnd)
plt.show()