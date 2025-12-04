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
from matplotlib.widgets import Slider, Button, RadioButtons
import time
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)


def Float(X):
    New=[]
    for el in X:
        New.append(float(el))
    return New
       
def ReadCoordinates():
    
    filenameImport='Data/Participant4_negative.csv'
    FullData= list(csv.reader(open(filenameImport)))
    FullData.pop(0)
    N=len(FullData)
    FullData=np.transpose(FullData)
    X=FullData[6-1].copy()
    X=Float(X)
    Y=FullData[7-1].copy()
    Y=Float(Y)
    T=FullData[5-1].copy()
    T=Float(T)
    return [X,Y,T] 
        
        
    
    
    
[X,Y,T]=ReadCoordinates()


EyeX=X
EyeY=Y

start=3
end=10000
length=end-start+1

eyeX=[]
eyeY=[]
for i in range(length):
    eyeX.append(EyeX[start+i])
    eyeY.append(EyeY[start+i])
        
    
Start=1
End=35000
    
# fig, ax = plt.subplots()
# DataPd=pd.DataFrame({'x':eyeX,'y':eyeY},columns=['x','y'])
# DataPd.plot.scatter(x='x',y='y',s=1.6,grid=1,title='')



# l, = plt.plot(t, s, lw=2)
# ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axStart = plt.axes([0.1, 0.25, 0.75, 0.03], facecolor=axcolor)
axEnd = plt.axes([0.1, 0.2, 0.75, 0.03], facecolor=axcolor)

sStart = Slider(axStart, 'Start', 1, End-Start, valinit=start, valstep=1)
sEnd = Slider(axEnd, 'End', 1, End-Start, valinit=end, valstep=1)


ax.scatter(np.array(eyeX),np.array(eyeY),s=1.6)

def update(val):
    start = int(sStart.val)
    end = int(sEnd.val)
    
    ax.clear()
    
    length=end-start+1
    eyeX=[]
    eyeY=[]
    for i in range(length):
        eyeX.append(EyeX[start+i])
        eyeY.append(EyeY[start+i])
       
    
    ax.scatter(np.array(eyeX),np.array(eyeY),s=1.6)
   




sStart.on_changed(update)
sEnd.on_changed(update)



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


plt.show()