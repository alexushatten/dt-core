

import glm
import math
import numpy as np
import math
import time
import csv


duckiebot = True
if duckiebot == False:
    IMAGE_RESOLUTION = (972, 1296)
    _fx = 639.928035083189
    _fy = 690.686664286865
    _cx = 641.536558113394
    _cy = 494.9626575073991
    _k1 = -0.25646120235827957
    _k2 = 0.04915146105142022
    _p1 = 0.0050461703714548565
    _p2 = -0.005462908287664201
    _k3 = 0.0
    _p11 = 211.56044006347656
    _p12 = 0.0
    _p13 = 318.87211753874
    _p21 = 0.0
    _p22 = 268.90362548828125
    _p23 = 231.14138426406498
    _p31 = 0.0
    _p32 = 0.0
    _p33 = 1.0 


else:
    IMAGE_RESOLUTION = (480, 640)
    _fx = 320.83628590652455
    _fy = 323.04325776720174
    _cx = 320.2332167518086
    _cy = 234.12811257055012
    _k1 = -0.24241406656348882
    _k2 = 0.0402747578682183
    _p1 = -5.477653022258039e-06
    _p2 = -0.0005012637588869646
    _k3 = 0.0
    _p11 = 211.56044006347656
    _p12 = 0.0
    _p13 = 318.87211753874
    _p21 = 0.0
    _p22 = 268.90362548828125
    _p23 = 231.14138426406498
    _p31 = 0.0
    _p32 = 0.0
    _p33 = 1.0

class MVPControl:

    def __init__(self,width=800,height=480,*args,**kwargs):

        self.ScreenWidth = IMAGE_RESOLUTION[1]
        self.ScreenHeight = IMAGE_RESOLUTION[0]
        self.reset()
        
    def reset(self):
        #Initial position : on +Z
        # 0.0398141,     0.313451,     0.115628
        #self.position = glm.vec3(0.0398141, 0.313451, 0.115628)
        self.position = glm.vec3(0.269814, 0.313451, 0.160811)
        self.positions = []
        #Initial horizontal angle : toward -Z
        self.XAngle = 3.14
        #Initial vertical angle : none
        self.YAngle = -1.65
        # Initial Field of View
        self.ZAngle = 0.0
        self.Fov = 2*np.arctan(0.5*self.ScreenHeight/_fy)*180/np.pi
        self.aspect = (self.ScreenWidth*_fy)/(self.ScreenHeight*_fx)
        self.computeMatrices()
        self.count=1

    def moveFoward(self,foward):
        self.position += self.direction*foward
        self.computeMatrices()
    def moveUp(self,up):    
        self.position += self.up*up
        self.computeMatrices()
    def moveRight(self,right):    
        self.position += self.right*right 
        self.computeMatrices()  
    def lookUpward(self,yaw):
        self.YAngle += yaw
        self.computeMatrices()  
    def turn(self,angle):
        self.XAngle += angle
        self.computeMatrices()  
    def replace(self,trigger):
        self.position = glm.vec3(0.269814, 0.313451, (0.160811-trigger))
        self.computeMatrices()  
        print(self.count)
        self.count += 1
    #calc direction right and up
    def computeMatrices(self):
        self.direction = glm.vec3(math.cos(self.YAngle) * math.sin(self.XAngle), 
                        math.sin(self.YAngle),
                        math.cos(self.YAngle) * math.cos(self.XAngle)
                        )
        self.right =  glm.vec3(
            math.sin(self.XAngle - 3.14/2.0), 
            0.0,
            math.cos(self.XAngle - 3.14/2.0 )
            )
        self.up = glm.cross(self.right,self.direction)

        self.lookPos = self.position+ self.direction

        self.ProjectionMatrix = glm.perspective(glm.radians(self.Fov), self.aspect, 0.1, 1000.0)
        
        self.ViewMatrix = glm.lookAt(self.position,           # Camera is here
                                     self.lookPos, # and looks here : at the same position, plus "direction"
                                     self.up                       # Head is up (set to 0,-1,0 to look upside-down)
                            )

        self.positions.append(self.position)
        print(self.position)
        with open('ground_truth.csv', 'w') as file:
            writer = csv.writer(file)
            j = 1
            writer.writerow(["posenr", "coordinate"])
            for i in self.positions:
                writer.writerow([j, i])
                j += 1
    def resize(self,width=0,height=0):
        self.ScreenWidth = width
        self.ScreenHeight = height
        self.computeMatrices()
    
    def calcMVP(self,modelMaterix):

        #print self.position
        #print self.XAngle,self.YAngle
        return self.ProjectionMatrix * self.ViewMatrix * modelMaterix                                  

def dummyUpdate():
    print "please implement update"
class MVPController(MVPControl):

    def __init__(self,updateCallback=dummyUpdate,*args,**kwargs):
        self.updateCallback =updateCallback
        MVPControl.__init__(self,*args,**kwargs)
        self.mouse_mode  = -1
        self.lastX =0 
        self.lastY =0 
        self.horizontal=0
        self.vertical=1
        self.step=0.005

        self.goright = True

    def on_special_key(self,key,x,y):  
        print key
        _key = key   
        if(_key==104): #page down
            self.moveUp(0.01) 
            self.updateCallback()
        elif(_key==105):
            self.moveUp(-0.01) #page up
            self.updateCallback()
        elif(_key==101):#up
            self.lookUpward(0.1)
            self.updateCallback()
        elif(_key==103):#down
            self.lookUpward(-0.1)
            self.updateCallback()             
        elif(_key==102):#right
            self.turn(0.1) 
            self.updateCallback()
        elif(_key==100): #left
            self.turn(-0.1) 
            self.updateCallback()  

    def on_keyboard(self,key,x,y):     
        _key = key.lower()
        if(_key=='w'):
            self.moveFoward(0.01) 
            self.updateCallback()
        elif(_key=='s'):
            self.moveFoward(-0.01)
            self.updateCallback()
        elif(_key=='a'):
            self.moveRight(-0.01) 
            self.updateCallback()
        elif(_key=='d'):
            self.moveRight(0.01)
            self.updateCallback()
        elif(_key=='y'):
            if self.vertical > 81:
                return
            if self.horizontal < 54:
                self.moveRight(-0.01) 
                self.updateCallback()
                self.horizontal += 1
            else:
                self.replace(self.vertical*self.step)
                self.updateCallback()
                self.horizontal = 0
                self.vertical += 1 
              
    def on_mouse(self,*args,**kwargs):

            (key,Up,x,y) = args
            if((key==0) & (Up == 0)):
                self.lastX = x
                self.lastY = y
                self.mouse_mode = 1
            elif((key==2) & (Up == 0)):
                self.lastX = x
                self.lastY = y
                self.mouse_mode = 2  
        
            else:
                self.lastX = -1
                self.lastY = -1    
                self.mouse_mode = -1           
            #print "please overrider on_mousemove" ,args
    def on_mousemove(self,*args,**kwargs):
            deltaX = self.lastX - args[0]
            deltaY = self.lastY - args[1]
            if(self.mouse_mode==1):
                (self.lastX,self.lastY) = args
                self.lookUpward(deltaY*0.01)
                self.turn(deltaX*0.01)
                self.updateCallback()
            elif(self.mouse_mode==2):
                (self.lastX,self.lastY) = args
                #self.lookUpward(deltaY*0.01)
                #print "."
                self.moveUp(-0.5*deltaX) #page up
                self.updateCallback()                
            #print "please overrider on_mousemove" ,args

