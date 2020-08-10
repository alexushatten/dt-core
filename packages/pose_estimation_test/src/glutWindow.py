
from __future__ import division
import OpenGL.GLUT as oglut
import sys
import OpenGL.GL as gl
import OpenGL.GLU as glu
class GlutWindow(object):

    def init_opengl(self):
        gl.glClearColor(0.0,0,0.4,0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_DEPTH_TEST)
    
    def display(self):    
        self.ogl_draw()
        oglut.glutSwapBuffers()
    def idle(self):
        pass    

    def on_keyboard(self,key,x,y):     
        if(self.controller!=None):
              self.controller.on_keyboard(key,x,y)
        else:
            print ("please overrider on_keyboard")

    def on_timer(self,pressed):     
        if(self.controller!=None):
              self.controller.on_timer(pressed)
        else:
            print ("please overrider on_timer")

    def on_special_key(self,key,x,y):     
        if(self.controller!=None):
              self.controller.on_special_key(key,x,y)
        else:
            print ("please overrider on_keyboard")         
        
    def on_mouse(self,*args,**kwargs):
        if(self.controller!=None):
              self.controller.on_mouse(*args,**kwargs)
        else:
            self.resize(x=0.1, z=0)     
            self.ogl_draw()
    def on_mousemove(self,*args,**kwargs):
        if(self.controller!=None):
              self.controller.on_mousemove(*args,**kwargs)
        else:                
            print ("please overrider on_mousemove")
                
    def __init__(self,*args,**kwargs):

        oglut.glutInit(sys.argv)
        oglut.glutInitDisplayMode(oglut.GLUT_RGBA | oglut.GLUT_DOUBLE | oglut.GLUT_DEPTH)
        oglut.glutInitWindowSize(640, 480)
        self.window = oglut.glutCreateWindow(b"window")
        oglut.glutDisplayFunc(self.display)
        #oglut.glutIdleFunc(self.display) 
        oglut.glutReshapeFunc(self.resize)  
        oglut.glutKeyboardFunc(self.on_keyboard)   
        oglut.glutSpecialFunc(self.on_special_key)  
        oglut.glutMouseFunc(self.on_mouse)
        oglut.glutMotionFunc(self.on_mousemove)
        self.controller = None
        self.update_if = oglut.glutPostRedisplay

    def run(self):
        oglut.glutMainLoop()



if __name__ == "__main__":

    win = GlutWindow()
    win.run()