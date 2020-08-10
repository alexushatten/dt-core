#!/usr/bin/env python

import cv2
import numpy as np
import os
import os.path
import numpy
import tf

from OpenGL.GL import *
import glm
from textureLoader import textureLoader
from glutWindow import GlutWindow
from shaderLoader import Shader
from dt_apriltags import Detector
from objLoader import objLoader
from MVPControl import MVPController
from worldsheet import worldSheet
from uv2d import UV2D

from PIL import Image
from PIL import ImageOps
import math

duckiebot = True
PATH_TO_STORE_IMG = os.path.join(os.path.abspath("."), 'dataset_generator/raw')
ROTATION_RANGES = ([-0.1 * np.pi, -0.1 * np.pi, -0.1 * np.pi], [0.1 * np.pi, 0.1 * np.pi, 0.1 * np.pi])
TRANSLATION_RANGES = ([-0.2, -0.2, 0.05], [0.2, 0.2, 0.5])

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


####################################################################

output_data = {
    'idx': list(),
    'tag_id': list(),
    'tvec': list(),
    'rvec': list(),
    'fx': list(),
    'fy': list(),
    'cx': list(),
    'cy': list(),
    'k1': list(),
    'k2': list(),
    'p1': list(),
    'p2': list(),
    'k3': list(),
    'p11': list(),
    'p12': list(),
    'p13': list(),
    'p21': list(),
    'p22': list(),
    'p23': list(),
    'p31': list(),
    'p32': list(),
    'p33': list()}

_placementx = 0.0
_placementy = 0.0
_placementz = 0.0

g_vertex_buffer_data = [
		-0.0325 +_placementx,-0.0325+_placementy,-0.0325+_placementz,
		-0.0325+_placementx,-0.0325+_placementy, 0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		-0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy, 0.0325+_placementz,
		-0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		-0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		-0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		 0.0325+_placementx,-0.0+_placementy, 0.0325+_placementz,
		-0.0325+_placementx,-0.0+_placementy, 0.0325+_placementz,
		-0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		-0.0325+_placementx,-0.0325+_placementy, 0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy, 0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy,-0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy, 0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy,-0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		 0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		-0.0325+_placementx, 0.0+_placementy, 0.0325+_placementz,
		 0.0325+_placementx,-0.0325+_placementy, 0.0325+_placementz]

g_uv_buffer_data = [
		0.000059, 1.0-0.000004, 
		0.000103, 1.0-0.336048, 
		0.335973, 1.0-0.335903, 
		1.000023, 1.0-0.000013, 
		0.667979, 1.0-0.335851, 
		0.999958, 1.0-0.336064, 
		0.667979, 1.0-0.335851, 
		0.336024, 1.0-0.671877, 
		0.667969, 1.0-0.671889, 
		1.000023, 1.0-0.000013, 
		0.668104, 1.0-0.000013, 
		0.667979, 1.0-0.335851, 
		0.000059, 1.0-0.000004, 
		0.335973, 1.0-0.335903, 
		0.336098, 1.0-0.000071, 
		0.667979, 1.0-0.335851, 
		0.335973, 1.0-0.335903, 
		0.336024, 1.0-0.671877, 
		1.000004, 1.0-0.671847, 
		0.999958, 1.0-0.336064, 
		0.667979, 1.0-0.335851, 
		0.668104, 1.0-0.000013, 
		0.335973, 1.0-0.335903, 
		0.667979, 1.0-0.335851, 
		0.335973, 1.0-0.335903, 
		0.668104, 1.0-0.000013, 
		0.336098, 1.0-0.000071, 
		0.000103, 1.0-0.336048, 
		0.000004, 1.0-0.671870, 
		0.336024, 1.0-0.671877, 
		0.000103, 1.0-0.336048, 
		0.336024, 1.0-0.671877, 
		0.335973, 1.0-0.335903, 
		0.667969, 1.0-0.671889, 
		1.000004, 1.0-0.671847, 
		0.667979, 1.0-0.335851
	]

family = "tag36h11"
nthreads = 1
quad_decimate = 1.0
quad_sigma = 0.0
refine_edges = 1
decode_sharpening = 0.25
_tag_size = 0.0527647

_at_detector = Detector(
            families=family,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            # TODO: remove this
            searchpath=['/code/catkin_ws/devel/lib/']
            # TODO: remove this
        )

class Tu01Win(GlutWindow):

	class GLContext(object):
		pass
	def init_opengl(self):
		glClearColor(0.0,0,0.4,0)
		glDepthFunc(GL_LESS)
		glEnable(GL_DEPTH_TEST)
		#glEnable(GL_CULL_FACE)

	def init_context(self):
		self.context = self.GLContext()

		# vertex = glGenVertexArrays(1) # pylint: disable=W0612
		# glBindVertexArray(vertex)

		self.shader = shader = Shader()

		shader.initShaderFromGLSL(["glsl-files/vertex.glsl"],["glsl-files/fragment.glsl"])

		self.context.MVP_ID   = glGetUniformLocation(shader.program,"MVP")
		self.context.TextureID =  glGetUniformLocation(shader.program, "myTextureSampler")

		texture = textureLoader("resources/uvtemplate.tga")
		#texture = textureLoader("resources/tu02/uvtemplate.dds")

		self.context.textureGLID = texture.textureGLID

		self.context.vertexbuffer  = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER,self.context.vertexbuffer)
		glBufferData(GL_ARRAY_BUFFER,len(g_vertex_buffer_data)*4,(GLfloat * len(g_vertex_buffer_data))(*g_vertex_buffer_data),GL_STATIC_DRAW)

		if(texture.inversedVCoords):
			for index in range(0,len(g_uv_buffer_data)):
				if(index % 2):
					g_uv_buffer_data[index] = 1.0 - g_uv_buffer_data[index]
 		
		self.context.uvbuffer  = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER,self.context.uvbuffer)
		glBufferData(GL_ARRAY_BUFFER,len(g_uv_buffer_data)*4,(GLfloat * len(g_uv_buffer_data))(*g_uv_buffer_data),GL_STATIC_DRAW)

        self.UV2d.bindUV(self.context.uvbuffer,self.context.elementbuffer,self.context.elementbufferSize)

	def calc_MVP(self,width=IMAGE_RESOLUTION[1],height=IMAGE_RESOLUTION[0],x=0, z=0):
		fovy = 2*np.arctan(0.5*height/_fy)*180/np.pi
		#fovy = 160
		aspect = (width*_fy)/(height*_fx)
		print(fovy)
		far_clip = 100
		near_clip = 0.00001

		self.context.Projection = glm.perspective(glm.radians(fovy),aspect,near_clip,far_clip)
		self.context.View =  glm.lookAt(glm.vec3(x, 0.30, z+0.00001), # Camera is at (4,3,-3), in World Space
						glm.vec3(x,0,z), #and looks at the (0.0.0))
						glm.vec3(0,-1,0) ) #Head is up (set to 0,-1,0 to look upside-down)
		#fixed Cube Size
		self.context.Model=  glm.mat4(1.0)
		#print self.context.Model
		self.context.MVP =  self.context.Projection * self.context.View * self.context.Model	

	def resize(self,Width=IMAGE_RESOLUTION[1],Height=IMAGE_RESOLUTION[0], x=0, z=0):
		
		glViewport(0, 0, Width, Height)
		self.calc_MVP(Width, Height, x, z)

	def ogl_draw(self,width=IMAGE_RESOLUTION[1],height=IMAGE_RESOLUTION[0]):

		print ("new frame")
		#print self.context.MVP
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |  GL_STENCIL_BUFFER_BIT)

		self.shader.begin()
		glUniformMatrix4fv(self.context.MVP_ID,1,GL_FALSE,glm.value_ptr(self.context.MVP))


		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.context.textureGLID)
		glUniform1i(self.context.TextureID, 0)


		glEnableVertexAttribArray(0)
		glBindBuffer(GL_ARRAY_BUFFER, self.context.vertexbuffer)
		glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

		glEnableVertexAttribArray(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.context.uvbuffer)
		glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,0,None)


		glDrawArrays(GL_TRIANGLES, 0, 12*3) # 12*3 indices starting at 0 -> 12 triangles

		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(1)

		data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, None)
		image = Image.frombytes("RGB", (width, height), data)
		#image = ImageOps.fit(image,(width, height))
		image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
		open_cv_image = numpy.array(image) 
		# Convert RGB to BGR 
		open_cv_image = open_cv_image[:, :, ::-1].copy() 
		open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
		camera_parameters = (_fx, _fy, _cx, _cy)
		tags = _at_detector.detect(open_cv_image, True, camera_parameters, _tag_size)

		for tag in tags:
			# turn rotation matrix into quaternion
			q = _matrix_to_quaternion(tag.pose_R)
			p = tag.pose_t.T[0]
			print(q)
			print(p)
		
		K = (_fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0)
		D = (_k1, _k2, _p1, _p2, _k3)
		P = (_p11, _p12, _p13, 0, _p21, _p22, _p23, 0, _p31, _p32, _p33, 0)
		
		_at_detector.enable_rectification_step(width, height, K, D, P)

		tags = _at_detector.detect(open_cv_image, True, camera_parameters, _tag_size)
		for tag in tags:
			# turn rotation matrix into quaternion
			q = _matrix_to_quaternion(tag.pose_R)
			p = tag.pose_t.T[0]
			print(q)
			print(p)

		self.shader.end()

def _matrix_to_quaternion(R):
    T = np.array((
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 1)
    ), dtype=np.float64)
    T[0:3, 0:3] = R
    return tf.transformations.quaternion_from_matrix(T)

if __name__ == "__main__":
	win = Tu01Win()
	win.init_opengl()
	win.init_context()
	win.run()