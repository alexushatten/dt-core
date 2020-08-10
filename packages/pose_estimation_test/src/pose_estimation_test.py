#!/usr/bin/env python
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))


import cv2
import numpy as np
import numpy
import os
import os.path
import tf
import csv
import sys

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
distortion = True

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

p = [[_p11,_p12,_p13], [_p21, _p22, _p23], [_p31, _p32, _p33]]
print (p)
iR = np.linalg.inv(p)
print (iR)
####################################################################
mapx = np.zeros((IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1]))
mapy = np.zeros((IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1]))
mapx_inv = np.zeros((IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1]))
mapy_inv = np.zeros((IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1]))
for i in range (IMAGE_RESOLUTION[0]):
	_x = i * iR[0][1] + iR[0][2]
	_y = i * iR[1][1] + iR[1][2]
	_w = i * iR[2][1] + iR[2][2]
	for j in range (IMAGE_RESOLUTION[1]):
		_x += iR[0][0]
		_y += iR[1][0]
		_w += iR[2][0]
		w = 1.0/_w 
		x = _x*w
		y = _y*w
		x2 = x * x
		y2 = y * y
		r2 = x2 + y2
		_2xy = 2 * x * y
		kr = 1 +((_k3*r2 + _k2)*r2 + _k1)*r2
		xd = x*kr + _p1*_2xy + _p2*(r2 + 2*x2)
		yd = y*kr + _p1*(r2 + 2*y2) + _p2*_2xy
		_u = _fx * xd + _cx
		_v = _fy * yd + _cy
		if _u < 0 or _u >= IMAGE_RESOLUTION[1]:
			continue
		if _v < 0 or _v >= IMAGE_RESOLUTION[0]:
			continue
		mapx[i][j] = _u
		mapy[i][j] = _v

		mapx_inv[int(math.floor(_v))][int(math.floor(_u))] = j
		mapy_inv[int(math.floor(_v))][int(math.floor(_u))] = i

mapx = np.float32(mapx)
mapy = np.float32(mapy)
mapx_inv = np.float32(mapx_inv)
mapy_inv = np.float32(mapy_inv)
numpy.set_printoptions(threshold=sys.maxsize)

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
            # TODO: rem
        )

if distortion == True:
	K = (_fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0)
	D = (_k1, _k2, _p1, _p2, _k3)
	P = (_p11, _p12, _p13, 0.0, _p21, _p22, _p23, 0.0, _p31, _p32, _p33, 0.0)

	_at_detector.enable_rectification_step(IMAGE_RESOLUTION[1],  IMAGE_RESOLUTION[0], K, D, P)

class Tu01Win(GlutWindow):

    class GLContext(object):
        pass
    def init_opengl(self):
        glClearColor(0.0,0.0,0.4,0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        #glEnable(GL_CULL_FACE)
        self.positions = []

    def init_context(self):
        self.context = self.GLContext()

		# vertex = glGenVertexArrays(1) # pylint: disable=W0612
		# glBindVertexArray(vertex)

        self.shader = shader = Shader()
        self.UV2d = UV2D()

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

        self.UV2d.bindUV(self.context.uvbuffer,12*3)


    def calc_MVP(self,width=0,height=0):
        if(width!=0):
            self.controller.resize(width,height)
        self.context.Model=  glm.mat4(1.0)
        self.context.MVP =  self.controller.calcMVP(self.context.Model)
        

    def resize(self,Width,Height):        
        glViewport(0, 0, Width, Height)
        self.calc_MVP(Width,Height)

    def ogl_draw(self):

        #print self.context.MVP
        self.calc_MVP()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.shader.begin()
        glUniformMatrix4fv(self.context.MVP_ID,1,GL_FALSE,glm.value_ptr(self.context.MVP))

        glActiveTexture(GL_TEXTURE0)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.vertexbuffer)
        glUniform1i(self.context.TextureID, 0)

        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.vertexbuffer)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)

        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.context.uvbuffer)
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,0,None)

        
        glBindBuffer(GL_ARRAY_BUFFER, self.context.uvbuffer)
        
        glDrawArrays(GL_TRIANGLES, 0, 12*3) # 12*3 indices starting at 0 -> 12 triangles

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

        data = glReadPixels(0, 0, IMAGE_RESOLUTION[1], IMAGE_RESOLUTION[0], GL_RGB, GL_UNSIGNED_BYTE, None)
        image = Image.frombytes("RGB", (IMAGE_RESOLUTION[1], IMAGE_RESOLUTION[0]), data)
		#image = ImageOps.fit(image,(width, height))
        image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
        open_cv_image = np.array(image) 
		# Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        distorted_image = cv2.remap(open_cv_image, mapx_inv, mapy_inv, cv2.INTER_LINEAR)
        filled_image = fill_gaps(distorted_image)
        undistorted_image = cv2.remap(filled_image, mapx, mapy, cv2.INTER_LINEAR)
        camera_parameters = (_fx, _fy, _cx, _cy)

        tags = _at_detector.detect(filled_image, True, camera_parameters, _tag_size)
        #cv2.imshow('fased',open_cv_image)
        #cv2.imshow('image',filled_image)
        #cv2.imshow('unimage',undistorted_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        found_tag = False
        for tag in tags:
            found_tag = True
			# turn rotation matrix into quaternion
            q = _matrix_to_quaternion(tag.pose_R)
            p = tag.pose_t.T[0]
            self.positions.append(p)
            print (p)
        if found_tag == False:
            p = "none"
            self.positions.append(p)
            print(p)

        print (self.positions)
        with open('vanilla_detector_no_distort.csv', 'w') as file:
            writer = csv.writer(file)
            j = 1
            writer.writerow(["posenr", "coordinate"])
            for i in self.positions:
                writer.writerow([j, i])
                j += 1

        self.shader.end()
        
        self.UV2d.draw()

def fill_gaps(image):
    for i in range(0, len(image[:,0])):
        for j in range(0, len(image[0,:])):
            if image[i,j] > 50 :
                continue
            else:
                if image[i - 1,j] > 50 and image[i + 1,j] > 50:
                    image[i,j] = (int(image[i - 1,j]) + int(image[i + 1,j]))/2 
                if image[i,j - 1] > 50 and image[i,j + 1] > 50:
                    image[i,j] = (int(image[i,j - 1]) + int(image[i,j + 1]))/2 
    return image




def _matrix_to_quaternion(R):
    T = np.array((
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 1)
    ), dtype=np.float64)
    T[0:3, 0:3] = R
    return tf.transformations.quaternion_from_matrix(T)

def GetBilinearPixel(imArr, posX, posY):
    out = numpy.zeros((imArr.shape[0], imArr.shape[1]), numpy.uint8)
    for r in range(imArr.shape[0]):
        for c in range(imArr.shape[1]):
            #Get integer and fractional parts of numbers
            modXi = int(posX[r][c])
            modYi = int(posY[r][c])
            modXf = posX[r][c] - modXi
            modYf = posY[r][c] - modYi
            modXiPlusOneLim = min(modXi+1,imArr.shape[1]-1)
            modYiPlusOneLim = min(modYi+1,imArr.shape[0]-1)
        
            bl = imArr[modYi, modXi]
            br = imArr[modYi, modXiPlusOneLim]
            tl = imArr[modYiPlusOneLim, modXi]
            tr = imArr[modYiPlusOneLim, modXiPlusOneLim]
    
            #Calculate interpolation
            b = modXf * br + (1. - modXf) * bl
            t = modXf * tr + (1. - modXf) * tl
            pxf = modYf * t + (1. - modYf) * b
            out[r][c] = int(pxf+0.5)

            """ modXi = int(posX[r][c])
            modYi = int(posY[r][c])
            if (r + modXi) >imArr.shape[1]:
                continue
            if (c + modYi) >imArr.shape[0]:
                continue
            if (r + modXi) <0:
                continue
            if (c + modYi) <0:
                continue
            modXf = int(r + modXi) -1
            modYf = int(c + modYi) -1
            out[r][c] = int(imArr[modYf, modXf]) """
        
    return out

def rectification (image,mapx, mapy):
	out = numpy.zeros((image.shape[0], image.shape[1]), numpy.uint8)
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			_x = mapx [y][x]
			_y = mapy [y][x]
			out[_y][_x] = image[y][x]
	return out

""" image_u8_t *image_u8_rectify(apriltag_detector_t *td, image_u8_t *image, float decimate){
    image_u8_t *dst = image_u8_create(image->width, image->height);
    // rectify
    for (int y = 0; y < image->height; y++) {
        for (int x = 0; x < image->width; x++) {
            int i = y * dst->stride + x;

            int sx = (int) (x * decimate);
            int sy = (int) (y * decimate);

            int _x = (int) (MATD_EL(td->mapx, sy, sx));
            int _y = (int) (MATD_EL(td->mapy, sy, sx));

            int _sx = (int) (_x / decimate);
            int _sy = (int) (_y / decimate);

            dst->buf[i] = image->buf[_sy * dst->stride + _sx];
        }
    }
    return dst;
} """

if __name__ == "__main__":

    win = Tu01Win()
    win.controller = MVPController(win.update_if)
    win.init_opengl()
    win.init_context()
    win.run()
