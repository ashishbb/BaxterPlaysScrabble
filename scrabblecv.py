import sys
import numpy as np
import cv2
from PIL import Image
cam = cv2.VideoCapture(0)
import scipy.misc
import Classify
import time
import scipy

# from matplotlib import pyplot as plt



def rotateImage(image, angle):
        row,col = image.shape
        center=tuple(np.array([row,col])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        new_image = cv2.warpAffine(image, rot_mat, (col,row))
        return new_image


def getrack(im):
    ##############TEMPORARY
    #globalim = bridge.imgmsg_to_cv2(data, "bgr8")
    #cv2.imshow("R", im)
    #cv2.waitKey(0)
    ################
    

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    lower_blue = np.array([0/2,(0)*2.55,(0)*2.55])
    upper_blue = np.array([(360)/2,(16)*2.55,(15)*2.55])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    kernel = np.ones((2,2),np.uint8)
    morph2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 2)
    # kernel = np.ones((1,2),np.uint8)
    # morph = cv2.morphologyEx(morph2, cv2.MORPH_OPEN, kernel,iterations = 1)
    # kernel = np.ones((2,2),np.uint8)
    # morph2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 2)
    # kernel = np.ones((2,2),np.uint8)
    # morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 1)
    
    # cropim = im[140:380,100:550].copy()
    imw = rotateImage(morph2,3)
    print imw.shape
    imw= imw[219:270,34:546].copy()
    timw = cv2.adaptiveThreshold(imw,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    print imw.shape
    # cv2.imshow("Result", imw)
    # cv2.waitKey(0)
    #print imw.shape
    rack_array = []
    classified = []
    start = time.time()
    for cut in range(0,8):
        i = imw[:,cut*63:(1+cut)*63].copy()
        # cv2.imshow("cut",i)
        # cv2.waitKey(0)
        #print i
        backtorgb = cv2.cvtColor(i,cv2.COLOR_GRAY2RGB)
        rack_array.append(backtorgb)
        scipy.misc.imsave("9.jpg",backtorgb)
        #classified.append(dl_classify.classify([backtorgb]))

    #classified.append(dl_classify([None]))
    
    elap = time.time() - start
    print elap
    #return (pytesseract.image_to_string(Image.fromarray(imw)))


##############TEMPORARY
cl = Classify.Classify()
im = cv2.imread('Rack2.png',1)
#print im
print im.shape
getrack(im)

################

