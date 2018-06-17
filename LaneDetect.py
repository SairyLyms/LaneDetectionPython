import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.optimize import curve_fit
import cv2
import math


def ReadVideoFrameBinary(frame):
    low_threshold = 100
    high_threshold = 200    
    
    frameHSV = cv2.cvtColor(frame[300:,:,:], cv2.COLOR_BGR2HSV)
    frame = frameHSV[:,:,2]
    displayImage(frame)
    frame = cv2.convertScaleAbs(cv2.Sobel(frame, cv2.CV_32F, 1,0,ksize=3))
    frame = ipm(frame)
    frame = cv2.convertScaleAbs(frame) > 80
    
    #displayImage(frame)
    #displayScatter(lanePixelsL[0],lanePixelsL[1])
    
    return frame

def displayImage(img):
    plt.figure(figsize=(12,8))
    plt.imshow(img)
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.gray()
    plt.show()
    
def displayScatter(x,y):
    plt.figure(figsize=(12,8))
    plt.scatter(x,y)
    plt.axis('equal')
    plt.show()

def TopViewTransform(frame,focalLen,tilt,height):
    x = 200
    y = 300
    xMax = 10
    yMax = 5
    
    hvc = 5
    fvc = 600
    dvc = 5
    
    tx = fvc/hvc * height * x/(focalLen*math.sin(tilt)-y*math.cos(tilt))
    ty = fvc/hvc * height * ((focalLen*math.cos(tilt) + y*math.sin(tilt))/(focalLen*math.sin(tilt)-y*math.cos(tilt)) - dvc)

#    tx = focalLen * x/(y*math.cos(tilt) + height*math.sin(tilt)) + xMax/2
#    ty = focalLen * (height * math.cos(tilt) - y * math.sin(tilt))/(y*math.cos(tilt) + height*math.sin(tilt)) + yMax/2

#    tx = focalLen * x/(y*math.cos(tilt) + height*math.sin(tilt)) + xMax/2
#    ty = focalLen * (height * math.cos(tilt) - y * math.sin(tilt))/(y*math.cos(tilt) + height*math.sin(tilt)) + yMax/2
    
#    d = abs(height*(math.sin(tilt)+focalLen+math.cos(tilt))/(focalLen*math.sin(tilt)-math.cos(tilt)))+1
#    tx = height*((x*math.sin(tilt) + focalLen * math.cos(tilt))/(-y*math.cos(tilt)+focalLen*math.sin(tilt))) + d
#    ty = height*((y*math.sin(tilt) + focalLen * math.cos(tilt))/(-y*math.cos(tilt)+focalLen*math.sin(tilt))) + d
    
def ipm(img):
    ptssrc = np.float32([[395,50],[80,200],[575,50],[900,200]])
    ptsdst = np.float32([[0,0],[0,512],[512,0],[512,512]])
    MTrans = cv2.getPerspectiveTransform(ptssrc,ptsdst)
    img = cv2.warpPerspective(img,MTrans,(512,512))
    return img

def SplitLRLanePixcel(frame):
   plt.imshow(frame)
   plt.show()
   frame =np.flipud(frame)
   lanePixelsL = list(np.where(frame[:,0:255] == 1))
   lanePixelsL[1] = lanePixelsL[1]
   #lanePixelsL[1] = lanePixelsL[1]-np.mean(lanePixelsL[1][lanePixelsL[0] < (np.min(lanePixelsL[0]) + 10)])
   lanePixelsR = list(np.where(frame[:,256:512] == 1))
   lanePixelsR[1] = lanePixelsR[1]
   #lanePixelsR[1] = lanePixelsR[1]-np.mean(lanePixelsR[1][lanePixelsR[0] < (np.min(lanePixelsR[0]) + 10)])
   return  lanePixelsL , lanePixelsR


def ejphi(phi0,phiv,phiu,s):
    return np.exp((phiu*s*s+phiv*s+phi0)*1j)
def ClothoidSimpon(h,s,phi0,phiv,phiu):
    s=s/h
    return h*((s-0)/6*(ejphi(phi0,phiv,phiu,0) + 4*ejphi(phi0,phiv,phiu,s/2) + ejphi(phi0,phiv,phiu,s)))
def FitFuncClothoid(s,y0,phi0,phiv,phiu):
    return np.imag(ClothoidSimpon(512,s,phi0,phiv,phiu)) + y0

def FitFuncClothoidPhiv(s,y0,phi0,phiv):
    return np.imag(ClothoidSimpon(512,s,phi0,phiv,0)) + y0

def PolyFitClothoid(lanePixelsX, lanePixelsY):
    param_init =[0,0,0,0]         # フィッティングパラメータ　初期条件
    param_opt =[0,0,0,0]         # フィッティングパラメータ　フィッティング後
    param_opt, cov = curve_fit(FitFuncClothoid, lanePixelsX, lanePixelsY, param_init)
    if len(lanePixelsX[0][lanePixelsX[0] == 0]) > 3: #手前に初期値となる点がない場合
            param_opt[0:3], cov = curve_fit(FitFuncClothoidPhiv, lanePixelsX, lanePixelsY, param_init[0:3]) #フィッティング曲線のうねり防止のためクロソイド縮率は0とする
    #For debug plot
            #plt.scatter(lanePixelsX,lanePixelsY)
    plt.scatter(lanePixelsY,lanePixelsX)
    #plt.scatter(np.arange(0,512,1),FitFuncClothoid(np.arange(0,512,1),para_opt[0],para_opt[1],para_opt[2],para_opt[3]),color='RED')
    plt.scatter(FitFuncClothoid(np.arange(0,512,1),param_opt[0],param_opt[1],param_opt[2],param_opt[3]),np.arange(0,512,1),color='RED')
    #plt.xlim(0,255)
    plt.axis('equal')
    param_opt[2:4] = [-param_opt[2], -param_opt[3]]
    return param_opt #Clothoid param(fitted)

cap = cv2.VideoCapture('solidYellowLeft.mp4')

laneParamL = []
laneParamR = []

"""
while(True):
    if cap.isOpened():
        #cap.set(0,12000)
        rval , frame = cap.read()
    frame = ReadVideoFrameBinary(frame)
    lanePixelsL , lanePixelsR = SplitLRLanePixcel(frame)    
    laneParamL.append(PolyFitClothoid(lanePixelsL[0], lanePixelsL[1]))
    laneParamR.append(PolyFitClothoid(lanePixelsR[0], lanePixelsR[1]))
    print(laneParamL[-1][2])
    plt.show()
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""

cap.release()
cv2.destroyAllWindows()