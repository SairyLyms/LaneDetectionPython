import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.optimize import curve_fit
import cv2
import math


def ReadVideoFrameBinary(frame):
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = frameHSV[:,:,2]
    frame = cv2.convertScaleAbs(cv2.Sobel(frame, cv2.CV_32F, 1,0,ksize=3))
    frame = cv2.convertScaleAbs(frame) > 80
    
    #displayImage(frame)
    #displayScatter(lanePixelsL[0],lanePixelsL[1])
    
    return frame

def displayImageLwr(img):
    #plt.figure(figsize=(12,8))
    plt.imshow(img,origin='lower')
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.gray()
    plt.show()

def displayImageUpr(img):
    #plt.figure(figsize=(12,8))
    plt.imshow(img,origin='upper')
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.gray()
    plt.show()
    
def displayScatter(x,y):
    plt.figure(figsize=(12,8))
    plt.scatter(x,y)
    plt.axis('equal')
    plt.show()

def TopViewTransform(frame):
    #チルト角
    tiltdeg = 25
    height = 60
    f = 320/2 * np.tan(1.04)
    w, h = 320, 240

    tilt = np.deg2rad(tiltdeg)      
    #3D変換
    A = np.matrix([
        [1,0,-w/2],
        [0,0,0],
        [0,1,-h/2],
        [0,0,1]])
    #回転
    R= np.matrix([
        [1,0,0,0],
        [0, np.cos(tilt), -np.sin(tilt), 0],
        [0, np.sin(tilt), np.cos(tilt), 0],
        [0,0,0,1]])
    #並進
    T =np.matrix([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,-height/np.sin(tilt)],
        [0,0,0,1]])
    #カメラパラメータ
    K =np.matrix([
        [f, 0, w/2, 0],
        [0, f, h/2,0],
        [0, 0, 1, 0],])
    #変換行列
    H = K * (T * (R * A))
    #パースペクティブ変換
    warp = cv2.warpPerspective(frame, H, (320, 240),flags=cv2.WARP_INVERSE_MAP)
    #トリミング範囲設定
    areaCropBase = np.array([[0, 240], [320, 240]], dtype='float32')
    #パースペクティブ変換後の座標取得
    areaCrop = cv2.perspectiveTransform(np.array([areaCropBase]), H.I)
    areaCrop = np.matrix([[areaCrop[0,1,0],0],[areaCrop[0,1,0],areaCrop[0,1,1]],[areaCrop[0,0,0],areaCrop[0,0,1]],[areaCrop[0,0,0],0]])
    #トリミング
    warp = warp[np.uint(areaCrop[0,1]):np.uint(areaCrop[2,1]),np.uint(areaCrop[0,0]):np.uint(areaCrop[2,0])]
    #warp = np.fliplr(np.flipud(warp))
    #displayImageLwr(warp)    
    return warp


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


#TopViewTransform(frame)
#displayImage(frame)
#frame = ReadVideoFrameBinary(frame)

#laneParamL = []
#laneParamR = []
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

"""
if cap.isOpened():
    rval , frame = cap.read()
#トップビュー変換
frame = TopViewTransform(frame)
#2値化
#frame = ReadVideoFrameBinary(frame)

#フィッティング
##

#表示
displayImageUpr(frame)

"""

while(True):
    if cap.isOpened():
        rval , frame = cap.read()
    frame = TopViewTransform(frame)
    frame = ReadVideoFrameBinary(frame)
    cv2.namedWindow('window')
    cv2.imshow('window', np.array(frame * 255, dtype=np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
