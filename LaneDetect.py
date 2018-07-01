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
    tiltdeg = 30
    height = 100#拡大率に相当(+:拡大)
    f = 500/2 * np.tan(1.04) #視野角に相当(+:望遠)
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
    #トリミング
#    warp = warp[np.uint(areaCrop[0,0,1]+1):np.uint(areaCrop[0,1,1]),np.uint(areaCrop[0,1,0]+1):np.uint(areaCrop[0,0,0])]
    warp = np.flipud(warp)
    displayImageLwr(warp)    
    return warp




"""
def TopViewTransform(frame):
    #チルト角
    tiltdeg = 40
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
    warp = np.flipud(warp)
    displayImageLwr(warp)    
    return warp
"""

def ejphi(phi0,phiv,phiu,s):
    return np.exp((phiu*s*s+phiv*s+phi0)*1j)
def ClothoidSimpon(h,s,phi0,phiv,phiu):
    s=s/h
    return h*((s-0)/6*(ejphi(phi0,phiv,phiu,0) + 4*ejphi(phi0,phiv,phiu,s/2) + ejphi(phi0,phiv,phiu,s)))

def FitFuncClothoid(s,y0,phi0,phiv,phiu):
    return np.imag(ClothoidSimpon(1,s,phi0,phiv,phiu)) + y0

def FitFuncClothoidPhiv(s,y0,phi0,phiv):
    return np.imag(ClothoidSimpon(1,s,phi0,phiv,0)) + y0

def PolyFitClothoid(frameSplit):
    param_init =[0,0,0,0]         # フィッティングパラメータ　初期条件
    param_opt =[0,0,0,0]         # フィッティングパラメータ　フィッティング後
    param_bounds=([-1,-np.pi/2,-np.pi,-np.pi],[1,np.pi/2,np.pi,np.pi])
    lanePixelsX , lanePixelsY = np.where(frameSplit)
    lanePixelsX = lanePixelsX/frameSplit.shape[0] #規格化
    lanePixelsY = lanePixelsY/frameSplit.shape[0] #規格化
    if frameSplit[frameSplit == True].size > 100: #frameの範囲内に点が100点以上存在する場合
        try:
            param_opt, cov = curve_fit(FitFuncClothoid, lanePixelsX, lanePixelsY, param_init,bounds=param_bounds)
        except RuntimeError:
            print("Error - curve_fit failed")
    #    if len(lanePixelsX[0][lanePixelsX[0] == 0]) > 3: #手前に初期値となる点がない場合
    #            param_opt[0:3], cov = curve_fit(FitFuncClothoidPhiv, lanePixelsX, lanePixelsY, param_init[0:3]) #フィッティング曲線のうねり防止のためクロソイド縮率は0とする
                
        #For debug plot
                #plt.scatter(lanePixelsX,lanePixelsY)
        #plt.scatter(lanePixelsX,lanePixelsY)
        #plt.scatter(np.arange(0,frameSplit.shape[0],1),FitFuncClothoid(np.arange(0,frameSplit.shape[0],1),param_opt[0],param_opt[1],param_opt[2],param_opt[3]),color='RED')
        #plt.figure()
        plt.scatter(lanePixelsY,lanePixelsX)
        plt.scatter(FitFuncClothoid(np.arange(0,1,0.01),param_opt[0],param_opt[1],param_opt[2],param_opt[3]),np.arange(0,1,0.01),color='RED')
        #plt.xlim(0,255)
        print([param_opt[0],param_opt[1] ,param_opt[2],param_opt[3]])
        plt.axis('equal')
    #param_opt[2:4] = [-param_opt[2], -param_opt[3]]
    #return param_opt #Clothoid param(fitted)

#TopViewTransform(frame)
#displayImage(frame)
#frame = ReadVideoFrameBinary(frame)

#laneParamL = []
#laneParamR = []
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

if cap.isOpened():
    rval , frame = cap.read()
    #トップビュー変換
#    displayImageLwr(frame)
    frame = TopViewTransform(frame)
    #2値化
#    frame = ReadVideoFrameBinary(frame)
    #フィッティング
#    frameR,frameL=np.array_split(frame,2,axis=1)
#    PolyFitClothoid(frameR)

#表示
#displayImageUpr(frame)
#fig = plt.gcf()
#fig.show()
#fig.canvas.draw()
"""
while(True):
    if cap.isOpened():
        rval , frame = cap.read()
    if rval:
        frame = TopViewTransform(frame)
        frame = ReadVideoFrameBinary(frame)
        #frameR,frameL=np.array_split(frame,2,axis=1)
        PolyFitClothoid(frame)
        fig.canvas.draw()
        fig.clf()
        cv2.namedWindow('window')
        cv2.imshow('window', cv2.flip(frame,0))
        cv2.imshow('window', cv2.flip(np.array(frame * 255, dtype=np.uint8),0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
"""
cap.release()
cv2.destroyAllWindows()