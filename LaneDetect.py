import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import interpolate
import cv2
import math

def zscore(x):
    xmean = x.mean()
    xstd  = np.std(x)
    zscore = (x-xmean)/xstd
    return zscore

def displayImageLwr(img):
    #plt.figure(figsize=(12,8))
    #plt.figure()
    plt.imshow(img,origin='lower')
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.gray()
    #plt.show()

def displayImageUpr(img):
    #plt.figure(figsize=(12,8))
    #plt.figure()
    plt.imshow(img,origin='upper')
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.gray()
    #plt.show()
    
def displayScatter(x,y):
    #plt.figure(figsize=(12,8))
    plt.scatter(x,y)
    plt.axis('equal')
    #plt.show()


def Plothist2d(x,y):
    #fig = plt.figure()
    ax = fig.add_subplot(111)
    H = ax.hist2d(x,y, bins=40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(H[3],ax=ax)
    #plt.show()


def polyfitt2d(point2D,point2Dp):
    #plt.figure()
    fit = np.polyfit(point2Dp[1],point2Dp[0],2)    
    x,y = np.power(np.e,fit[0]*np.power(point2Dp[1],2)+fit[1]*point2Dp[1]+fit[2]) * np.cos(point2Dp[1]),np.power(np.e,fit[0]*np.power(point2Dp[1],2)+fit[1]*point2Dp[1]+fit[2]) * np.sin(point2Dp[1])
    plt.scatter(point2D[1],point2D[0])
    plt.scatter(x,y)
    #plt.show()

    
def circfit(data,weights):
    #transform data according to Coopes method
    A=np.concatenate((data,np.matrix([np.ones(data.shape[0])]).T),axis=1)
    y=np.matrix(np.power(data,2).sum(axis=1)).T
    p = (A.T * np.diag(weights) * A).I * (A.T * np.diag(weights) * y)
    center = 0.5 * p[:2,:]
    radius = np.sqrt(p[2] + center.T * center)
    return [center[0,0],center[1,0]] , radius[0,0]


def plot_data_circle(x , y , xc, yc, R):
    #plt.figure()
    theta_fit = np.linspace(-np.pi,0, 180)
    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit,y_fit)
    plt.plot(xc,yc,marker='x')
    plt.scatter(x,y)
    plt.ylim(4,6)
    plt.xlim([0,np.pi])
    plt.axis('equal')
    #plt.show()
    
    
def ReadVideoFrameBinary(frame):
    #床の色に合わせて明度・彩度マスクを変更すること
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frameH =frameHSV[:,:,0]
    frameS = frameHSV[:,:,1]
    frameV = frameHSV[:,:,2]
    frameSobScaleX = cv2.Sobel(frameV,cv2.CV_32F,1,0,ksize=5)
    frameSobScaleY = cv2.Sobel(frameV,cv2.CV_32F,0,1,ksize=5)
    
    frameSStd = zscore(frameS)#彩度標準化
    frameVStd = zscore(frameV)#明度標準化
    
    #輪郭抽出
    frameSobScaleX = zscore(frameSobScaleX)
    frameSobScaleY = zscore(frameSobScaleY)
    #輪郭情報からマスク画像生成
    kernDilateEdge = np.ones((5,20),np.uint8)
    maskEdge = cv2.dilate(np.uint8(frameSobScaleX > 3),kernDilateEdge,iterations = 1) *  cv2.dilate(np.uint8(frameSobScaleX > 3),kernDilateEdge,iterations = 1) #膨張
    plt.imshow(frameSStd)
    #彩度・明度情報からマスク画像生成(アスファルト・白線の場合)
    thresS = 0.4 #彩度しきい値アスファルト用・白線用(パスフィルタ)
    thresV = 1 #明度しきい値アスファルト用(パスフィルタ)
    kernDilateSV = np.ones((5,20),np.uint8)#膨張用カーネル
    maskS = cv2.erode(np.uint8(frameSStd < thresS),kernDilateSV,iterations = 1) > 0#彩度マスク
    #maskV = cv2.dilate(np.uint8(frameVStd < thresV),kernDilateSV,iterations = 1) > 0#明度マスク
    maskPassLine = (np.uint8(frameSStd > 2) * np.uint8(frameVStd > thresV) * np.uint8(frameH < 40) * np.uint8(frameH > 10))
    #明度2値化画像をマスクして出力
    ret , frameBin = cv2.threshold(frameV,0,1,cv2.THRESH_OTSU)
    plt.imshow(frameBin*maskEdge*maskS + maskPassLine)
    plt.show()
    #frame = cv2.convertScaleAbs(cv2.Sobel(frame, cv2.CV_32F, 1,0,ksize=3))#x方向のみエッジ検出(通常はこっち)
    #frame = cv2.convertScaleAbs(cv2.Sobel(frame, cv2.CV_32F, 0,1,ksize=3))#y方向のエッジ検出
    return frameBin*maskEdge*maskS + maskPassLine

def TopViewTransformPoints(frameBinary):
    matPoint = np.matrix([frameBinary.nonzero()[1],frameBinary.nonzero()[0]])
    matPoint = np.vstack((matPoint,np.ones([1,matPoint.shape[1]])))
    
    #チルト角
    tiltdeg = 18
    height = 100 #拡大率に相当(+:拡大)
    f = 500 / 2 * np.tan(1.04) #視野角に相当(+:望遠)

    w, h = frameBinary.shape[1] ,  frameBinary.shape[0]
    tilt = np.deg2rad(tiltdeg)

    #3D変換    
    A = np.matrix([
        [1,0,0,0],
        [0,0,0,0],
        [0,0,1,0],
        [0,0,0,1]])
    
    #回転
    R= np.matrix([
        [1,0,0,0],
        [0,np.cos(tilt),-np.sin(tilt),0],
        [0,np.sin(tilt), np.cos(tilt),0],
        [0,0,0,1]])
    
    #並進
    T =np.matrix([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,-height/np.sin(tilt)],
        [0,0,0,0]])
    
    #カメラパラメータ
    K =np.matrix([
        [f, 0,w/2,0],
        [0,f,h/2,0],
        [0, 0, 1,0],
        ])
    
    #変換行列
    H = K * (T * (R* A))
    H = H[:,np.array([0,2,3])]
    
    #点列を変換、勾配情報を付加
    matPointTrans = cv2.perspectiveTransform(np.array([matPoint.T[:,0:2]]), H.I)[0].T
    pointTrans = np.squeeze(np.asarray(matPointTrans))#MatrixからArrayへ変換
    pointTrans[1,:] = -pointTrans[1,:]#距離情報反転する
    pointTrans = pointTrans[:, pointTrans[0,:].argsort()]#x方向の値でソート
    #IPM変換済みの点列情報をリターンする
    return pointTrans

def CreateTrajectory(pointTrans):
    pointTrans = np.vstack((pointTrans,np.gradient(pointTrans[0,:])))#x方向の勾配  
    pointTransList = np.split(pointTrans,np.where(pointTrans[2,:] > 5)[0],axis=1)#x方向の勾配を利用してレーンを分割
    pointTransList = [elem for elem in pointTransList if elem.shape[1] > 100]#要素数がしきい値以下の場合は外れ値と見なし、削除する
    #idea:主成分分析して固有ベクトルの比で外れ値をさらに減らす??
    if pointTransList : #外れ値を削除して残ったレーン情報に関して処理継続する。
    #各車線情報のプロット
        plt.clf()
        for elem in pointTransList:
            plt.scatter(-elem[0,:],elem[1,:])
        plt.axis('equal')
        plt.grid()
        plt.show()
    #各車線情報のプロットおわり
    #最近傍車線の横位置算出 : ステアリング制御用
        latPosNearLine = np.array([elem[0,elem[1,:].argmin()] for elem in pointTransList])
        latPosNearLine = [latPosNearLine[(latPosNearLine < 0)].max(),latPosNearLine[(latPosNearLine > 0)].min()]
        pointTransList = [np.array([elem[0,:]-elem[0,elem[1,:].argmin()],elem[1,:],elem[2,:]]) for elem in pointTransList]#各レーンの初期x座標を0とする
        pointTrans = np.concatenate(([elem for elem in pointTransList]), axis=1)#分割した要素を再度結合して戻す
        pointTransunique = pd.DataFrame(pointTrans[0:2,:].T,columns=['x', 'y',])#pandasで距離方向の重複を削除しつつ、横位置の平均を出す
        pointTransunique = pointTransunique.groupby('y').mean().reset_index()#↑
        pointTrans = np.fliplr(pointTransunique.values).T#pandasからnumpy配列へ
        pointTrans[1,:] = pointTrans[1,:] - pointTrans[1,0]#結合したので、初期のy座標を0にする
    #Bスプライン補完による曲率算出 : 速度制御用
        idx = np.array(np.interp(np.arange(0,pointTrans[1,-1],pointTrans[1,-1]*0.1),pointTrans[1,:],np.arange(pointTrans.shape[1])),dtype=np.int64)#等間隔距離インデックス取得
        ctrlPoint = pointTrans[:,idx]
        t,c,k = interpolate.splrep(ctrlPoint[1,:],ctrlPoint[0,:],s=0,k=3)
        Spl=interpolate.BSpline(t,c,k)#Bスプライン関数
        Spld1 = Spl.derivative(1)#Bスプライン1階微分
        Spld2 = Spl.derivative(2)#Bスプライン2階微分
        iptY = np.arange(0,ctrlPoint[1,-1],ctrlPoint[1,-1]*0.02)#Bスプライン補間用距離
        cvmax = max(abs(Spld2(iptY)/pow((1+pow(Spld1(iptY),2)),1.5)))#Bスプライン補間による最大曲率
    #最近傍車線検知
    #Bスプライン補完のプロット
        plt.figure()
        plt.scatter(ctrlPoint[0,:],ctrlPoint[1,:])
        plt.plot(Spl(iptY),iptY)
        #plt.axis('equal' )
        plt.xlim([-20,20])
        plt.grid()
        plt.show()
        plt.figure()
        plt.plot(Spld1(iptY),iptY)
        plt.plot(Spld2(iptY),iptY)
        plt.grid()
        plt.show()
    #速度制御用の曲率、ステア制御用に、自車近傍の車線情報・自車近傍の車線の傾きを出力する
    return cvmax , latPosNearLine , Spld1(iptY[0])
    

def ejphi(phi0,phiv,phiu,s):
    return np.exp((phiu*s*s+phiv*s+phi0)*1j)
def ClothoidSimpon(h,s,phi0,phiv,phiu):
    s=s/h
    return h*((s-0)/6*(ejphi(phi0,phiv,phiu,0) + 4*ejphi(phi0,phiv,phiu,s/2) + ejphi(phi0,phiv,phiu,s)))

def FitFuncClothoid(s,y0,phi0,phiv,phiu):
    return np.imag(ClothoidSimpon(1,s,phi0,phiv,phiu)) + y0

def FitFuncClothoidPhiv(s,y0,phi0,phiv):
    return np.imag(ClothoidSimpon(1,s,phi0,phiv,0)) + y0

def FitFuncClothoidPhiu(s,y0,phi0,phiu):
    return np.imag(ClothoidSimpon(1,s,phi0,0,phiu)) + y0

def PolyFitClothoid(frame):
    param_init =[0,0,0,0]         # フィッティングパラメータ　初期条件
    param_opt =[0,0,0,0]         # フィッティングパラメータ　フィッティング後
    param_bounds=([-1,-np.pi/2,-np.pi/20],[1,np.pi/2,np.pi/20])
    lanePixelsX , lanePixelsY = np.where(frame)
    lanePixelsX = lanePixelsX/frame.shape[0] #規格化
    lanePixelsY = lanePixelsY/frame.shape[0] #規格化
    if frame[frame > 128].size > 500: #frameの範囲内に点が100点以上存在する場合
        try:
            param_opt[0:3], cov = curve_fit(FitFuncClothoidPhiv, lanePixelsX, lanePixelsY, param_init[0:3],bounds=param_bounds)
        except RuntimeError:
            print("Error - curve_fit failed")
    #    if len(lanePixelsX[0][lanePixelsX[0] == 0]) > 3: #手前に初期値となる点がない場合
    #            param_opt[0:3], cov = curve_fit(FitFuncClothoidPhiv, lanePixelsX, lanePixelsY, param_init[0:3]) #フィッティング曲線のうねり防止のためクロソイド縮率は0とする
                
        #For debug plot
                #plt.scatter(lanePixelsX,lanePixelsY)
        #plt.scatter(lanePixelsX,lanePixelsY)
        #plt.scatter(np.arange(0,frameSplit.shape[0],1),FitFuncClothoid(np.arange(0,frameSplit.shape[0],1),param_opt[0],param_opt[1],param_opt[2],param_opt[3]),color='RED')
        plt.figure()
        plt.scatter(lanePixelsY,lanePixelsX)
        plt.scatter(FitFuncClothoid(np.arange(0,1,0.01),param_opt[0],param_opt[1],param_opt[2],param_opt[3]),np.arange(0,1,0.01),color='RED')
        #plt.xlim(0,255)
        #print([param_opt[0],param_opt[1] ,param_opt[2],param_opt[3]])
        print(param_opt[2])
        plt.xlim([-0.1,0.5])
    return param_opt[2]
        #plt.axis('equal')
    #param_opt[2:4] = [-param_opt[2], -param_opt[3]]
    #return param_opt #Clothoid param(fitted)
    
def LanePosDetectfromEachSector(sectorFrameBinary):
    plt.figure()
    displayImageUpr(sectorFrameBinary)
    plt.show()
    sectorFrameBinaryUint8 = np.uint8(sectorFrameBinary)
    #hough変換画像でレーン以外をマスクする
    mask = np.zeros([sectorFrameBinary.shape[0],sectorFrameBinary.shape[1]],dtype=np.uint8)
    #破線部の補完もする
    lineinterp = np.zeros([sectorFrameBinary.shape[0],sectorFrameBinary.shape[1]],dtype=np.uint8)
    lines = cv2.HoughLinesP(np.uint8(sectorFrameBinaryUint8), 1, np.pi / 180, 50,  np.array([]), minLineLength=20, maxLineGap=100)[:,0,:]
    lines = np.c_[lines,(lines[:,2]-lines[:,0])/(lines[:,3]-lines[:,1])]
    lines = np.c_[lines,lines[:,2] - (lines[:,4] * lines[:,3])]
    linesInfo = np.array([(lines[lines[:,4] < 0,4].mean() , lines[lines[:,4] < 0,5].mean()),(lines[lines[:,4] > 0,4].mean() , lines[lines[:,4] > 0,5].mean())])
    # Draw the lines
    if linesInfo is not None and ~np.isnan(linesInfo).any() and ~np.isinf(linesInfo).any():
        for i in range(0, len(linesInfo)):
            #cv2.line(mask,(np.int(linesInfo[i][1]),0), (np.int(linesInfo[i][0]*sectorFrameBinary.shape[1] + linesInfo[i][1]),sectorFrameBinary.shape[1]), (255,255,255), 100)
            cv2.line(lineinterp,(np.int(linesInfo[i][1]),0), (np.int(linesInfo[i][0]*sectorFrameBinary.shape[1] + linesInfo[i][1]),sectorFrameBinary.shape[1]), (255,255,255), 2)
    #エラー時の処理
    else:
            return 0
    
    #直線補完のデバッグ用
    plt.imshow(lineinterp) 
    plt.show()
    #マスク処理
    sectorFrameBinary = np.logical_and(sectorFrameBinary,mask==255)
    #直線補完処理(破線部をなくす)
    sectorFrameBinary = np.logical_xor(sectorFrameBinary,lineinterp==255)
    #y方向にフレーム5分割
    splitFrame = np.array_split(sectorFrameBinary,5,axis=0)
    #分割フレームからレーン検出
    for i in range(0, len(splitFrame)):
        sectorFrameBinary = splitFrame[i]
        posLanedata = np.sum(sectorFrameBinary, axis=0)
        posLanedata = np.where(posLanedata > (np.mean(posLanedata) + 2 * np.std(posLanedata)))[0]
        #y=0時のx軸との切片の平均値で左右分割する
        posLanedataMean = linesInfo[:,1].mean()
        posLane = np.array([posLanedata[posLanedata < posLanedataMean].mean(),posLanedata[posLanedata > posLanedataMean].mean()])
        print(i,posLane)
    #点列情報を格納し、Bスプライン補完してみる
    return 1

#TopViewTransform(frame)
#displayImage(frame)
#frame = ReadVideoFrameBinary(frame)

#laneParamL = []
#laneParamR = []
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('solidYellowLeft.mp4')
#cap = cv2.VideoCapture('solidWhiteRight.mp4')
#cap = cv2.VideoCapture('sample.mp4')
while(True):
    if cap.isOpened():
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
        rval , frame = cap.read()
    
        #トップビュー変換
        #frame = TopViewTransform(frame)
        #LogPolar変換
        #frame = LogpolarTrans(frame)
       #yピクセルセクタ毎にレーン位置の抽出
        plt.figure()
        displayImageUpr(frame)
        plt.show()
       #2値化
        frame = ReadVideoFrameBinary(frame[350:,:,:])
        #IPM変換、車線点列情報取得
        pointTrans = TopViewTransformPoints(frame)
        #車線点列情報を用いた走行軌跡生成(b-spline法)
        CreateTrajectory(pointTrans)
        #LanePosDetectfromEachSector(frame)
#displayImageUpr(frame)
#フィッティング
#frameR,frameL=np.array_split(frame,2,axis=1)
#PolyFitClothoid(frameL)

#表示
#displayImageUpr(frame)

"""
curve = np.array([])
fig , ax = plt.subplots(2,2)
while(True):
    rval = 0
    if cap.isOpened():
        rval , frame = cap.read()
    if rval and frame.size:
        for i in range(0,ax.shape[0]):
            for j in range(0,ax.shape[1]):
                ax[i,j].clear()  
        #入力画像
        ax[0,0].imshow(frame)
        #IPM変換
        #frame = TopViewTransform(frame)
        #2値化画像
        plt.figure()
        displayImageUpr(frame)
        plt.show()
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frameHSV = frameHSV[340:340+200,:]
        MaskingVehicle(frameHSV)
        frameHSV = ReadVideoFrameBinary(frameHSV)#2値化
        LanePosDetectfromEachSector(frameHSV)
        plt.pause(0.1)
        #fig.clf()
        #cv2.namedWindow('window')
        #cv2.imshow('window', cv2.flip(frame,0))
        #cv2.imshow('window', cv2.flip(np.array(frame * 255, dtype=np.uint8),0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
"""
cap.release()
cv2.destroyAllWindows()






"""Log-Polar変換はカーブとの相性がよろしくないのでボツ
def LogpolarTrans(frame):
    vanHeiPx = 0#消失点のY座標(pix)
    frame =  frame[vanHeiPx:,:]
    frame = np.flipud(frame)
    #frame = np.fliplr(frame)
    #frame = ReadVideoFrameBinary(frame)
    #displayImageLwr(frame)
    point2D = np.array(np.where(frame))
    point2D[1] = point2D[1] - frame.shape[1] * 0.5
    #point2Dp = np.array([np.linalg.norm(point2D,axis=0),np.arctan2(point2D[0,:],point2D[1,:])])
    point2Dp = np.array([np.arctan2(point2D[0,:],point2D[1,:]),np.log(np.linalg.norm(point2D,axis=0))])
    #center , radius = circfit(point2Dp.T,np.ones(point2Dp.shape[1]))
    #plot_data_circle(point2Dp[0],point2Dp[1],center[0],center[1],radius)
    #polyfitt2d(point2D,point2Dp)
    #Plothist2d(point2Dp[0],point2Dp[1])
    return point2Dp
"""
