import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math



def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
  
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=7):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def displayImage(img):
    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.gray()
    plt.show()
 
def ReadVideoFrameBinary():
    low_threshold = 100
    high_threshold = 200
    cap = cv2.VideoCapture('sample.mp4')
    if cap.isOpened():
        rval , frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.Canny(frame, low_threshold, high_threshold)
    frame = cv2.convertScaleAbs(cv2.Sobel(frame, cv2.CV_32F, 1,0,ksize=3))
    frame = cv2.convertScaleAbs(frame)
    displayImage(frame)
    return frame

def ConnectXline(img_SobelX):
    imgThreshold = img_SobelX * (img_SobelX > 100))
    


ReadVideoFrameBinary()
