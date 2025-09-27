import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv.imread(r'D:\Aaa\task5\Part 01-Module 04-Lesson 03_Advanced Techniques for Lane Finding\img\color-shadow-example.jpg')
#img = mpimg.imread('D:\\Aaa\\task5\Part 01-Module 04-Lesson 03_Advanced Techniques for Lane Finding\\img/color-shadow-example.jpg')
resized = cv.resize(img, (500,300))
    
cv.circle(resized ,(178,206),2,(255,255,255),3) 
cv.circle(resized ,(316,200),2,(255,255,255),3) 
cv.circle(resized ,(488,293),2,(255,255,255),3) 
cv.circle(resized ,(30,295),2,(255,255,255),3) 
   
src = np.float32(
    [[178,206], #TL
     [316,200], #TR
     [488,293], #BR
     [30,295]]  #BL
                )
    
dst = np.float32(
    [[0, 0],         
    [500, 0],       
    [500, 300],     
    [0, 300]]      
                )
    
M = cv.getPerspectiveTransform(src, dst)
warped = cv.warpPerspective(img, M, (500, 300)) 

cv.imshow('Original image', resized)
cv.imshow('Warped image', warped)

hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
S = hsv[:,:,1]
V = hsv[:,:,2]
cv.imshow("s_hsv",S)
cv.imshow("v_hsv",V)

hls = cv.cvtColor(warped, cv.COLOR_BGR2HLS)
H =hls[:,:,1]
s = hls[:,:,2]
cv.imshow("h_hls",H)
cv.imshow("s_hls",s)

cv.waitKey(0)
#cv.destroyAllWindows()
