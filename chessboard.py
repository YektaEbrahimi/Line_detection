import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('D:\\Aaa\\task5\\Part 01-Module 04-Lesson 01_Camera Calibration\\img/orig-and-undist - Copy.png')
img = (img * 255).astype(np.uint8)

objpoints=[]
imgpoints=[]

objp=np.zeros((6*8,3),np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret , corners =cv.findChessboardCorners(gray, (8,6),None)

if ret==True:
    imgpoints.append(corners)
    objpoints.append(objp)
    img=cv.drawChessboardCorners(img,(8,6),corners,ret)

    img_size= (img.shape[1] , img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,imgpoints,gray.shape[: :-1],None ,None)
    dst=cv.undistort(img ,mtx ,dist ,None ,mtx)    
    plt.imshow(dst)
    plt.show()
       
