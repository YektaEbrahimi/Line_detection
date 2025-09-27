import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("D:\\Aaa\\task5\Part 01-Module 04-Lesson 01_Camera Calibration\\img/Screenshot 2025-09-21 163309.png")

plt.imshow(img)

plt.plot( 418, 156,'.')
plt.plot( 430, 219,'.')
plt.plot( 262, 105,'.')
plt.plot( 262, 174,'.')

def warp(img):
                #width        #hight
    img_size = (img.shape[1] , img.shape[0])
    
    src = np.float32(
        [[418, 156], #TL
         [430, 219], #TR
         [262, 174], #BR
         [262, 105]] #BL
    )
    
    dst = np.float32(
        [[420, 95],
         [420, 160],
         [260, 160],
         [260, 95]]      
    )
    
    m=cv.getPerspectiveTransform(src, dst)
    #minv=cv.getPerspectiveTransform(dst,src)
    warped=cv.warpPerspective(img,m,img_size, flags=cv.INTER_LINEAR)
    return warped 

warped_im = warp(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(img)
ax2.set_title('Warped image')
ax2.imshow(warped_im)

plt.show()
