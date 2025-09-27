import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img= mpimg.imread('D:\\Aaa\\task5\\Part 01-Module 04-Lesson 02_Gradients and Color Spaces\\img/curved-lane.jpg')

hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
H = hsv[:,:,0]
S = hsv[:,:,1]
V = hsv[:,:,2]

hls=cv.cvtColor(img, cv.COLOR_BGR2HLS)
h = hls[:,:,0]
l = hls[:,:,1]
s = hls[:,:,2]

f, axs=plt.subplots(2,3,figsize=(18,10))

axs[0,0].imshow(H, cmap='gray')
axs[0,0].set_title("H_hsv")

axs[0,1].imshow(S, cmap='gray')
axs[0,1].set_title("S_hsv")

axs[0,2].imshow(V, cmap='gray')
axs[0,2].set_title("v_hsv")

axs[1,0].imshow(h, cmap='hsv')
axs[1,0].set_title("H_hls")

axs[1,1].imshow(l, cmap='gray')
axs[1,1].set_title("L_hls")

axs[1,2].imshow(s, cmap='gray')
axs[1,2].set_title("S_hls")

plt.tight_layout()

plt.show()


