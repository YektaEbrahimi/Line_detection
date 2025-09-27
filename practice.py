import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


capture = cv.VideoCapture(r'Desktop/Final.png')

while True:
    isTrue , frame = capture.read()
    new_width , new_height = 600,400
    resized_frame = cv.resize(frame, (new_width , new_height))
    
    cv.imshow('Resized Video', resized_frame)
    if cv.waitKey(20) & 0XFF==ord('d'):
        break
    
capture.release()

'''plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='blue')
    plt.xlim(0, binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0], 0)
    plt.title("Lane Detection Result")
    plt.show()'''
    
    '''plt.figure(figsize=(10, 4))
    plt.plot(histogram, color='gray')
    plt.title("Histogram of Bottom Half of Binary Image")
    plt.xlabel("Column Index (X)")
    plt.ylabel("Sum of Pixel Intensities")
    plt.grid(True)
    plt.show()'''
    
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#img = mpimg.imread('D:\\Aaa\\task5\Part 01-Module 04-Lesson 03_Advanced Techniques for Lane Finding\\img/color-shadow-example.jpg')

'''img = cv.imread(r'D:\Aaa\task5\Part 01-Module 04-Lesson 03_Advanced Techniques for Lane Finding\img\color-shadow-example.jpg')
resized = cv.resize(img, (500, 300))

resized_rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
plt.plot( 209,178,'.')
plt.plot( 314,178,'.')
plt.plot( 462,295,'.')
plt.plot( 73,295,'.')

plt.imshow(resized_rgb)
plt.title("Resized Image")
plt.axis('off')  
plt.show()'''

#resized = cv.resize(img, (500,300))

#cv.circle(resized ,(187,200),2,(255,255,255),3) 
#cv.circle(resized ,(316,200),2,(255,255,255),3) 
#cv.circle(resized ,(488,293),2,(255,255,255),3) 
#cv.circle(resized ,(50,295),2,(255,255,255),3) 