import numpy as np
import cv2 as cv

capture = cv.VideoCapture(r'D:\Aaa\task5/project_video.mp4')

while True:
    
    isTrue , frame = capture.read()
    resized_frame = cv.resize(frame, (300, 200)) 
    
    src = np.float32(
        [[120, 135], #TL
        [183, 135],  #TR
        [263, 182],  #BR
        [60, 187]]   #BL
                    )
        
    dst = np.float32(
        [[0, 0],         
        [300, 0],       
        [300, 200],     
        [0, 200]]      
                    )
    
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(resized_frame, M, (300, 200)) 
    
    hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1]
    S = hsv[:,:,1]
    hsv[:,:,2]*3
    V = hsv[:,:,2]
    cv.imshow("s_hsv",S)
    cv.imshow("v_hsv",V)
    
    bitwise_or_SV = cv.bitwise_or(S,V)
    cv.imshow("Bitwise_or_SV", bitwise_or_SV)
    adaptive_thresh_v= cv.adaptiveThreshold(V, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, -3)
    cv.imshow("adaptive_thresh_v",adaptive_thresh_v)

    hls = cv.cvtColor(warped, cv.COLOR_BGR2HLS)
    h =hls[:,:,0]
    hls[:,:,2]/3
    s = hls[:,:,2]
    hls[:,:,1]/4
    L =hls[:,:,1]
    
    adaptive_thresh_s= cv.adaptiveThreshold(s, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -3)
    cv.imshow("adaptive_thresh_s",adaptive_thresh_s)
    cv.imshow("L_hls",L)
    cv.imshow("s_hls",s)
    
    bitwise_or_sV = cv.bitwise_or(s,V)
    cv.imshow("Bitwise_or_sV", bitwise_or_sV)
    
    blurred = cv.GaussianBlur(bitwise_or_sV,(21,21),0)
    #cv.imshow("Blurred", blurred)
    
    adaptive_thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, -3)
    cv.imshow("Adaptive threshold", adaptive_thresh)
    
    #cv.imshow('Warped frames', warped)
    #cv.imshow("Final", resized_frame)

    if cv.waitKey(20) & 0XFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()