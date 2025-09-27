import numpy as np
import cv2 as cv

capture = cv.VideoCapture(r'D:\Aaa\task5/project_video.mp4')

def find_road_lines(binary_warped):
    
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 10
    margin = 30
    minpix = 50
    window_height = int(binary_warped.shape[0] // nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=np.int32)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=np.int32)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else [0, 0, 0]
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else [0, 0, 0]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    center_fitx = (left_fitx + right_fitx) / 2
    center_points = np.array([np.transpose(np.vstack([center_fitx, ploty]))], dtype=np.int32)

    center_fit = np.polyfit(ploty, center_fitx, 2)
    a, b, c = center_fit

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    cv.polylines(out_img, center_points, isClosed=False, color=(0, 255, 255), thickness=2)
    
    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, a, b, c
   
#-----------------------------------------------------------------
   
while True:
    
    isTrue , frame = capture.read()
    resized_frame = cv.resize(frame, (300, 200))   

    src = np.float32(
        [[115, 135], #TL
        [185, 135],  #TR
        [268, 182],  #BR
        [57, 187]]   #BL
                    )
        
    dst = np.float32(
        [[0, 0],         
        [300, 0],       
        [300, 200],     
        [0, 200]]      
                    )
    
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(resized_frame, M, (300, 200)) 

    #---------------------------------------------------------------------
    
    hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
    hsv[:,:,2]
    V = hsv[:,:,2]

    hls = cv.cvtColor(warped, cv.COLOR_BGR2HLS)
    hls[:,:,2]
    s = hls[:,:,2]
    
    bitwise_or_sV = cv.bitwise_or(s,V)
    
    blurred = cv.GaussianBlur(bitwise_or_sV,(15,15), 0)
    
    adaptive_thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2)
    cv.imshow("Adaptive threshold", adaptive_thresh)
    
    white_pixel_count = cv.countNonZero(adaptive_thresh)
    print(white_pixel_count)
    
    #---------------------------------------------------------------------
    
    lane_overlay, left_fitx, right_fitx, ploty, left_fit, right_fit, a, b, c= find_road_lines(adaptive_thresh)
    center_fitx = (left_fitx + right_fitx) / 2
    
    left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.float32)
    right_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.float32)
    center_pts = np.array([np.transpose(np.vstack([center_fitx, ploty]))], dtype=np.float32)

    Minv = cv.getPerspectiveTransform(dst, src)
    left_unwarped = cv.perspectiveTransform(left_pts, Minv)
    right_unwarped = cv.perspectiveTransform(right_pts, Minv)
    center_unwarped = cv.perspectiveTransform(center_pts, Minv)
    
    '''a = (left_fit[0] + right_fit[0]) / 2
    b = (left_fit[1] + right_fit[1]) / 2
    c = (left_fit[2] + right_fit[2]) / 2
    center_equation = a* ploty**2 + b*ploty + c'''
    
    distance_sum = 0
    derivation_sum = 0
    
    for y in range(190, 201):
        #find_distance
        center_equation = a* y**2 + b*y + c  
        distance = center_equation - 150
        distance_sum += distance
        # find_degree
        derivation = 2*a*y + b 
        radian = np.arctan(derivation ) 
        degree= np.degrees(radian)   
        derivation_sum += degree
    
    mean_distance = distance_sum/10
    cv.putText(resized_frame, "Distance:"+str(mean_distance), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1 )
    mean_degree = derivation_sum/10
    cv.putText(resized_frame, "Degree:"+str(mean_degree), (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1 )    
      
    #---------------------------------------------------------------------
    
    cv.polylines(resized_frame, [np.int32(left_unwarped)], isClosed=False, color=(0, 0, 255), thickness=2)  
    cv.polylines(resized_frame, [np.int32(right_unwarped)], isClosed=False, color=(255, 0, 0), thickness=2)  
    cv.polylines(resized_frame, [np.int32(center_unwarped)], isClosed=False, color=(0, 255, 255), thickness=2) 
    cv.imshow('Warped frames', warped)
    cv.imshow("Lane Detection", lane_overlay)
    cv.imshow("Final", resized_frame)

    if cv.waitKey(20) & 0XFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()