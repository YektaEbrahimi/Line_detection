import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

binary_warped = cv.imread(r'D:\Aaa\task5\Part 01-Module 04-Lesson 03_Advanced Techniques for Lane Finding\img\warped-example.jpg')


histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

midpoint = int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

nwindows = 9

margin = 50

minpix = 50

window_height = int(binary_warped.shape[0]//nwindows)

nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx_current = leftx_base
rightx_current = rightx_base

left_lane_inds = []
right_lane_inds = []

left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[lefty, leftx] = [255, 0, 0]
out_img[righty, rightx] = [0, 0, 255]

plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, binary_warped.shape[1])
plt.ylim(binary_warped.shape[0], 0)
plt.title("Lane Detection Result")
plt.show()


cv.waitKey(0)
#cv.destroyAllWindows()