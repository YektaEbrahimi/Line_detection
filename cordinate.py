import cv2
import matplotlib.pyplot as plt

img = cv2.imread("D:/Aaa/Final.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(img_rgb, (300, 200))
plt.imshow(resized_img)
plt.show()