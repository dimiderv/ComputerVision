import cv2
import numpy as np

filename= 'c:/image.jpg'
img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

cv2.imshow('Initial image',img)
cv2.waitKey(0)
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img2=cv2.filter2D(img,cv2.CV_8UC1,kernelx)
img3=cv2.filter2D(img,cv2.CV_8UC1,kernely)
filter2d=img2+img3
cv2.imshow('Filter 2D',filter2d)
cv2.waitKey(0)

kernel_y=kernely.shape[0]

kernel_x=kernelx.shape[1]
x_axis=kernel_x//2
y_axis=kernel_y//2


Prewitt=np.zeros(img.shape)

for m in range(x_axis,img.shape[0]-x_axis):
    for n in range(y_axis,img.shape[1]-y_axis):
        y_sum = 0
        x_sum = 0
        for i in range(0,kernel_y):
            for j in range(0,kernel_x):
                x_sum=kernelx[i][j]*img[m+i-x_axis][n+j-y_axis]+x_sum
                y_sum=kernely[i][j]*img[m+i-x_axis][n+j-y_axis]+y_sum
        Prewitt [m][n]= x_sum + y_sum

cv2.imshow('initial Prewitt',Prewitt)
cv2.waitKey(0)

final_Prewitt=1/255*Prewitt
cv2.imshow('Constructed Prewitt',final_Prewitt)
cv2.waitKey(0)
Differences=final_Prewitt-filter2d
cv2.imshow('result',Differences)
cv2.waitKey(0)





