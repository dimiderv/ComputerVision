import cv2
import numpy as np

filename = 'c:/image_bw.png'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('main', )
cv2.imshow('main', img)
cv2.waitKey(0)
print(img.shape[0])
print(img.shape[1])
#np.ones((5,5))
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
erode= cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
cv2.imshow('erode',erode)
cv2.waitKey(0)
kernel_h=kernel.shape[0]
kernel_w=kernel.shape[1]
h=kernel_h//2
w=kernel_h//2

dilation=np.zeros((img.shape[0],img.shape[1]))
#dilation
for i in range(h,img.shape[0]-h):
    for j in range(w, img.shape[1] - w):
        t=0
        for m in range(kernel_h):
            for n in range(kernel_w):

                    t=kernel[m][n]*img[i-h+m][j-w+n]+t
        dilation[i][j] = t

erosion = np.zeros((img.shape[0],img.shape[1]))



#erosion
for i in range(2,img.shape[0]-5):
    for j in range(2,img.shape[1]-5):
        for m in range(5):
            for n in range(5):
                if img[i-m][j]==0 or img[i][j-n]==0 or img[i-m][j-n]==0 or img[i+m][j]==0 or img[i][j+n]==0:
                    erosion[i][j]=0
                else:
                    erosion[i][j]=img[i][j]
#closing
closing=np.zeros((erosion.shape[0],img.shape[1]))
for i in range(2,dilation.shape[0]-5):
    for j in range(2,dilation.shape[1]-5):
        for m in range(5):
            for n in range(5):
                if dilation[i-m][j]==0 or dilation[i][j-n]==0 or dilation[i-m][j-n]==0 or dilation[i+m][j]==0 or dilation[i][j+n]==0:
                    closing[i][j]=0
                else:
                    closing[i][j]=dilation[i][j]
#opening
opening=np.zeros(closing.shape)
for i in range(h,erosion.shape[0]-h):
    for j in range(w, erosion.shape[1] - w):
        t=0
        for m in range(kernel_h):
            for n in range(kernel_w):

                    t=kernel[m][n]*erosion[i-h+m][j-w+n]+t
        opening[i][j] = t

cv2.imshow('dilation',dilation)
cv2.waitKey(0)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.imshow('closing',closing)
cv2.waitKey(0)
cv2.imshow('opening',opening)
cv2.waitKey(0)

opening1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing1= cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
op=opening1-opening
cl=closing1-closing
cv2.imshow('op',op)
cv2.waitKey(0)
cv2.imshow('cl',cl)
cv2.waitKey(0)
