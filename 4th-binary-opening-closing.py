import cv2
import numpy as np

filename = 'c:/image_bw.png'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('main', )
cv2.imshow('main', img)
cv2.waitKey(0)
print(img.shape[0])
print(img.shape[1])

kernel = np.ones((5,5))
erode = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
cv2.imshow('erode', erode)
cv2.waitKey(0)
kernel_h = kernel.shape[0]
kernel_w = kernel.shape[1]
h=kernel_h//2
w=kernel_h//2

#binary

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] != 0:
            img[i][j] = 255

erosion = np.zeros(img.shape)
dilation = np.zeros(img.shape)


# erosion

for i in range(2, img.shape[0]-2):
    for j in range(2, img.shape[1]-2):
        t = 0
        for m in range(0, 5):
            for n in range(0, 5):
                if img[i+m-2][j+n-2] == 255:
                    t = t+1
        if t == 25 :
            erosion[i][j] = 255
        else:
            erosion [i][j] = 0

# dilation

for i in range(2,img.shape[0]-2):
    for j in range(2,img.shape[1]-2):
        t=0
        for m in range(0,5):
            for n in range(0,5):
                if img[i+m-2][j+n-2]==0:
                    t=t+1
        if t==25 :
            dilation[i][j]=0
        else:
            dilation[i][j]=255





cv2.imshow("dilation",dilation)
cv2.waitKey(0)
constructed_opening = np.zeros(dilation.shape)

for i in range(2,dilation.shape[0]-2):
    for j in range(2,dilation.shape[1]-2):
        t=0
        for m in range(0,5):
            for n in range(0,5):
                if dilation[i+m-2][j+n-2]==255:
                    t=t+1
        if t==25:
            constructed_opening[i][j]=255
        else:
            constructed_opening [i][j]=0
constructed_closing=np.zeros(erosion.shape)
for i in range(2,erosion.shape[0]-2):
    for j in range(2,erosion.shape[1]-2):
        t=0
        for m in range(0,5):
            for n in range(0,5):
                if img[i+m-2][j+n-2]==0:
                    t=t+1
        if t==25 :
            constructed_closing[i][j]=0
        else:
            constructed_closing[i][j]=255
closing =cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening',opening)
cv2.waitKey(0)

cv2.imshow('constructed opening',constructed_opening)
cv2.waitKey(0)


cv2.imshow('closing',closing)
cv2.waitKey(0)

cv2.imshow('constructed closing',constructed_closing)
cv2.waitKey(0)

opening_result=opening-constructed_opening
closing_result=closing-constructed_closing

cv2.imshow('opening result',opening_result)
cv2.waitKey(0)
cv2.imshow('closing result',closing_result)
cv2.waitKey(0)




