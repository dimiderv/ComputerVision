import cv2
import numpy as np


filename = 'c:/image.jpg'
img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)



x=img.shape[0]

y=img.shape[1]

A=np.zeros((x+1,y+1))
Integral= np.zeros((x+1,y+1))
for i in range (1,x):
    for j in range (1,y):
        A[i][j]= img[i][j] + A[i][j-1]
        Integral[i][j] = Integral[i-1][j] + A[i][j]


# mean value integral
a=(1/250000)*(Integral[501,501]+Integral[1,1]-(Integral[1,501]+Integral[501,1]))
b=(1/240000)*(Integral[301,301]+Integral[801,801]-(Integral[301,801]+Integral[801,301]))
c=(1/480000)*(Integral[1,401]+Integral[801,1201]-(Integral[1,1201]+Integral[801,401]))


imageIntegral1 =cv2.integral( img)

f1=(1/250000)*(imageIntegral1[501,501]+imageIntegral1[1,1]-(imageIntegral1[1,501]+imageIntegral1[501,1]))
f2=(1/240000)*(imageIntegral1[301,301]+imageIntegral1[801,801]-(imageIntegral1[301,801]+imageIntegral1[801,301]))
f3=(1/480000)*(imageIntegral1[1,401]+imageIntegral1[801,1201]-(imageIntegral1[1,1201]+imageIntegral1[801,401]))
print('First result',a,'Second',b,'Third',c)
print('Cv2 results')
print('First',f1,'Second',f2,'Third',f3)
print('Their differences')
print('first',f1-a,'second',f2-b,'third',f3-c)










