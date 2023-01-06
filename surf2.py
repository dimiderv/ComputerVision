
import numpy as np
import cv2 as cv

surf = cv.xfeatures2d_SURF.create(500, 4, 3, 0, 1)



img1 = cv.imread('1.jpg')
img1 = cv.resize(img1, (0,0), fx=0.2, fy=0.2)
cv.namedWindow('image1')
cv.imshow('image1', img1)
cv.waitKey(0)

# Key points and descriptor of 1st image
kp1 = surf.detect(img1)
dimg = cv.drawKeypoints(img1, kp1, None, 100, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('main', dimg)
cv.waitKey(0)

desc1 = surf.compute(img1, kp1)

# Key points and descriptor of 2nd image

img2 = cv.imread('2.jpg')
img2 = cv.resize(img2, (0,0), fx=0.2, fy=0.2)

cv.namedWindow('image2')
cv.imshow('image2', img2)
cv.waitKey(0)
kp2 = surf.detect(img2)
desc2 = surf.compute(img2, kp2)

def match(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    for i in range(n1):
        fv = d1[i, :]
        diff = d2 - fv
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        distances[i2] = np.inf

        i3 = np.argmin(distances)
        mindist3 = distances[i3]

        if mindist2 / mindist3 < 0.5:
            matches.append(cv.DMatch(i, i2, mindist2))

    return matches


matches = match(desc1[1], desc2[1])

img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches])
img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches])

M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)
# Βρίσκει πώς πρέπει να μετατραπει η πρώτη για να "ταιριάξει" με τη δευτερη

# Enwsh twn 2 eikonwn
img3 = cv.warpPerspective(img2, M, (img1.shape[1]+200, img1.shape[0]+400))
img3[0: img2.shape[0], 0: img2.shape[1]] = img1

cv.namedWindow('1st and 2nd image', cv.WINDOW_NORMAL)
cv.imshow('1st and 2nd image', img3)
cv.waitKey(0)

# Key points ths enwshs twn eikonwn 1 kai 2

kp3 = surf.detect(img3)
desc3 = surf.compute(img3, kp3)
dimg = cv.drawKeypoints(img1, kp3, 100, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('enwsh 1,2 keypoints', dimg)
cv.waitKey(0)
# key points, descriptor ths 3hs eikonas
img4 = cv.imread('4.jpg')
img4 = cv.resize(img4, (0,0), fx=0.2, fy=0.2)

cv.namedWindow('image3')
cv.imshow('image3', img4)
cv.waitKey(0)


kp4 = surf.detect(img4)
desc4 = surf.compute(img4, kp4)

# key points descriptors ths 4hs eikonas

img5 = cv.imread('3.jpg')
img5 = cv.resize(img5,(0,0), fx=0.2, fy=0.2)

cv.namedWindow('image4')
cv.imshow('image4', img5)
cv.waitKey(0)


kp5 = surf.detect(img5)
desc5 = surf.compute(img5, kp5)

# Omoia keypoints
matches = match(desc4[1], desc5[1])

img_pt4 = np.array([kp4[x.queryIdx].pt for x in matches])
img_pt5 = np.array([kp5[x.trainIdx].pt for x in matches])

M, mask = cv.findHomography(img_pt5, img_pt4, cv.RANSAC)
# Βρίσκει πώς πρέπει να μετατραπει η πρώτη για να "ταιριάξει" με τη δευτερη
# enwsh 3 kai 4
img6 = cv.warpPerspective(img5, M, (img4.shape[1]+200, img4.shape[0]+600))
img6[0: img5.shape[0], 0: img5.shape[1]] = img4
cv.namedWindow('img6',cv.WINDOW_NORMAL)
cv.imshow('img6',img6)
cv.waitKey(0)

#keypoint ths enwsh ths 3hs kai 4hs eikonas
kp6 = surf.detect(img6)
desc6 = surf.compute(img6, kp6)
dimg = cv.drawKeypoints(img1, kp6, 100, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('enwsh 3,4 keypoints', dimg)
cv.waitKey(0)
# omoia keypoints twn enwsewn
matches = match(desc3[1], desc6[1])

img_pt1 = np.array([kp3[x.queryIdx].pt for x in matches])
img_pt2 = np.array([kp6[x.trainIdx].pt for x in matches])

M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)
# telikh eikona
panoramic = cv.warpPerspective(img6, M, (2*img3.shape[1]+400, img3.shape[0]))
panoramic[0: img6.shape[0], 0: img6.shape[1]] = img3

cv.namedWindow('panoramic', cv.WINDOW_NORMAL)
cv.imshow('panoramic', panoramic)
cv.waitKey(0)

