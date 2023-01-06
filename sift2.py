import numpy as np
import cv2 as cv

img1 = cv.imread('c:/notre-dame.jpg')

cv.namedWindow('main')
cv.imshow('main', img1)
cv.waitKey(0)

sift = cv.xfeatures2d_SIFT.create(100)

keypoints1 = sift.detect(img1)

dimg = cv.drawKeypoints(img1, keypoints1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('main', dimg)
cv.waitKey(0)

descriptors1 = sift.compute(img1, keypoints1)

img2 = cv.imread('c:/notre-dame-partial-rotate.jpg')

cv.namedWindow('main2')
cv.imshow('main2', img2)
cv.waitKey(0)

keypoints2 = sift.detect(img2)
descriptors2 = sift.compute(img2, keypoints2)

dimg2 = cv.drawKeypoints(img2, keypoints2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('main2', dimg2)
cv.waitKey(0)

img3 = cv.imread('c:/road1.jpg')
keypoints3 = sift.detect(img3)
descriptors3 = sift.compute(img3, keypoints3)

def match1(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    for i in range(n1):
        fv = d1[i, :]
        diff = d2 - fv #ana grammh afairei [1,128]-[100,128] kai katalabanei ti prepei na kanei
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1) #pros8etei ana sthles dld o distances einai [100,1]

        i2 = np.argmin(distances)# 8esh tis min timhs tou distances
        mindist2 = distances[i2]

        matches.append(cv.DMatch(i, i2, mindist2))

    return matches

def match2(d1, d2):
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
#to kanei apeiro gia na parei to amesw mikrotero
        i3 = np.argmin(distances)
        mindist3 = distances[i3]
#pairnei ton logo ton duo mikroterwna apostasewn kai an brei polu mikres apoastaseiss ta kanei match
#kai kanei ton logo tous o poiois an einai mikros polu 8a parei auto to key point
#to 0.5 exei megalueterj anoxh
        if mindist2 / mindist3 < 0.5:
            matches.append(cv.DMatch(i, i2, distances[i2]))
#adeia lista bazei akoma ena allo stoixeio

    return matches


#matches = match1(descriptors1[1], descriptors2[1])
#matches = match2(descriptors1[1],descriptors2[1])
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1[1], descriptors2[1])

dimg = cv.drawMatches(img1, descriptors1[0], img2, descriptors2[0], matches, None)#o descriptors exei mesa pinaka einai taple

cv.imshow('main', dimg)
cv.waitKey(0)
pass