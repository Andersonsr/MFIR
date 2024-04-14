import os, sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from alignmentEval.util import showImg
from math import ceil


def keyPointsDetectionSift(img, mask=None, show=True, savePath='../images/results/sift.png'):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray, mask)
    kp, descriptor = sift.compute(gray, kp)
    if show:
        detected = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        showImg(detected, savePath)
    return kp, descriptor


# usar orb, nao esquecer de usar a distancia de Hamming
def keyPointsDetectionORB(img, mask=None, show=True, savePath='../images/results/sift.png'):
    orb = cv.ORB_create()
    kp, desc = orb.detectAndCompute(img, mask)
    if show:
        tmp = cv.drawKeypoints(img, kp, None)
        showImg(tmp, savePath)
    return kp, desc


def matchKeyPoints(img1, kp1, des1, img2, kp2, des2, algo='SIFT', show=True, savePath='../images/results/matcher.png'):
    if algo == 'SIFT':
        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        matches = good
        matched = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    elif algo == 'ORB':
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matched = cv.drawMatches(img1, kp1, img2, kp2, matches, None)

    if show:
        showImg(matched, savePath)

    return matches


def findHomography(kp1, kp2, matches, algo=cv.RANSAC):
    points1 = []
    points2 = []
    for m in matches:
        x1 = kp1[m.queryIdx].pt[0]
        y1 = kp1[m.queryIdx].pt[1]
        points1.append([x1, y1])

        x2 = kp2[m.trainIdx].pt[0]
        y2 = kp2[m.trainIdx].pt[1]
        points2.append([x2, y2])

    arr1 = np.array(points1)
    arr2 = np.array(points2)

    H, _ = cv.findHomography(arr1, arr2, algo)
    return H


def matchesK(kp1, kp2, matches, k, show=True):
    # print(matches)
    vs = []
    dists = []
    index = 0
    for m in matches:
        # print(m)
        a = kp1[m.queryIdx].pt
        b = kp2[m.trainIdx].pt
        vs.append([a[0] - b[0], a[1] - b[1], 0, m.queryIdx, m.trainIdx, index])
        dists.append(np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2)))
        index += 1

    vs = np.array(vs)
    bestCentroid = [0, 0]
    bestValue = 9999999999999
    for v in vs:
        centrox = v[0]
        centroy = v[1]

        vs[:, 2] = np.sqrt(np.power(vs[:, 0] - centrox, 2) + np.power(vs[:, 1] - centroy, 2))
        sorted_arr = vs[np.argsort(vs[:, 2])]
        value = np.mean(sorted_arr[:k, 2])
        if value < bestValue:
            bestValue = value
            bestCentroid[0] = centrox
            bestCentroid[1] = centroy

    vs[:, 2] = np.sqrt(np.power(vs[:, 0] - bestCentroid[0], 2) + np.power(vs[:, 1] - bestCentroid[1], 2))
    sorted_arr = vs[np.argsort(vs[:, 2])]

    if show:
        plt.subplot(1, 2, 1)
        plt.scatter(sorted_arr[k:, 0], sorted_arr[k:, 1], label={'distâncias'})
        plt.scatter(sorted_arr[:k, 0], sorted_arr[:k, 1], label='K próximos')
        plt.scatter(bestCentroid[0], bestCentroid[1], label='centro')
        plt.ylabel('dy')
        plt.xlabel('dx')

        plt.subplot(1, 2, 2)
        plt.hist(dists, bins=ceil(len(dists)/10))
        plt.xlabel('distância')
        plt.ylabel('frequência')
        plt.show()

    result = []
    for s in sorted_arr[:k, :]:
        index = int(s[-1])
        result.append(matches[index])

    return result


if __name__ == '__main__':
    parent = '../dataset/multiframe-com-defeito/'
    directory = os.listdir(parent)
    result = []
    for sub in directory:
        path = os.path.join(parent, sub, 'frame0.png')
        img = cv.imread(path, cv.IMREAD_COLOR)
        kps, descs = keyPointsDetectionORB(img, None, show=False)
        result.append(len(kps))

    print(np.mean(result))
    print(np.std(result))
