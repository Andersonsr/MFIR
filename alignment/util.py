import cv2 as cv
import numpy as np
import os, sys

try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def readFlow(filename):
    bytes = open(filename).read()

    return flo


def showFlow(frame1, frame2, flow):
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    while True:
        cv.imshow('test', bgr)
        key = cv.waitKey(30)
        if key >= 0:
            break


if __name__ == '__main__':
    flowPath = os.path.join(Paths['root'], 'flows', 'sem-defeito-result', 'img0-1-7.flo')
    frame1Path = os.path.join(Paths['images'], 'sem-defeito', 'img0', 'frame1.png')
    frame2Path = os.path.join(Paths['images'], 'sem-defeito', 'img0', 'frame7.png')
    frame1 = cv.imread(frame1Path, cv.IMREAD_COLOR)
    frame2 = cv.imread(frame2Path, cv.IMREAD_COLOR)
    flow = readFlow(flowPath)
    # showFlow(frame1, frame2, flow)

