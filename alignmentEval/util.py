import argparse
import cv2 as cv
import numpy as np
import os, sys
from io import BytesIO
from flow_vis import flow_to_color

try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def readFlow(filename):
    b = open(filename, 'rb').read()
    np_bytes = BytesIO(b)

    flo = np.load(np_bytes, allow_pickle=True)
    h, w = flo.shape[1:3]
    aux = np.zeros((w, h, 2), np.float32)
    aux[:, :, 0] = flo[0, :, :]
    aux[:, :, 1] = flo[1, :, :]
    return aux


def showImg(img, savePath):
    while True:
        k = cv.waitKey(30)
        cv.imshow('stack.png', img)

        if k == ord('s'):
            print('saving at {}'.format(savePath))
            cv.imwrite(savePath, img)
        if k == ord('n'):
            return True
        if k == ord('q'):
            return False


def showStack(n=0):
    file = 'img' + str(n)
    savePath = '../images/results/{}.png'.format(file)

    # row 1, unchanged pair + flow
    pathA = '../dataset/flow-comparison-semdefeito/{}/gt.png'.format(file)
    pathB = '../dataset/flow-comparison-semdefeito/{}/frame0.png'.format(file)
    pathFlow = '../flows/sem-defeito-comp/{}.flo'.format(file)

    frameA = cv.imread(pathA, cv.IMREAD_COLOR)
    frameB = cv.imread(pathB, cv.IMREAD_COLOR)
    opticalFlow = flow_to_color(readFlow(pathFlow))
    row1 = np.hstack((frameA, frameB, opticalFlow))
    showImg(row1, savePath)

    # row 2, synthetic pair + flow
    pathA = '../dataset/flow-comparison-comdefeito/{}/gt.png'.format(file)
    pathB = '../dataset/flow-comparison-comdefeito/{}/frame0.png'.format(file)
    pathFlow = '../flows/com-defeito-comp/{}.flo'.format(file)

    frameA = cv.imread(pathA, cv.IMREAD_COLOR)
    frameB = cv.imread(pathB, cv.IMREAD_COLOR)
    opticalFlow = flow_to_color(readFlow(pathFlow))
    row2 = np.hstack((frameA, frameB, opticalFlow))
    showImg(row2, savePath)

    showImg(np.vstack((row1, row2)), savePath)


def showComparisonDir():
    results = os.listdir('../dataset/flow-comparison-comdefeito')
    for file in results:
        pathA = '../dataset/flow-comparison-semdefeito/{}/gt.png'.format(file)
        pathB = '../dataset/flow-comparison-semdefeito/{}/frame0.png'.format(file)
        pathFlow = '../flows/sem-defeito-comp/{}.flo'.format(file)

        savePath = '../images/results/{}.png'.format(file)

        frameA = cv.imread(pathA, cv.IMREAD_COLOR)
        frameB = cv.imread(pathB, cv.IMREAD_COLOR)
        opticalFlow = flow_to_color(readFlow(pathFlow))
        row1 = np.hstack((frameA, frameB, opticalFlow))

        pathA = '../dataset/flow-comparison-comdefeito/{}/gt.png'.format(file)
        pathB = '../dataset/flow-comparison-comdefeito/{}/frame0.png'.format(file)
        pathFlow = '../flows/com-defeito-comp/{}.flo'.format(file)

        frameA = cv.imread(pathA, cv.IMREAD_COLOR)
        frameB = cv.imread(pathB, cv.IMREAD_COLOR)
        opticalFlow = flow_to_color(readFlow(pathFlow))
        row2 = np.hstack((frameA, frameB, opticalFlow))

        stack = np.vstack((row1, row2))

        if not showImg(stack, savePath):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--imgn', type=int, default=253)
    args = parser.parse_args()

    # showStack(args.imgn)
    showComparisonDir()

