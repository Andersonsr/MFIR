import numpy as np
import cv2 as cv
import argparse


def comparison2x2stack(gt1, dst1, gt2, dst2):
    # print(gt1.shape)
    # print(gt2.shape)
    # print(dst2.shape)
    # print(dst1.shape)

    sd = np.hstack((gt1, dst1))
    cd = np.hstack((gt2, dst2))
    fin = np.vstack((sd, cd))
    cv.imwrite('comp2x2.png', fin)


def comparison(imgN):
    gt1Path = '../dataset/flow-comparison-semdefeito/img{}/gt.png'.format(imgN)
    gt2Path = '../dataset/flow-comparison-comdefeito/img{}/gt.png'.format(imgN)
    dst1Path = '../dataset/flow-comparison-semdefeito/img{}/frame0.png'.format(imgN)
    dst2Path = '../dataset/flow-comparison-comdefeito/img{}/frame0.png'.format(imgN)

    gt1 = cv.imread(gt1Path, cv.IMREAD_COLOR)
    gt2 = cv.imread(gt2Path, cv.IMREAD_COLOR)
    dst1 = cv.imread(dst1Path, cv.IMREAD_COLOR)
    dst2 = cv.imread(dst2Path, cv.IMREAD_COLOR)

    comparison2x2stack(gt1, dst1, gt2, dst2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--img', type=int, default=0, help='image id number')
    args = parser.parse_args()

    comparison(args.img)
