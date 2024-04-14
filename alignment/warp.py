import os
from numpy.linalg import inv
import numpy as np
import cv2 as cv
from homography import *
from skimage.metrics import structural_similarity as ssim


def getMask(source, maskAux, H, size):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    grey = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(grey, 10, 255, cv.THRESH_BINARY)
    if maskAux is not None:
        warpedMask = cv.warpPerspective(maskAux, H, size, cv.INTER_LINEAR)
        mask = cv.bitwise_and(mask, cv.bitwise_not(warpedMask))

    mask = cv.erode(src=mask, kernel=kernel)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    return mask


def warpTwoTransparent(img1, img2, H, mask1, mask2, resizeToFit=True):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    ptsWarp = cv.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts2, ptsWarp), axis=0)

    if resizeToFit:
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

        T = np.float32([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

        warped = cv.warpPerspective(img1, T.dot(H), (xmax - xmin, ymax - ymin), cv.INTER_LINEAR)
        validArea = getMask(warped, mask1, H, (w2, h2))
        destination = np.zeros_like(warped)
        destination[-ymin:-ymin+h2, -xmin:-xmin+w2] = img2
        return np.where(validArea, warped, destination)
    else:
        warped = cv.warpPerspective(img1, H, (w2, h2), cv.INTER_LINEAR)
        validArea = getMask(warped, mask1, H, (w2, h2))

        destination = img2
        destination = np.where(validArea, warped, destination)

        return destination


def warpBurstToRef(i=0):
    ref = cv.imread('../dataset/multiframe-com-defeito/img{}/frame{}.png'.format(i, 0), cv.IMREAD_COLOR)
    maskr = cv.imread('../dataset/multiframe-com-defeito/img{}/mask{}.png'.format(i, 0), cv.IMREAD_COLOR)
    kpr, descr = keyPointsDetectionORB(ref, show=False, savePath='../images/results/orb1.png')
    try:
        os.mkdir('resultados/img{}'.format(i))
    except FileExistsError:
        pass

    for j in range(1, 10):

        mask = cv.imread('../dataset/multiframe-com-defeito/img{}/mask{}.png'.format(i, j), cv.IMREAD_COLOR)
        img = cv.imread('../dataset/multiframe-com-defeito/img{}/frame{}.png'.format(i, j), cv.IMREAD_COLOR)
        kp, desc = keyPointsDetectionORB(img, show=False, savePath='../images/results/orb2.png')

        matches = matchKeyPoints(img, kp, desc, ref, kpr, descr, show=False, algo='ORB')
        matches = matchesK(kp, kpr, matches, 50, show=False)
        H = findHomography(kp, kpr, matches, algo=None)
        img = warpTwoTransparent(img, ref, H, mask, maskr)
        showImg(img, '../images/results/warped.png')


def warpDatasetToEvaluate(k=40):
    baseFolder = '../dataset/flow-comparison-comdefeito'
    dirs = os.listdir(baseFolder)

    for dir in dirs:
        print('warping {}'.format(dir))
        source = cv.imread(os.path.join(baseFolder, dir, 'frame0.png'), cv.IMREAD_COLOR)
        destination = cv.imread(os.path.join(baseFolder, dir, 'gt.png'), cv.IMREAD_COLOR)

        sourceMask = cv.imread('../defeitos/flow/{}/mask/frame0.png'.format(dir), cv.IMREAD_GRAYSCALE)
        destinationMask = cv.imread('../defeitos/flow/{}/mask/gt.png'.format(dir), cv.IMREAD_GRAYSCALE)

        kpDestination, descDestination = keyPointsDetectionORB(destination, None, show=False)
        kpSource, descSource = keyPointsDetectionORB(source, None, show=False)

        matches = matchKeyPoints(source, kpSource, descSource, destination, kpDestination, descDestination, algo='ORB',
                                 show=False)
        matches = matchesK(kpSource, kpDestination, matches, k=k, show=False)
        H = findHomography(kpSource, kpDestination, matches, algo=None)
        warped = warpTwoTransparent(source, destination, H, sourceMask, destinationMask, resizeToFit=False)

        cv.imwrite('../images/results/warpingEvaluation/{}/{}-warped-{}.png'.format(k, dir, k), warped)
    print('done!')


if __name__ == '__main__':

    destination = cv.imread('../images/reais/piquet2.png')
    source = cv.imread('../images/reais/piquet1.png')
    # print(source.shape)
    # print(destination.shape)

    kpDestination, descDestination = keyPointsDetectionORB(destination, None, show=False, savePath='real/kpDest.png')
    kpSource, descSource = keyPointsDetectionORB(source, None, show=False, savePath='real/kpSource.png')

    matches = matchKeyPoints(source, kpSource, descSource, destination, kpDestination, descDestination, show=False, algo='ORB')
    matches = matchesK(kpSource, kpDestination, matches, 130, show=False)

    preview = cv.drawMatches(source, kpSource, destination, kpDestination, matches, None)
    showImg(preview, './real/preview.png')
    H = findHomography(kpSource, kpDestination, matches, algo=None)
    result = warpTwoTransparent(source, destination, H, None, None, resizeToFit=False)
    showImg(result, './real/result.png')

    print(cv.PSNR(result, destination))
    print(ssim(result, destination, data_range=result.max()-result.min(), channel_axis=2))
