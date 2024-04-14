import os, sys, cv2
import random

import numpy as np
from math import pi
from random import randint, uniform
try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def randomWarpingTransform(src, rotX, rotY, rotZ, distX, distY, distZ, f=500):
    dst = np.zeros_like(src)
    h, w = src.shape[:2]

    rotX = rotX*np.pi/180
    rotY = rotY*np.pi/180
    rotZ = rotZ*np.pi/180

    # Camera intrinsic matrix
    K = np.array([
        [f, 0, w/2, 0],
        [0, f, h/2, 0],
        [0, 0,   1, 0]
    ])

    # K inverse
    Kinv = np.zeros((4, 3))
    Kinv[:3, :3] = np.linalg.inv(K[:3, :3])*f
    Kinv[-1, :] = [0, 0, 1]

    # Rotation matrices around the X,Y,Z axis
    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotX), -np.sin(rotX), 0],
        [0, np.sin(rotX), np.cos(rotX), 0],
        [0, 0, 0, 1]])

    RY = np.array([
        [np.cos(rotY), 0, np.sin(rotY), 0],
        [0, 1, 0, 0],
        [-np.sin(rotY), 0, np.cos(rotY), 0],
        [0, 0, 0, 1]
    ])

    RZ = np.array([
        [np.cos(rotZ), -np.sin(rotZ), 0, 0],
        [np.sin(rotZ), np.cos(rotZ), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Composed rotation matrix with (RX,RY,RZ)
    R = np.linalg.multi_dot([RX, RY, RZ])

    # Translation matrix
    T = np.array([
        [1, 0, 0, distX],
        [0, 1, 0, distY],
        [0, 0, 1, distZ],
        [0, 0, 0, 1]
    ])

    # Overall homography matrix
    H = np.linalg.multi_dot([K, R, T, Kinv])

    # Apply matrix transformation
    cv2.warpPerspective(src, H, (w, h), dst, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

    return dst


def randomMasks(n, path):
    masks = [os.path.join(path, file) for file in os.listdir(path)]
    files = []
    while len(files) < n:
        mask = masks[randint(0, len(masks)-1)]
        if mask not in files:
            files.append(mask)
    return files


def readMask(filename):
    opts = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
    mask = cv2.imread(filename, cv2.IMREAD_COLOR)
    mask = cv2.rotate(mask, opts[randint(0, len(opts)-1)])
    return mask


def randomCrop(img, width, height):
    h = img.shape[0]
    w = img.shape[1]
    if w < width or h < height:
        scale = max(width/w, height/h)
        newWidth = int(w * scale)
        newHeight = int(h * scale)
        img = cv2.resize(img, (newWidth, newHeight))
        h, w, c = img.shape

    cropX = randint(0, w-width)
    cropY = randint(0, h-height)
    return img[cropY:cropY+height, cropX:cropX+width]


if __name__ == '__main__':
    src = cv2.imread('../images/indiana.png')
    # src = cv2.resize(src, (256, 256))
    for i in range(100):
        xr = random.uniform(-2, 2)
        yr = random.uniform(-2, 2)
        dst = randomWarpingTransform(src, xr, yr, 0, yr*-18, xr*18, 0)
        cv2.imwrite(os.path.join('../dataset', 'frame{}.png'.format(i)), dst.astype(np.uint8))
