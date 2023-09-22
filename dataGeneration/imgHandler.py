import os, sys, cv2, numpy
from math import pi
from random import randint, uniform
try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def projection(v, z0):
    x = v[0]/(v[2]+z0)
    y = v[1]/(v[2]+z0)
    return [x, y]


def rotateX(v, angle):
    x = v[0]
    y = v[1] * numpy.cos(angle) + v[2] * numpy.sin(angle)
    z = v[1] * -numpy.sin(angle) + v[2] * numpy.cos(angle)
    return [x, y, z]


def rotateY(v, angle):
    x = v[0]
    y = v[1] * numpy.cos(angle) + v[2] * numpy.sin(angle)
    z = v[1] * -numpy.sin(angle) + v[2] * numpy.cos(angle)
    return [x, y, z]


def randomWarpingTransform(maxRotation, minRotation=0, z=10, w=100, h=100, z0=1):
    box3d = [[-w, -h, z], [-w, h, z], [w, -h, z], [w, h, z]]

    # random angle
    xRotation = uniform(minRotation, maxRotation) * (pi / 180)
    yRotation = uniform(minRotation, maxRotation) * (pi / 180)

    # random direction
    xRotation *= -1 if randint(0, 1) else 1
    yRotation *= -1 if randint(0, 1) else 1

    dst = [rotateY(v, yRotation) for v in box3d]
    dst = [rotateX(v, xRotation) for v in dst]
    dst = [projection(v, z0) for v in dst]

    src = [projection(v, z0) for v in box3d]

    return cv2.getPerspectiveTransform(numpy.array(src, numpy.float32), numpy.array(dst, numpy.float32))


def randomTranslationTransform(minTranslation, maxTranslation):
    # random translation
    dx = randint(minTranslation, maxTranslation)
    dy = randint(minTranslation, maxTranslation)

    # random direction
    dx *= 1 if randint(0, 1) == 0 else -1
    dy *= 1 if randint(0, 1) == 0 else -1

    # print('dx: ' + str(dx) + ' dy: ' + str(dy))
    box = [[10, 20], [20, 10], [20, 20], [10, 10]]
    points = numpy.array(box, numpy.float32)
    displaced = numpy.array([[e[0] + dx, e[1] + dy] for e in box], numpy.float32)

    return cv2.getPerspectiveTransform(points, displaced)


def randomMasks(n):
    masks = [os.path.join(Paths['masks'], file) for file in os.listdir(Paths['masks'])]
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
    h, w, c = img.shape
    if w < width or h < height:
        scale = max(width/w, height/h)
        newWidth = w * scale
        newHeight = h * scale
        img = cv2.resize(img, (newWidth, newHeight))
        h, w, c = img.shape

    cropX = randint(0, w-width)
    cropY = randint(0, h-height)
    return img[cropY:cropY+height, cropX:cropX+width]
