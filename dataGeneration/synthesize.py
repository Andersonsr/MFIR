import logging
import os, sys, cv2, numpy
from math import ceil
from PIL import Image
from random import randint, uniform, choice
from wan2020 import synthesize_gaussian, synthesize_speckle, synthesize_salt_pepper, synthesize_low_resolution
from imgHandler import randomMasks, randomWarpingTransform, randomTranslationTransform, readMask

try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def generateSyntheticBurst(imgPath, savePath, n, cropSize=256, **opt):
    textures = [os.path.join(Paths['textures'], file) for file in os.listdir(Paths['textures'])]
    masks = randomMasks(n)

    logging.debug('source: {}'.format(imgPath))
    logging.debug('destination: {}'.format(savePath))

    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    h, w, c = img.shape

    if w < 256 or h < 256:
        scale = 256/min(w, h)
        newWidth = int(ceil(w*scale))
        newHeight = int(ceil(h*scale))
        logging.debug('image {} is too small, {}x{} resizing to {}x{}'.format(os.path.basename(imgPath), w, h,
                                                                              newWidth, newHeight))
        img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)
        h, w, c = img.shape

    # crop's top left corner
    xcrop = randint(0, w - cropSize)
    ycrop = randint(0, h - cropSize)

    logging.debug('generating {} frames for current image'.format(n))
    for i in range(n):
        # generate transformation matrices
        M_displacement = randomTranslationTransform(opt['minTranslation'], opt['maxTranslation'])
        M_warp = randomWarpingTransform(opt['maxRotation'], opt['minRotation'])

        # convert to PIL
        frame_i = Image.fromarray(numpy.uint8(img)).convert('RGB')

        # add noise
        noiseType = choice(opt['noiseTypes'])
        if noiseType == 1:
            frame_i = synthesize_gaussian(frame_i, 30, 30)

        if noiseType == 2:
            frame_i = synthesize_speckle(frame_i, 5, 50)

        if noiseType == 3:
            frame_i = synthesize_salt_pepper(frame_i, uniform(0, 0.01), uniform(0.3, 0.8))

        # convert back to numpy
        frame_i = numpy.asarray(frame_i)

        # apply transformations
        frame_i = cv2.warpPerspective(frame_i, M_displacement, (w, h))
        frame_i = cv2.warpPerspective(frame_i, M_warp, (w, h))

        # read random mask and texture
        mask = readMask(masks[i])
        texture = cv2.imread(textures[randint(0, len(textures) - 1)], cv2.IMREAD_COLOR)

        # crop image
        if opt['crop']:
            gt = img[ycrop:ycrop + cropSize, xcrop:xcrop + cropSize]
            frame_i = frame_i[ycrop:ycrop+cropSize, xcrop:xcrop+cropSize]
        else:
            mask = cv2.resize(mask, (w, h))
            texture = cv2.resize(texture, (w, h))

        # write Ground Truth image
        cv2.imwrite(os.path.join(savePath, 'gt.png'), gt.astype(numpy.uint8))

        # apply scratches
        frame_i = numpy.where(mask == (255, 255, 255), texture, frame_i)

        # write synthesized frame
        cv2.imwrite(os.path.join(savePath, 'frame{}.png'.format(i)), frame_i.astype(numpy.uint8))

        # write masks
        cv2.imwrite(os.path.join(savePath, 'mask{}.png'.format(i)), mask.astype(numpy.uint8))

        logging.debug('frame {} created'.format(i))


if __name__ == '__main__':
    options = {
        'maxTranslation': 0,
        'minTranslation': 0,
        'maxRotation': 0,
        'minRotation': 0,
        'noiseTypes': [1],
        'crop': False,
        'applyScratches': True,
    }
    generateSyntheticBurst(os.path.join(Paths['images'], 'indiana.png'), Paths['images'], 10, **options)

