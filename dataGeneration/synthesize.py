import logging
import os, sys, cv2, numpy
from math import ceil
from PIL import Image
from random import randint, uniform, choice
from wan2020 import synthesize_gaussian, synthesize_speckle, synthesize_salt_pepper, synthesize_low_resolution, \
    blur_image_v2
from imgHandler import randomMasks, randomWarpingTransform, randomTranslationTransform, readMask, randomCrop

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

    # if image smaller than cropSize upscale image keeping the same aspect ratio
    if w < cropSize or h < cropSize:
        scale = 256/min(w, h)
        newWidth = int(ceil(w*scale))
        newHeight = int(ceil(h*scale))
        logging.debug('image {} is too small, {}x{} resizing to {}x{}'.format(os.path.basename(imgPath), w, h,
                                                                              newWidth, newHeight))
        img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)
        h, w, c = img.shape

    # define crop's position for all frames in this burst
    xcrop = randint(0, w - cropSize)
    ycrop = randint(0, h - cropSize)

    logging.debug('generating {} frames for current image'.format(n))
    for i in range(n):
        # generate transformation matrices
        M_displacement = randomTranslationTransform(opt['translationRange'][0], opt['translationRange'][1])
        M_warp = randomWarpingTransform(opt['rotationRange'][0], opt['rotationRange'][1])

        # convert to PIL to use wan2020's functions
        frame_i = Image.fromarray(numpy.uint8(img)).convert('RGB')

        # add noise and degradation
        if uniform(0, 1) < opt['lowResProbability']:
            logging.debug('low resolution')
            frame_i = synthesize_low_resolution(frame_i)

        if uniform(0, 1) < opt['blurProbability']:
            logging.debug('blur')
            frame_i = blur_image_v2(frame_i)

        if uniform(0, 1) < opt['noiseProbability']:
            noiseType = choice(opt['noiseTypes'])
            if noiseType == 'gaussian':
                logging.debug('gaussian')
                frame_i = synthesize_gaussian(frame_i, opt['gaussianRange'][0], opt['gaussianRange'][1])

            if noiseType == 'speckle':
                logging.debug('speckle')
                frame_i = synthesize_speckle(frame_i, opt['speckleRange'][0], opt['speckleRange'][1])

            if noiseType == 'salt&pepper':
                logging.debug('salt & pepper')
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
            texture = randomCrop(texture, cropSize, cropSize)

        else:
            mask = cv2.resize(mask, (w, h))
            texture = cv2.resize(texture, (w, h))
            gt = img

        # blend image and texture
        alpha = uniform(opt['alphaRange'][0], opt['alphaRange'][1])
        beta = 1.0 - alpha
        gamma = 0.0
        frame_i = cv2.addWeighted(frame_i, alpha, texture, beta, gamma)

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
        'translationRange': (0.0, 0.0),
        'rotationRange': (0.0, 0.0),
        'speckleRange': (5, 30),
        'gaussianRange': (5, 30),
        'noiseProbability': 0.5,
        'noiseTypes': ['gaussian', 'speckle'],
        'crop': True,
        'applyScratches': True,
        'lowResProbability': 0.8,
        'blurProbability': 0.05,
        'alphaRange': (0.75, 0.85),
    }
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    generateSyntheticBurst(os.path.join(Paths['images'], 'indiana.png'), Paths['images'], 10, **options)

