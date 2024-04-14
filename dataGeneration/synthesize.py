import argparse
import logging
import os, sys, cv2
import numpy as np
from math import ceil
from PIL import Image
from random import randint, uniform, choice
from wan2020 import synthesize_gaussian, synthesize_speckle, synthesize_salt_pepper, synthesize_low_resolution, \
    blur_image_v2
from imgHandler import randomMasks, randomWarpingTransform, readMask, randomCrop

try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def generateSyntheticBurst(imgPath, savePath, n, textures, masks, cropSize=256, **opt):
    textures = [os.path.join(textures, file) for file in os.listdir(textures)]
    masks = randomMasks(n, masks)  # set of n random different masks

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
    xCrop = int(w/2 - cropSize/2)
    yCrop = int(h/2 - cropSize/2)

    logging.debug('generating {} frames for current image'.format(n))
    for i in range(n):
        # convert to PIL to use wan2020's functions
        frame_i = Image.fromarray(np.uint8(img)).convert('RGB')

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
        frame_i = np.asarray(frame_i)

        # read random mask and texture
        logging.debug('{}'.format(masks[i]))
        mask = readMask(masks[i])
        texture = cv2.imread(textures[randint(0, len(textures) - 1)], cv2.IMREAD_COLOR)

        # apply spatial transformations
        xr = uniform(opt['rotationRangeX'][0], opt['rotationRangeX'][1])
        yr = uniform(opt['rotationRangeY'][0], opt['rotationRangeY'][1])
        dx = randint(opt['translationRangeX'][0], opt['translationRangeX'][1])
        dy = randint(opt['translationRangeY'][0], opt['translationRangeY'][1])
        logging.debug('xr: {} yr: {} dx: {} dy: {}'.format(xr, yr, dx, dy))
        frame_i = randomWarpingTransform(frame_i, xr, yr, 0, dx, dy, 0)

        # crop image or resize
        if opt['crop']:
            gt = img[yCrop:yCrop + cropSize, xCrop:xCrop + cropSize]
            frame_i = frame_i[yCrop:yCrop + cropSize, xCrop:xCrop + cropSize]
            texture = randomCrop(texture, cropSize, cropSize)
            mask = randomCrop(mask, cropSize, cropSize)

        else:
            mask = cv2.resize(mask, (w, h))
            texture = cv2.resize(texture, (w, h))
            gt = img

        # blend image and texture
        if opt['blendTexture']:
            alpha = uniform(opt['alphaRange'][0], opt['alphaRange'][1])
            beta = 1.0 - alpha
            gamma = 0.0
            frame_i = cv2.addWeighted(frame_i, alpha, texture, beta, gamma)

        # write Ground Truth image
        if opt['saveGT']:
            cv2.imwrite(os.path.join(savePath, 'gt.png'), gt.astype(np.uint8))

        # apply scratches
        if opt['applyScratches']:
            frame_i = (mask/255) * texture + frame_i * (1 - mask/255)
            # frame_i = np.where(mask == (255, 255, 255), texture, frame_i)
            # write masks
            if opt['saveMask']:
                cv2.imwrite(os.path.join(savePath, 'mask{}.png'.format(i)), mask.astype(np.uint8))

        # write synthesized frame
        cv2.imwrite(os.path.join(savePath, 'frame{}.png'.format(i)), frame_i.astype(np.uint8))

        logging.debug('frame {} created'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=os.path.join(Paths['images'], 'img.png'))
    parser.add_argument('--dest', type=str, default=Paths['output'])
    parser.add_argument('--texture', type=str, default=Paths['textures'])
    parser.add_argument('--mask', type=str, default=Paths['masks'])
    parser.add_argument('--cropSize', type=int, default=256)
    parser.add_argument('-n', '--burst_size', type=int, default=10)

    args = parser.parse_args()

    options = {
        'translationRangeX': (0.0, 0.0),
        'translationRangeY': (0.0, 0.0),
        'rotationRangeX': (0, 0),
        'rotationRangeY': (0, 0),
        'noiseProbability': 0.0,
        'lowResProbability': 0.0,
        'blurProbability': 0.0,
        'noiseTypes': ['gaussian', 'speckle'],
        'crop': True,
        'applyScratches': True,
        'saveMask': False,
        'blendTexture': True,
        'saveGT': False,
        'alphaRange': (0.75, 0.85),
        'speckleRange': (5, 10),
        'gaussianRange': (5, 10),
    }

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    generateSyntheticBurst(args.path,
                           args.dest,
                           args.burst_size,
                           args.texture,
                           args.mask,
                           cropSize=args.cropSize,
                           **options)

