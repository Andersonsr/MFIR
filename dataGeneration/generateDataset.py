import os, sys, argparse, logging
import random
from synthesize import generateSyntheticBurst
try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def generateDataset(readDirectory, savePath, burstSize, **opt):
    formats = ['png', 'jpg', 'jpeg']
    imageList = os.listdir(readDirectory)
    # random.shuffle(imageList)
    counter = len(os.listdir(savePath))

    if counter > 0:
        logging.warning('destination directory is not empty')
    else:
        logging.debug('generating bursts for images in {}\n'.format(readDirectory))

        for img in imageList:
            img = img.strip('\n')
            logging.debug('generating burst for {}'.format(img))

            if img.split('.')[-1] in formats:
                directory = os.path.join(savePath, 'img{}'.format(counter))
                os.mkdir(directory)
                generateSyntheticBurst(os.path.join(readDirectory, img),
                                       directory,
                                       burstSize,
                                       Paths['textures'],
                                       Paths['masks'],
                                       **opt)
                counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=Paths['images'],
                        help='path to folder containing images used as input')
    parser.add_argument('-o', '--output', type=str, default=Paths['output'],
                        help='path to folder used to store the result')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                        help='show debugging level log')
    parser.add_argument('-n', '--burstSize', type=int, default=10, help='show debugging level log')
    parser.add_argument('-l', '--list', type=str, default=None, help='list of imgs')
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
        'saveMask': True,
        'blendTexture': True,
        'saveGT': False,
        'alphaRange': (0.75, 0.8),
        'speckleRange': (5, 10),
        'gaussianRange': (5, 10),
    }

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    if args.list is None:
        generateDataset(args.input, args.output, args.burstSize, **options)

    else:
        lista = open(args.list, 'r')
        for row in lista:
            row = row.strip('\n')
            name = row.split('/')[-1]
            name = name.split('.')[-2]
            dirPath = os.path.join(args.output, row.split('/')[-2], name)

            try:
                os.makedirs(dirPath)
            except FileExistsError:
                logging.debug('file {} already exist'.format(dirPath))

            logging.debug('creating directory: {}'.format(dirPath))
            logging.debug('input file: '+row)
            logging.debug('output file: '+dirPath)

            generateSyntheticBurst(row,
                                   dirPath,
                                   args.burstSize,
                                   Paths['textures'],
                                   Paths['masks'],
                                   **options)
