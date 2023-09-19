import os, sys, argparse, logging
import random
from synthesize import generateSyntheticBurst
try:
    from paths import Paths
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import Paths


def generateDataset(readDirectory, savePath):
    formats = ['png', 'jpg']
    imageList = os.listdir(readDirectory)
    counter = 0

    if len(os.listdir(savePath)) > 0:
        logging.debug('destination directory is not empty')
    else:
        logging.debug('generating bursts for images in {}\n'.format(readDirectory))

        for img in imageList:
            logging.debug('generating burst for {}'.format(img))

            if img.split('.')[-1] in formats:
                directory = os.path.join(savePath, 'img{}'.format(counter))
                os.mkdir(directory)

                options = {
                    'maxTranslation': 0,
                    'minTranslation': 0,
                    'maxRotation': 0,
                    'minRotation': 0,
                    'noiseTypes': [1, 2],
                    'crop': True,
                    'applyScratches': True,
                }

                generateSyntheticBurst(os.path.join(readDirectory, img), directory, 10, **options)
                counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=Paths['images'],
                        help='path to folder containing images used as input')
    parser.add_argument('-o', '--output', type=str, default=Paths['output'],
                        help='path to folder used to store the result')
    parser.add_argument('-d', '--debug', default=False, action='store_true',
                        help='show debugging level log')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    status = generateDataset(args.input, args.output)
