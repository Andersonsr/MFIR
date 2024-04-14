import os
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def evaluateK():
    path = '../dataset/flow-comparison-comdefeito/'
    imgs = os.listdir(path)
    ks = [40, 50, 60, 70, 80, 90, 100]
    psnr = []
    sim = []
    for img in imgs:
        psnrLine = []
        simLine = []

        reference = cv.imread(os.path.join(path, img, 'gt.png'), cv.IMREAD_COLOR)
        notWarped = cv.imread(os.path.join(path, img, 'frame0.png'), cv.IMREAD_COLOR)
        psnrLine.append(cv.PSNR(reference, notWarped))
        simLine.append(ssim(reference, notWarped, data_range=notWarped.max() - notWarped.min(), channel_axis=2))

        for k in ks:
            warped = cv.imread('../images/results/warpingEvaluation/k{}/{}-warped-{}.png'.format(k, img, k))
            psnrLine.append(cv.PSNR(reference, warped))
            simLine.append(ssim(reference, warped, data_range=notWarped.max() - notWarped.min(), channel_axis=2))

        nomask = cv.imread('../images/results/warpingEvaluation/k50nomask/{}-warped-50.png'.format(img), cv.IMREAD_COLOR)
        psnrLine.append(cv.PSNR(reference, nomask))
        simLine.append(ssim(reference, nomask, data_range=nomask.max()-nomask.min(), channel_axis=2))

        ransacked = cv.imread('../images/results/warpingEvaluation/33/{}-warped-33.png'.format(img), cv.IMREAD_COLOR)
        psnrLine.append(cv.PSNR(reference, ransacked))
        simLine.append(ssim(reference, ransacked, data_range=nomask.max() - nomask.min(), channel_axis=2))

        psnr.append(psnrLine)
        sim.append(simLine)

    with open('psnr.txt', 'w') as file:
        for row in psnr:
            file.write('{} {} {} {} {} {} {} {} {} {}\n'.format(row[0], row[1], row[2], row[3],
                                                          row[4], row[5], row[6], row[7], row[8], row[9]))
        file.close()

    with open('ssim.txt', 'w') as file:
        for row in sim:
            file.write('{} {} {} {} {} {} {} {} {} {}\n'.format(row[0], row[1], row[2], row[3],
                                                          row[4], row[5], row[6], row[7], row[8], row[9]))
        file.close()


def plotHists():
    psnr = np.genfromtxt('psnr.txt')
    sim = np.genfromtxt('ssim.txt')

    plt.subplot(1, 2, 1)
    plt.hist(psnr[:, 0], label='sem warp', density=False)
    plt.hist(psnr[:, 2], label='k50', density=False)
    plt.xlabel('psnr')
    plt.ylabel('frequência')

    plt.subplot(1, 2, 2)
    plt.hist(sim[:, 0], label='sem warp', density=False)
    plt.hist(sim[:, 2], label='k50', density=False)
    plt.xlabel('ssim')
    plt.ylabel('frequência')

    plt.show()


def plotMean():
    psnr = np.genfromtxt('psnr.txt')
    sim = np.genfromtxt('ssim.txt')

    plt.subplot(1, 2, 1)
    plt.bar(1, np.mean(psnr[:, 0]), yerr=np.std(psnr[:, 0]), label='sem warp', capsize=10)
    # plt.bar(2, np.mean(psnr[:, 1]), yerr=np.std(psnr[:, 1]), label='k40+mask', capsize=10)
    # plt.bar(3, np.mean(psnr[:, 2]), yerr=np.std(psnr[:, 2]), label='k50+mask', capsize=10)
    # plt.bar(4, np.mean(psnr[:, 3]), yerr=np.std(psnr[:, 3]), label='k60+mask', capsize=10)
    # plt.bar(5, np.mean(psnr[:, 4]), yerr=np.std(psnr[:, 4]), label='k70+mask', capsize=10)
    # plt.bar(6, np.mean(psnr[:, 5]), yerr=np.std(psnr[:, 5]), label='k80+mask', capsize=10)
    # plt.bar(7, np.mean(psnr[:, 6]), yerr=np.std(psnr[:, 6]), label='k90+mask', capsize=10)
    # plt.bar(8, np.mean(psnr[:, 7]), yerr=np.std(psnr[:, 7]), label='k100+mask', capsize=10)
    plt.bar(2, np.mean(psnr[:, 8]), yerr=np.std(psnr[:, 8]), label='k50', capsize=10)
    plt.bar(4, np.mean(psnr[:, 9]), yerr=np.std(psnr[:, 9]), label='ransac+mask', capsize=10)

    plt.title('PSNR')
    plt.legend(loc='lower right')
    plt.xticks([])

    plt.subplot(1, 2, 2)
    plt.title('SSIM')
    plt.bar(1, np.mean(sim[:, 0]), yerr=np.std(sim[:, 0]), label='sem warp', capsize=10)
    # plt.bar(2, np.mean(sim[:, 1]), yerr=np.std(sim[:, 1]), label='k40+mask', capsize=10)
    # plt.bar(3, np.mean(sim[:, 2]), yerr=np.std(sim[:, 2]), label='k50+mask', capsize=10)
    # plt.bar(4, np.mean(sim[:, 3]), yerr=np.std(sim[:, 3]), label='k60+mask', capsize=10)
    # plt.bar(5, np.mean(sim[:, 4]), yerr=np.std(sim[:, 4]), label='k70+mask', capsize=10)
    # plt.bar(6, np.mean(sim[:, 5]), yerr=np.std(sim[:, 5]), label='k80+mask', capsize=10)
    # plt.bar(7, np.mean(sim[:, 6]), yerr=np.std(sim[:, 6]), label='k90+mask', capsize=10)
    # plt.bar(8, np.mean(sim[:, 7]), yerr=np.std(sim[:, 7]), label='k100+mask', capsize=10)
    plt.bar(2, np.mean(sim[:, 8]), yerr=np.std(sim[:, 8]), label='k50', capsize=10)
    plt.bar(4, np.mean(sim[:, 9]), yerr=np.std(sim[:, 9]), label='ransac+mask', capsize=10)

    print('k50 ssim mean: ' + str(np.mean(sim[:, 8])) + ' std: ' + str(np.std(sim[:, 8])))
    print('k50 psnr mean: ' + str(np.mean(psnr[:, 8])) + ' std: ' + str(np.std(psnr[:, 8])))

    print('k50+mask ssim mean: ' + str(np.mean(sim[:, 2])) + ' std: ' + str(np.std(sim[:, 2])))
    print('k50+mask psnr mean: ' + str(np.mean(psnr[:, 2])) + ' std: ' + str(np.std(psnr[:, 2])))

    print('ransac+mask ssim mean: ' + str(np.mean(sim[:, 9])) + ' std: ' + str(np.std(sim[:, 9])))
    print('ransac+mask psnr mean: ' + str(np.mean(psnr[:, 9])) + ' std: ' + str(np.std(psnr[:, 9])))

    plt.legend(loc='lower right')
    plt.xticks([])
    plt.show()


if __name__ == '__main__':
    # evaluateK()
    # plotHists()
    plotMean()
