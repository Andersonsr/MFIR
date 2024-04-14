from flow_vis import flow_to_color
from util import readFlow
import numpy as np
import os


def averageError(flow1, flow2):
    w = flow1.shape[1]
    h = flow2.shape[2]
    dist = np.zeros((w, h))
    dist = np.sqrt(np.power(flow1[0, ...] - flow2[0, ...], 2) + np.power(flow1[1, ...] - flow2[1, ...], 2))
    EPE3 = np.sum(dist >= 3)
    EPE1 = np.sum(dist >= 1)
    return np.mean(dist), (EPE1/(h * w)) * 100, (EPE3/(h * w)) * 100


def evaluateDataset():
    flows = os.listdir('../flows/com-defeito-comp')
    errors = []
    EPE1 = []
    EPE3 = []
    f = open('evaluation.txt', 'w')
    for flow in flows:
        flow1 = readFlow('../flows/sem-defeito-comp/{}'.format(flow))
        flow2 = readFlow('../flows/com-defeito-comp/{}'.format(flow))
        mean, epe1, epe3 = averageError(flow1, flow2)
        errors.append(mean)
        EPE1.append(epe1)
        EPE3.append(epe3)
        f.write('{} {} {} {}\n'.format(flow, mean, epe1, epe3))
    f.close()
    np_errors = np.array(errors, np.float64)
    np_epe1 = np.array(EPE1, np.float64)
    np_epe3 = np.array(EPE3, np.float64)

    print('average end-point error ')
    print('mean: {}'.format(np.mean(np_errors)))
    print('std: {}'.format(np.std(np_errors)))
    print('max: {}'.format(np.max(np_errors)))
    print('min: {}'.format(np.min(np_errors)))

    print('EPE1 ')
    print('mean: {}'.format(np.mean(np_epe1)))
    print('std: {}'.format(np.std(np_epe1)))
    print('max: {}'.format(np.max(np_epe1)))
    print('min: {}'.format(np.min(np_epe1)))

    print('EPE3 ')
    print('mean: {}'.format(np.mean(np_epe3)))
    print('std: {}'.format(np.std(np_epe3)))
    print('max: {}'.format(np.max(np_epe3)))
    print('min: {}'.format(np.min(np_epe3)))


if __name__ == '__main__':
    # f1 = readFlow('../flows/com-defeito-result/img3-9-4.flo')
    # f2 = readFlow('../flows/com-defeito-result/img4-7-9.flo')
    evaluateDataset()
