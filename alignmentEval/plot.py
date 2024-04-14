import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def fileToMatrix(fileName):
    file = open(fileName, 'r')
    columns = []
    for line in file:
        ls = line.split(' ')
        # print(ls)
        row = [float(ls[1].strip()), float(ls[2].strip()), float(ls[3])]
        columns.append(row)
    return np.array(columns, np.float64)


def plotEvaluation( xlabel, ylabel, title, colum):
    eva = fileToMatrix('evaluation.txt')

    std = np.std(eva[..., colum])
    mean = np.mean(eva[..., colum])

    clean = eva[~(eva[:, colum] > mean+std*1.96)]

    print(clean.shape)

    plt.hist(clean[:, colum])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if colum > 0:
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()


if __name__ == '__main__':
    plotEvaluation('epe3', 'frequência', 'epe3 entre cada par de optical flow', 2)
