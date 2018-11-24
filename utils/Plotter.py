import math
import numpy as np
import matplotlib.pyplot as pyplot

from utils.Types import AggregationType


class Plotter:

    def plotResults(self, results, algorithms, xAxisDataType, yAxisData, title, experiments):
        if xAxisDataType == AggregationType.NumberOfAgents:
            xValues = list(map(lambda result: math.log(result[xAxisDataType.name], 2), results))
            xLabel = "log of number of agents"
        else:
            xValues = list(map(lambda result: result[xAxisDataType.name], results))
            xLabel = xAxisDataType.name

        newXValues = []
        xValues = list(set(xValues))
        for i in range(experiments):
            newXValues += xValues
        xValues = newXValues

        fignum = self.setupPlotFigure(xLabel, max(xValues), title)

        dataPointsColor = ['b', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c', 'm', 'y',
                           'b', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c', 'm', 'y']
        dataPointsShape = ['*', '*', '*', '*', '*', '*', 'o', 'o', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x', 'x', 'x',
                           'v', 'v', 'v', 'v', 'v', 'v', '^', '^', '^', '^', '^', '^', 'D', 'D', 'D', 'D', 'D', 'D']
        idx = 0
        for algorithm in algorithms:
            algResults = [result for result in results if result["Algorithm"] == algorithm]
            for dataset in yAxisData:
                yValues = list(map(lambda result: result[dataset], algResults))
                self.plotDataset(algorithm + "-" + dataset, fignum, dataPointsShape[idx], dataPointsColor[idx], xValues,
                            yValues)
                idx += 1

        pyplot.figure(fignum)
        pyplot.ylabel("gain (or loss)")
        pyplot.legend(loc='upper left')
        pyplot.show()

    def plotDataset(self, label, fignum, shape, color, xValues, yValues):
        pyplot.figure(fignum)
        pyplot.plot(xValues, yValues, color + shape, label=label)

        a, b = np.polyfit(xValues, yValues, 1)
        tlYValues = [a * x + b for x in [0] + xValues]

        pyplot.plot([0] + xValues, tlYValues, color + ':')

    def setupPlotFigure(self, xLabel, maxXValues, title):
        fig, ax = pyplot.subplots()
        fignum = fig.number
        pyplot.figure(fignum)
        pyplot.xlabel(xLabel)
        if type(maxXValues) is list:
            maxXValues = max(maxXValues)
        ax.hlines(y=0, linewidth=1, xmin=0, xmax=maxXValues + 0.5)
        pyplot.axis(xmin=0, xmax=maxXValues + 0.5)
        pyplot.title(title)
        return fignum
