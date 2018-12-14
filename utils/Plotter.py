import math
import numpy as np
import matplotlib.pyplot as pyplot

from utils.Types import AggregationType


class Plotter:

    def plotResults(self, results, methods, xAxisDataType, yAxisData, title, experiments):
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

        dataPointsColor = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c', 'm', 'y',
                           'b', 'r', 'g', 'c', 'm', 'y', 'k', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c', 'm', 'y']
        dataPointsShape = ['s', 's', '*', '*', '*', '*', '*', '*', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x', 'x', 'x',
                           'v', 'v', 'v', 'v', 'v', 'v', '^', '^', '^', '^', '^', '^', 'D', 'D', 'D', 'D', 'D', 'D']
        idx = 0
        offsets = [-0.5+(x/len(methods)) for x in range(1,len(methods)+1)]
        for i,method in enumerate(methods):
            algResults = [result for result in results if result["Method"] == method]
            if len(algResults)>0:
                for dataset in yAxisData:
                    yValues = list(map(lambda result: result[dataset], algResults))
                    self.plotDataset(method + "-" + dataset, fignum, dataPointsShape[idx], dataPointsColor[idx], xValues,
                                yValues,offsets[i])
                    idx += 1

        pyplot.figure(fignum)
        pyplot.ylabel("gain (or loss)")
        pyplot.legend(loc='upper left')
        pyplot.show()

    def plotDataset(self, label, fignum, shape, color, xValues, yValues, offset):
        pyplot.figure(fignum)

        # """ Start Box Plot """
        # points = [(x, y) for x in xValues for y in yValues]
        # uniqueXValues = list(set(xValues))
        # yValuesByX = [[y for (x, y) in points if x == currX] for currX in uniqueXValues]
        # uniqueXValues = [x+offset for x in uniqueXValues]
        # pyplot.boxplot(yValuesByX, positions=uniqueXValues)
        # """ End Box Plot """
        #
        # """ Start Plot All Points """
        # pyplot.plot(xValues, yValues, color + shape, label=label)
        # """ End Plot All Points """

        """ Start Plot Avg Points """
        points = [(x, y) for x in xValues for y in yValues]
        uniqueXValues = list(set(xValues))
        yValuesByX = [np.average([y for (x, y) in points if x == currX]) for currX in uniqueXValues]
        pyplot.plot(uniqueXValues, yValuesByX, color + shape, label=label)
        """ End Plot Avg Points """

        # print("xv")
        # print(uniqueXValues)
        # print("yv")
        # print(yValuesByX)

        a, b = np.polyfit(xValues, yValues, 1)
        tlYValues = [a * x + b for x in [0] + xValues]

        pyplot.plot([0] + xValues, tlYValues, color + ':')
        print("%s line = %sx + %s" % (label, a, b))

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
