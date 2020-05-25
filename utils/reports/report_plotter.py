import itertools
import json
import math

import matplotlib.pyplot as pyplot
import numpy as np

from utils.simulation.cc_types import AggregationType, AlgType, CutPattern


class Plotter:
    def plot_results(
        self,
        results,
        methods,
        x_axis_data_type,
        y_axis_data,
        title,
        experiments,
    ):
        if x_axis_data_type == AggregationType.NumberOfAgents:
            x_values = list(
                map(
                    lambda result: math.log(result[x_axis_data_type.name], 2),
                    results,
                )
            )
            x_label = "log of number of agents"
        else:
            x_values = list(
                map(lambda result: result[x_axis_data_type.name], results)
            )
            x_label = x_axis_data_type.name

        new_x_values = []
        x_values = list(set(x_values))
        for i in range(experiments):
            new_x_values += x_values
        x_values = new_x_values

        fig_num = self.setup_plot_figure(x_label, max(x_values), title)

        data_points_color = [
            "r",
            "b",
            "g",
            "c",
            "m",
            "y",
            "k",
            "r",
            "g",
            "c",
            "m",
            "y",
            "b",
            "r",
            "g",
            "c",
            "m",
            "y",
            "b",
            "r",
            "g",
            "c",
            "m",
            "y",
            "k",
            "r",
            "g",
            "c",
            "m",
            "y",
            "b",
            "r",
            "g",
            "c",
            "m",
            "y",
        ]
        data_points_shape = [
            "s",
            "s",
            "*",
            "*",
            "*",
            "*",
            "*",
            "*",
            "o",
            "o",
            "o",
            "o",
            "x",
            "x",
            "x",
            "x",
            "x",
            "x",
            "v",
            "v",
            "v",
            "v",
            "v",
            "v",
            "^",
            "^",
            "^",
            "^",
            "^",
            "^",
            "D",
            "D",
            "D",
            "D",
            "D",
            "D",
        ]
        idx = 0
        offsets = [
            -0.5 + (x / len(methods)) for x in range(1, len(methods) + 1)
        ]
        for i, method in enumerate(methods):
            alg_results = [
                result for result in results if result["Method"] == method
            ]
            if len(alg_results) > 0:
                for dataset in y_axis_data:
                    y_values = list(
                        map(lambda result: result[dataset], alg_results)
                    )
                    self.plot_dataset(
                        method + "-" + dataset,
                        fig_num,
                        data_points_shape[idx],
                        data_points_color[idx],
                        x_values,
                        y_values,
                        offsets[i],
                    )
                    idx += 1

        pyplot.figure(fig_num)
        pyplot.ylabel("gain (or loss)")
        pyplot.legend(loc="upper left")
        pyplot.show()

    def plot_dataset(
        self, label, fig_num, shape, color, x_values, y_values, offset
    ):
        pyplot.figure(fig_num)

        # """ Start Box Plot """
        # points = [(x, y) for x in xValues for y in yValues]
        # unique_x_values = list(set(xValues))
        # y_values_by_x = [[y for (x, y) in points if x == curr_x] for curr_x in unique_x_values]
        # unique_x_values = [x+offset for x in unique_x_values]
        # pyplot.boxplot(y_values_by_x, positions=unique_x_values)
        # """ End Box Plot """
        #
        # """ Start Plot All Points """
        # pyplot.plot(xValues, yValues, color + shape, label=label)
        # """ End Plot All Points """

        """ Start Plot Avg Points """
        points = list(zip(x_values, y_values))
        unique_x_values = list(set(x_values))
        y_values_by_x = [
            np.average([y for (x, y) in points if x == curr_x])
            for curr_x in unique_x_values
        ]
        pyplot.plot(unique_x_values, y_values_by_x, color + shape, label=label)

        """ End Plot Avg Points """

        # print("xv")
        # print(unique_x_values)
        # print("yv")
        # print(y_values_by_x)

        a, b = np.polyfit(x_values, y_values, 1)
        tl_y_values = [a * x + b for x in [0] + x_values]

        pyplot.plot([0] + x_values, tl_y_values, color + ":")
        print(
            "%s | line = %s + %s |"
            % (label.ljust(45), (str(a) + "x").ljust(20), str(b).ljust(20)),
            list(zip(unique_x_values, y_values_by_x)),
        )

    def setup_plot_figure(self, x_label, max_x_values, title):
        fig, ax = pyplot.subplots()
        fig_num = fig.number
        pyplot.figure(fig_num)
        pyplot.xlabel(x_label)
        if type(max_x_values) is list:
            max_x_values = max(max_x_values)
        ax.hlines(y=0, linewidth=1, xmin=0, xmax=max_x_values + 0.5)
        pyplot.axis(xmin=0, xmax=max_x_values + 0.5)
        pyplot.title(title)
        return fig_num



if __name__ == "__main__":
    """ plot experiment results from json file """

    # jsonfilename = 'D:/MSc/Thesis/CakeCutting/results/2018-12-23T16-09-46_NoiseProportion_random_30_exp.json'
    # jsonfilename = 'D:/MSc/Thesis/CakeCutting/results/2018-12-24T03-45-40_NoiseProportion_0.2_30_exp.json'
    # jsonfilename = 'D:/MSc/Thesis/CakeCutting/results/2019-01-07T08-20-31_NoiseProportion_0.2_30_exp.json'
    jsonfilename = "D:/MSc/Thesis/CakeCutting/results/2019-02-10T10-13-22/IsraelMaps02_2019-02-10T20-29-15_NoiseProportion_0.2_50_exp.json"
    filepath_elements = jsonfilename.split("_")
    aggText = filepath_elements[2]
    aggParam = filepath_elements[3]
    experiments_per_cell = int(filepath_elements[4])
    dataParamType = AggregationType.NumberOfAgents
    with open(jsonfilename) as json_file:
        results = json.load(json_file)

    plotter = Plotter()
    # plotting
    plotter.plot_results(
        results,
        list(
            map(
                lambda pair: pair[0].name + pair[1].name,
                list(itertools.product(AlgType, CutPattern)),
            )
        ),
        x_axis_data_type=dataParamType,
        y_axis_data=["largestFaceRatio"],
        title="largestFaceRatio for " + aggText + " " + str(aggParam),
        experiments=experiments_per_cell,
    )
    plotter.plot_results(
        results,
        list(
            map(
                lambda pair: pair[0].name + pair[1].name,
                list(itertools.product(AlgType, CutPattern)),
            )
        ),
        x_axis_data_type=dataParamType,
        y_axis_data=["averageFaceRatio"],
        title="averageFaceRatio for " + aggText + " " + str(aggParam),
        experiments=experiments_per_cell,
    )
    plotter.plot_results(
        results,
        list(
            map(
                lambda pair: pair[0].name + pair[1].name,
                list(itertools.product(AlgType, CutPattern)),
            )
        ),
        x_axis_data_type=dataParamType,
        y_axis_data=["largestEnvy"],
        title="largestEnvy for " + aggText + " " + str(aggParam),
        experiments=experiments_per_cell,
    )
    plotter.plot_results(
        results,
        list(
            map(
                lambda pair: pair[0].name + pair[1].name,
                list(itertools.product(AlgType, CutPattern)),
            )
        ),
        x_axis_data_type=dataParamType,
        y_axis_data=["utilitarianGain"],
        title="utilitarianGain for " + aggText + " " + str(aggParam),
        experiments=experiments_per_cell,
    )
    plotter.plot_results(
        results,
        list(
            map(
                lambda pair: pair[0].name + pair[1].name,
                list(itertools.product(AlgType, CutPattern)),
            )
        ),
        x_axis_data_type=dataParamType,
        y_axis_data=["egalitarianGain"],
        title="egalitarianGain for " + aggText + " " + str(aggParam),
        experiments=experiments_per_cell,
    )
