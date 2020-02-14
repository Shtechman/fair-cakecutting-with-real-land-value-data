import ast
import csv
import json
import os
import pickle
import random
import time
from math import exp

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from matplotlib import patches


class MapFileHandler:
    """/**
	* A class that represents an excel file containing a collection of value maps, a new sheet for each map.
	* the first sheet should only contain a list the names of the value maps in the following sheets.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/
	"""

    def __init__(self, path):
        """
		:param path to the excel file
		"""

        print("creating handler for file", path)
        self.path = path
        self.workbook = xlrd.open_workbook(self.path, on_demand=True)
        first_sheet = self.workbook.sheet_by_index(0)
        self.map_index = dict()
        for row in range(first_sheet.nrows):
            self.map_index[first_sheet.cell_value(row, 0)] = row + 1
        print("handler created.")

    def import_map(self, map_name):
        """
        >>>

        """
        if map_name not in self.map_index.keys():
            raise ValueError(
                "In given file there is no map named: " + str(map_name)
            )

        print("importing data for map", map_name)
        sheet = self.workbook.sheet_by_index(self.map_index[map_name])
        map = []
        for row in range(sheet.nrows):
            cuurent_row = sheet.row_values(row)
            map.append(
                [
                    x if (type(x) == float and x > 0) else 0.0
                    for x in cuurent_row
                ]
            )

        return map

    def read_all_maps(self):
        """
        >>>

        """

        try:
            self.maps
        except:
            self.maps = dict()
            for map_name in self.map_index.keys():
                self.maps[map_name] = self.import_map(map_name)
        return self.maps

    def retrieve_map(self, map_name):
        """
        >>>

        """

        try:
            self.maps
        except:
            self.read_all_maps()

        if map_name not in self.maps.keys():
            raise ValueError(
                "In given file there is no map named: " + str(map_name)
            )

        return self.maps[map_name]

    def save_maps_to_json(
        self, folder=".", prefix="", suffix="_2D", mapreduce=lambda m: m
    ):
        """
        >>>

        """
        self.read_all_maps()
        paths = []
        for map_name in self.maps.keys():
            file_path = folder + "\\" + prefix + map_name + suffix + ".txt"
            with open(file_path, "w") as json_file:
                json.dump(mapreduce(self.maps[map_name]), json_file)
            paths.append(file_path)
        return paths

    def save_maps_as_1D_horizontal_to_json(
        self, folder=".", prefix="", suffix=""
    ):
        """
        >>>

        """
        return self.save_maps_to_json(
            folder,
            prefix,
            suffix + "_1DHor",
            lambda map: np.array(map).sum(axis=0).tolist(),
        )

    def save_maps_as_1D_vertical_to_json(
        self, folder=".", prefix="", suffix=""
    ):
        """
        >>>

        """
        return self.save_maps_to_json(
            folder,
            prefix,
            suffix + "_1DVer",
            lambda map: np.array(map).sum(axis=1).tolist(),
        )

    @staticmethod
    def load_map_from_json(filepath):
        with open(filepath) as json_file:
            return json.load(json_file)


def hotspot_noise_function(
    original_map, noise_proportion, normalized_sum, max_value
):
    rows = len(original_map)
    cols = len(original_map[0])
    hotspot_center = (random.randint(0, rows), random.randint(0, cols))
    print(hotspot_center)

    def hotspot_noise(xj, yj):
        """ hotspot_noise = noise*exp(-((xj-xc)^2+(yj-yc)^2)^0.1) """
        dx = pow((hotspot_center[1] - xj), 2)
        dy = pow((hotspot_center[0] - yj), 2)

        noise_addition = noise_proportion * exp(-pow(dx + dy, 0.1))
        return noise_addition

    new_map = [
        [original_map[r][c] * (1 + hotspot_noise(c, r)) for c in range(cols)]
        for r in range(rows)
    ]

    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map


def nonzero_uniform_noise_function(
    original_map, noise_proportion, normalized_sum, max_value=1000000
):
    rows = len(original_map)
    cols = len(original_map[0])
    new_map = [
        [
            random.uniform(0, max_value) if original_map[r][c] > 0 else 0
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map


def uniform_noise_function(
    original_map, noise_proportion, normalized_sum, max_value
):
    rows = len(original_map)
    cols = len(original_map[0])

    """ Done to ensure noisy outcome value is not negative """
    neg_noise_proportion = max(-1, -noise_proportion)

    new_map = [
        [
            original_map[r][c]
            * (1 + random.uniform(neg_noise_proportion, noise_proportion))
            for c in range(cols)
        ]
        for r in range(rows)
    ]

    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map


def normalize_map(cols, new_map, normalized_sum, rows):
    if normalized_sum is not None and normalized_sum > 0:
        aggregated_sum = sum([sum(new_map[r]) for r in range(rows)])
        if aggregated_sum > 0:
            normalization_factor = normalized_sum / aggregated_sum
            new_map = [
                new_map[r][c] * normalization_factor
                for r in rows
                for c in cols
            ]
    return new_map


def random_values(rows, cols, max_value, normalized_sum):
    new_map = [
        [(random.uniform(0, max_value)) for _ in range(cols)]
        for _ in range(rows)
    ]
    if normalized_sum is not None and normalized_sum > 0:
        aggregated_sum = sum([sum(new_map[r]) for r in range(rows)])
        if aggregated_sum > 0:
            normalization_factor = normalized_sum / aggregated_sum
            new_map = [
                new_map[r][c] * normalization_factor
                for r in rows
                for c in cols
            ]
    return new_map


def generate_value_maps_to_file(
    original_map_file,
    folder,
    dataset_name,
    noise,
    num_of_maps,
    normalized_sum,
    noise_function,
    rows=1490,
    cols=1020,
):
    folder = folder + dataset_name

    if not os.path.exists(folder):
        os.mkdir(folder)

    random_maps = False
    if original_map_file is None:
        random_maps = True
        if noise is None:
            noise = 1000000
        print(
            "Creating %s random value maps to folder %s with max value %s"
            % (num_of_maps, folder, noise)
        )

    else:
        with open(original_map_file, "rb") as data_file:
            original_map_data = pickle.load(data_file)
            max_value = max([max(row) for row in original_map_data])
        print(
            "Creating %s value maps to folder %s with noise proportion %s"
            % (num_of_maps, folder, noise)
        )

    index_output_path = "%s/index.txt" % folder
    start_all = time.time()
    paths = []
    end = time.time()
    for i in range(num_of_maps):
        start = end
        output_path = "%s/%s_valueMap_noise%s.txt" % (folder, i, noise)
        paths.append(output_path)
        print("\tstart saving value maps to file %s" % output_path)
        with open(output_path, "wb") as object_file:
            if random_maps:
                new_map = random_values(rows, cols, noise, normalized_sum)
                if i == 0:
                    original_map_file = output_path
            else:
                new_map = noise_function(
                    original_map_data, noise, normalized_sum, max_value
                )
            pickle.dump(new_map, object_file)
        end = time.time()
        print("\t\tmap %s creation time was %s seconds" % (i, end - start))

    paths = [p.replace("..", ".") for p in paths]
    index = {
        "datasetName": dataset_name,
        "numOfMaps": num_of_maps,
        "folder": folder.replace("..", "."),
        "originalMapFile": original_map_file.replace("..", "."),
        "noise": noise,
        "mapsPaths": paths,
    }

    with open(index_output_path, "w") as index_file:
        json.dump(index, index_file)
    print("The whole creation process time was %s seconds" % (end - start_all))

    return index_output_path


def read_value_maps_from_file(map_path):
    print("\tFetching map file %s" % map_path)
    with open(map_path, "rb") as object_file:
        result = pickle.load(object_file)
    return result


def read_value_maps_from_csv(csv_file_path, idx):
    with open(csv_file_path, "r", newline="") as csv_file:
        csv_file_reader = csv.reader(csv_file)
        raw_file_data = []
        num_of_agents = None
        raw_map = []
        for line in csv_file_reader:
            if not num_of_agents:
                if (not line) or line[0].startswith("#"):
                    continue
                else:
                    num_of_agents = int(line[0])
                    if idx >= num_of_agents:
                        raise ValueError(
                            "searching for agent number %s "
                            "while file has only %s agents"
                            % (idx + 1, num_of_agents)
                        )
            elif (not line) or (not line[0].isdigit()):
                if not raw_map:
                    continue
                else:
                    raw_file_data.append(raw_map)
                    raw_map = []
            else:
                raw_map.append(line)

        if raw_map:
            raw_file_data.append(raw_map)
    return raw_file_data[idx][::-1]


def read_value_maps_from_files(index_path, num_of_maps):
    start_all = time.time()
    paths = get_value_maps_from_index(index_path, num_of_maps)
    result = []
    end = time.time()
    for i, map_path in enumerate(paths):
        start = end
        result.append(read_value_maps_from_file(map_path))
        end = time.time()
        print(
            "\t\tmap %s/%s load time was %s seconds"
            % (i + 1, num_of_maps, end - start)
        )
    print("The whole read process time was %s seconds" % (end - start_all))
    return result


def get_dataset_name_from_index(index_path):
    with open(index_path) as index_file:
        index = json.load(index_file)
    return index["datasetName"]


def get_value_maps_from_index(index_path, num_of_maps):
    print("Reading index file %s" % index_path)
    with open(index_path) as index_file:
        index = json.load(index_file)

    num_of_files = index["numOfMaps"]
    if num_of_files < num_of_maps:
        raise ValueError(
            "Not enough map files in folder (asked for %s but %s exist)"
            % (num_of_maps, num_of_files)
        )

    paths = random.sample(index["mapsPaths"], num_of_maps)
    return paths


def get_original_map_from_index(index_path):
    print("Reading index file %s" % index_path)
    with open(index_path) as index_file:
        index = json.load(index_file)
    return index["originalMapFile"]


def plot_partition_from_path(partition_path):
    partition_data = {}
    with open(partition_path) as partition_file:
        reader = csv.reader(partition_file)
        for row in reader:
            if row[0] == " ":
                continue
            if row[1][0] == "." or row[1][0].isalpha():
                partition_data[row[0]] = row[1]
            else:
                partition_data[row[0]] = ast.literal_eval(row[1])

    input_path = os.path.join(
        os.path.dirname(partition_data["Agent Files"][0]).replace("./", ""),
        "orig.txt",
    )
    with open(input_path, "rb") as mapfile:
        baseline_map = pickle.load(mapfile)

    plot_partition(baseline_map, partition_data["Partition"])


def plot_partition(baseline_map, partition_data):
    partition_plots = {}
    for part in partition_data:
        agent = part.split("(")[1].split(")")[0]
        partition_plots[agent] = ast.literal_eval(
            part.split("receives ")[1].split(" - ")[0]
        )
    fig, ax = plt.subplots(1)
    ax.imshow(baseline_map, cmap="hot")
    plt.axis([0, len(baseline_map[0]), 0, len(baseline_map)])
    for agent in partition_plots:
        part = partition_plots[agent]
        xy = (part[1], part[0])
        height = part[2] - part[0]
        width = part[3] - part[1]
        centerx = xy[0] + 0.5 / 2
        centery = xy[1] + 0.5 / 2
        # Create a Rectangle patch
        rect = patches.Rectangle(
            xy, width, height, linewidth=1, edgecolor="r", facecolor="none"
        )
        plt.text(centerx, centery, "Agent " + agent, color="red")
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    pass
