import csv
import json
import os
import pickle
import random
import time

import numpy as np
import xlrd




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


if __name__ == "__main__":
    pass
