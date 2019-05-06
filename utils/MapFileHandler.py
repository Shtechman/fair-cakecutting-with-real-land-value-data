import csv
import os
import pickle
import random
import time
from math import exp

import numpy as np
import json

import xlrd



class MapFileHandler:
    """/**
	* A class that represents an excel file containing a collection of value maps, a new sheet for each map.
	* the first sheet should only contain a list the names of the value maps in the following sheets.
	*
	* @author Itay Shtechman
	* @since 2018-10
	*/"""

    def __init__(self, path):
        """
		:param path to the excel file
		"""

        print('creating handler for file',path)
        self.path = path
        self.workbook = xlrd.open_workbook(self.path, on_demand=True)
        firstSheet = self.workbook.sheet_by_index(0)
        self.mapIndex = dict()
        for row in range(firstSheet.nrows):
            self.mapIndex[firstSheet.cell_value(row, 0)] = row+1
        print('handler created.')

    def importMap(self, mapName):
        """
        >>>

        """
        if mapName not in self.mapIndex.keys():
            raise ValueError("In given file there is no map named: " + str(mapName))

        print('importing data for map', mapName)
        sheet = self.workbook.sheet_by_index(self.mapIndex[mapName])
        map = []
        for row in range(sheet.nrows):
            cuurentRow = sheet.row_values(row)
            map.append([x if (type(x) == float and x > 0) else 0.0 for x in cuurentRow])

        return map

    def readAllMaps(self):
        """
        >>>

        """

        try:
            self.maps
        except:
            self.maps = dict()
            for mapName in self.mapIndex.keys():
                self.maps[mapName] = self.importMap(mapName)
        return self.maps

    def retrieveMap(self, mapName):
        """
        >>>

        """

        try:
            self.maps
        except:
            self.readAllMaps()

        if mapName not in self.maps.keys():
            raise ValueError("In given file there is no map named: " + str(mapName))

        return self.maps[mapName]

    def saveMapsToJson(self, folder='.', prefix='', suffix='_2D', mapreduce = lambda m : m):
        """
        >>>

        """
        self.readAllMaps()
        paths = []
        for mapName in self.maps.keys():
            filePath = folder+"\\"+prefix+mapName+suffix+".txt"
            with open(filePath, 'w') as json_file:
                json.dump(mapreduce(self.maps[mapName]), json_file)
            paths.append(filePath)
        return paths

    def saveMapsAs1DHorizontalToJson(self, folder='.', prefix='', suffix=''):
        """
        >>>

        """
        return self.saveMapsToJson(folder,prefix,suffix+'_1DHor',lambda map : np.array(map).sum(axis=0).tolist())

    def saveMapsAs1DVerticalToJson(self, folder='.', prefix='', suffix=''):
        """
        >>>

        """
        return self.saveMapsToJson(folder,prefix,suffix+'_1DVer',lambda map : np.array(map).sum(axis=1).tolist())

    @staticmethod
    def mapFromJson(filepath):
        with open(filepath) as json_file:
            return json.load(json_file)


def hotspot_noise_function(original_map, noise_proportion, normalized_sum, max_value):
    rows = len(original_map)
    cols = len(original_map[0])
    hotspot_center = (random.randint(0, rows), random.randint(0, cols))
    print(hotspot_center)
    # noise*exp(-((xj-xc)^2+(yj-yc)^2)^0.1)
    def hotspot_noise(xj, yj):
        dx = pow((hotspot_center[1]-xj), 2)
        dy = pow((hotspot_center[0]-yj), 2)

        noise_addition = noise_proportion*exp(-pow(dx+dy, 0.1))
        return noise_addition

    new_map = [[original_map[r][c]*(1+hotspot_noise(c,r)) for c in range(cols)]
              for r in range(rows)]

    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map


def uniform_noise_function(original_map, noise_proportion, normalized_sum, max_value):
    rows = len(original_map)
    cols = len(original_map[0])
    neg_noise_proportion = max(-1, -noise_proportion)  # done to ensure noisy outcome value is not negative
    new_map = [[original_map[r][c]*(1+random.uniform(neg_noise_proportion, noise_proportion)) for c in range(cols)]
              for r in range(rows)]
    new_map = normalize_map(cols, new_map, normalized_sum, rows)
    return new_map


def normalize_map(cols, new_map, normalized_sum, rows):
    if normalized_sum is not None and normalized_sum > 0:
        aggregated_sum = sum([sum(new_map[r]) for r in range(rows)])
        if aggregated_sum > 0:
            normalization_factor = normalized_sum / aggregated_sum
            new_map = [new_map[r][c] * normalization_factor for r in rows for c in cols]
    return new_map




def randomValues(rows, cols, maxValue, normalized_sum):
    new_map = [[(random.uniform(0, maxValue)) for _ in range(cols)] for _ in range(rows)]
    if normalized_sum is not None and normalized_sum > 0:
        aggregated_sum = sum([sum(new_map[r]) for r in range(rows)])
        if aggregated_sum > 0:
            normalization_factor = normalized_sum / aggregated_sum
            new_map = [new_map[r][c] * normalization_factor for r in rows for c in cols]
    return new_map


def generate_valueMaps_to_file(original_map_file, folder, datasetName, noise, num_of_maps, normalized_sum, noise_function, rows=1490, cols=1020):
    folder = folder+datasetName

    if not os.path.exists(folder):
        os.mkdir(folder)

    randomMaps = False
    if original_map_file is None:
        randomMaps = True
        if noise is None:
            noise = 1000000
        print("Creating %s random value maps to folder %s with max value %s" % (num_of_maps, folder, noise))

    else:
        with open(original_map_file, "rb") as data_file:
            original_map_data = pickle.load(data_file)
            max_value = max([max(row) for row in original_map_data])
        print("Creating %s value maps to folder %s with noise proportion %s" % (num_of_maps, folder, noise))

    index_output_path = "%s/index.txt" % folder
    startAll = time.time()
    paths = []
    end = time.time()
    for i in range(num_of_maps):
        start = end
        output_path = "%s/%s_valueMap_noise%s.txt" % (folder, i, noise)
        paths.append(output_path)
        print("\tstart saving value maps to file %s" % output_path)
        with open(output_path, "wb") as object_file:
            if randomMaps:
                new_map = randomValues(rows, cols, noise, normalized_sum)
                if i == 0:
                    original_map_file = output_path
            else:
                new_map = noise_function(original_map_data, noise, normalized_sum, max_value)
            pickle.dump(new_map, object_file)
        end = time.time()
        print("\t\tmap %s creation time was %s seconds" % (i, end - start))

    paths = [p.replace('..','.') for p in paths]
    index = {"datasetName": datasetName,
             "numOfMaps" : num_of_maps,
             "folder" : folder.replace('..','.'),
             "originalMapFile" : original_map_file.replace('..','.'),
             "noise" : noise,
             "mapsPaths" : paths}

    with open(index_output_path, "w") as index_file:
        json.dump(index, index_file)
    print("The whole creation process time was %s seconds" % (end-startAll))

    return index_output_path


def read_valueMaps_from_file(map_path):
    print("\tfetching map file %s" % map_path)
    result = []
    with open(map_path, "rb") as object_file:
        result = pickle.load(object_file)
    return result


def read_valueMaps_from_files(index_path, num_of_maps):
    startAll = time.time()
    paths = get_valueMaps_from_index(index_path, num_of_maps)
    result = []
    end = time.time()
    for i, map_path in enumerate(paths):
        start = end
        result.append(read_valueMaps_from_file(map_path))
        end = time.time()
        print("\t\tmap %s/%s load time was %s seconds" % (i+1, num_of_maps, end - start))
    print("The whole read process time was %s seconds" % (end-startAll))
    return result


def get_datasetName_from_index(index_path):
    with open(index_path) as index_file:
        index = json.load(index_file)
    return index["datasetName"]


def get_valueMaps_from_index(index_path, num_of_maps):
    print("Reading index file %s" % index_path)
    with open(index_path) as index_file:
        index = json.load(index_file)
    numOfFiles = index["numOfMaps"]
    if numOfFiles < num_of_maps:
        raise ValueError("Not enough map files in folder (asked for %s but %s exist)" % (num_of_maps, numOfFiles))
    paths = random.sample(index["mapsPaths"], num_of_maps)
    return paths

def get_originalMap_from_index(index_path):
    print("Reading index file %s" % index_path)
    with open(index_path) as index_file:
        index = json.load(index_file)
    return index["originalMapFile"]



if __name__ == '__main__':
    nz1d = 'D:\MSc\Thesis\CakeCutting\data\\newzealand_forests_npv_4q.1d.json'
    nz2d = 'D:/MSc/Thesis/CakeCutting/data/newzealand_forests_2D.txt'
    nz1dHor = 'D:\MSc\Thesis\CakeCutting\data\\‏‏newzealand_forests_1DHor.txt'
    nz1dVer = 'D:\MSc\Thesis\CakeCutting\data\\‏‏newzealand_forests_1DVer.txt'

    noise = 0.2
    numOfAgents = 512

    og_maps = {"israel": '../data/originalMaps/IsraelMap.txt',
               "newzealand": '../data/originalMaps/newzealand_forests_2D_low_res.txt',
               "random":'../data/originalMaps/RandomNoiseMap.txt'}

    noise_methods = {"uniform": uniform_noise_function,
                     "hotspot": hotspot_noise_function}

    datasets = [
        {"datasetName": "newZealandLowResAgents06HS", "input_file": og_maps["newzealand"], "noise": 0.6, "numOfAgents": numOfAgents,
         "noise_method": noise_methods["hotspot"]},

        {"datasetName": "IsraelMaps06HS", "input_file": og_maps["israel"], "noise": 0.6, "numOfAgents": numOfAgents,
         "noise_method": noise_methods["hotspot"]},

        {"datasetName": "newZealandLowResAgents04HS", "input_file": og_maps["newzealand"], "noise": 0.4, "numOfAgents": numOfAgents,
         "noise_method": noise_methods["hotspot"]},

        {"datasetName": "IsraelMaps04HS", "input_file": og_maps["israel"], "noise": 0.4, "numOfAgents": numOfAgents,
         "noise_method": noise_methods["hotspot"]},

        {"datasetName": "newZealandLowResAgents02HS", "input_file": og_maps["newzealand"], "noise": 0.2, "numOfAgents": numOfAgents,
         "noise_method": noise_methods["hotspot"]},

        {"datasetName": "IsraelMaps02HS", "input_file": og_maps["israel"], "noise": 0.2, "numOfAgents": numOfAgents,
         "noise_method": noise_methods["hotspot"]},
    ]
    folder = "../data/"
    # folder = "data/test02"
    # input_file = TD2_MAP_2D_DATA_FILE_NAME
    for dataset in datasets:
        indexFile = generate_valueMaps_to_file(dataset["input_file"], folder, dataset["datasetName"],
                                               dataset["noise"], dataset["numOfAgents"], None, dataset["noise_method"])
        print(indexFile)

    # for input_file in ['../data/originalMaps/RandomNoiseMap.txt','../data/originalMaps/newzealand_forests_2D_low_res.txt','../data/originalMaps/IsraelMap.txt']:
    #     with open(input_file, "rb") as data_file:
    #         original_map_data = pickle.load(data_file)
    #     csvfilename = input_file+".csv"
    #     with open(csvfilename, "w", newline='') as csv_file:
    #         csv_file_writer = csv.writer(csv_file)
    #         for line in original_map_data:
    #             csv_file_writer.writerow(line)




