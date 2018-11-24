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


if __name__ == '__main__':
    nz1d = 'D:\MSc\Thesis\CakeCutting\data\\newzealand_forests_npv_4q.1d.json'
    nz2d = 'D:\MSc\Thesis\CakeCutting\data\\‏‏newzealand_forests_2D.txt'
    nz1dHor = 'D:\MSc\Thesis\CakeCutting\data\\‏‏newzealand_forests_1DHor.txt'
    nz1dVer = 'D:\MSc\Thesis\CakeCutting\data\\‏‏newzealand_forests_1DVer.txt'
    a = MapFileHandler.mapFromJson(nz1d)
    b = MapFileHandler.mapFromJson(nz2d)
    c = MapFileHandler.mapFromJson(nz1dHor)
    d = MapFileHandler.mapFromJson(nz1dVer)

    print([i for i, j in zip(a, d) if i != j])
    print([i for i, j in zip(a, b) if i != j])


