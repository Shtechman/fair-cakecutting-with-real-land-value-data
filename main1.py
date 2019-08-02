#!python3
import csv
import json
import os
import pickle
import re
import sys
import urllib
from time import sleep
from urllib.parse import quote,unquote
from xml.etree import ElementTree

from utils.MapFileHandler import plot_partition_from_path
from utils.Measurements import Measurements as Measure
import multiprocessing as mp
import ast


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
#import requests

from utils.Agent import Agent
from utils.AllocatedPiece import AllocatedPiece
from utils.ReportGenerator import write_results_to_folder
from utils.Types import AggregationType


def coor_to_list(coor_value_list, valueKey):
	cols = 1000
	rows = 1150
	westLine = 34.2
	eastLine = 35.92
	northLine = 33.42
	southLine = 29.46
	cellWidth = (eastLine-westLine)/cols
	cellHeight = (northLine-southLine)/rows
	israel_map = [[0 for _ in range(cols)] for _ in range(rows)]
	index_range = [x-10 for x in range(21)]

	for entry in coor_value_list:
		coor = entry["coordinate"]
		lat = float(coor[0])
		lng = float(coor[1])
		mid_i = int((lat-southLine)/cellHeight)
		mid_j = int((lng-westLine)/cellWidth)
		i_list = [min(max(x+mid_i, 0), rows-1) for x in index_range]
		j_list = [min(max(x+mid_j, 0), cols-1) for x in index_range]
		for i in i_list:
			for j in j_list:
				israel_map[i][j] = int(entry[valueKey])

	return israel_map


def measure_largest_envy(numberOfAgents,noiseProportion,method,experiment,partition):
	largestEnvy = Measure.calculateLargestEnvy(partition)
	if 'Assessor' in method:
		algName = 'Assessor'
		method = method.replace(algName,'')
	else:
		algName = 'EvenPaz'
		method = method.replace(algName, '')
	return {
		AggregationType.NumberOfAgents.name: numberOfAgents,
		AggregationType.NoiseProportion.name: noiseProportion,
		"Algorithm": algName,
		"Method": method,
		"egalitarianGain": 0,
		"utilitarianGain": 0,
		"averageFaceRatio": 0,
		"largestFaceRatio": 0,
		"smallestFaceRatio": 0,
		"averageInheritanceGain": 0,
		"largestInheritanceGain": 0,
		"largestEnvy": largestEnvy,
		"experimentDurationSec": 0,
		"experiment": experiment,
	}


def parseResults(cur_log_file):
	print("parsing", cur_log_file)
	with open(cur_log_file) as csv_log_file:
		csv_reader = csv.reader(csv_log_file, delimiter=',')
		log_dict = {}
		for row in csv_reader:
			log_dict[row[0]] = row[1]
	#	print(log_dict)
	numberOfAgents = int(log_dict['Number of Agents'])
	noise = log_dict['Noise']
	method = log_dict['Method']
	experiment = log_dict['Experiment']
	agent_mapfiles_list = log_dict['Agent Files'].replace('\'','').replace('[','').replace(']','').replace(' ','').split(',')
	cuts = log_dict['Partition'].replace('\'', '').replace('receives [', '$').replace('] -', '$').replace('[', '').replace(']', '').replace('Anonymous(', '#').replace(') $', '# $')

	def _parsePartition(p):
		matchObj = re.match(r'#([^#]*)# \$([^\$]*)\$[^\(]* ', p, re.M | re.I)
		return matchObj.group(1), matchObj.group(2)

	cuts_list = [_parsePartition(p) for p in cuts.split('), ')]
	agents = list(map(Agent, agent_mapfiles_list))

	agent_piece_list = []
	for p in cuts_list:
		for agent in agents:
			if p[0] in agent.file_num:
				agent_piece_list = agent_piece_list + [[agent, p[1]]]

	def _allocatePiece(agent_piece):
		indexes = [float(i) for i in agent_piece[1].split(',')]
		return AllocatedPiece(agent_piece[0],indexes[0],indexes[1],indexes[2],indexes[3])

	partition = list(map(_allocatePiece,agent_piece_list))
	return measure_largest_envy(numberOfAgents, noise, method, experiment, partition)





if __name__ == '__main__':

	# index_path = 'data/madlanDataDump/wholeIsraelIndex.json'
	# output_path = 'data/madlanDataDump/wholeIsraelIdsList.json'
	# with open(index_path, encoding="utf8") as index_file:
	#     index = json.load(index_file)
	# with open(output_path, 'w') as json_file:
	#     json.dump([item["id"].decode("utf8") for item in index["heatmap"]["polys"]], json_file)

	# index_path = 'data/madlanDataDump/wholeIsraelIdsList.json'
	# output_path = 'data/madlanDataDump/CitiesData.json'
	# with open(index_path) as index_file:
	# 	cities = json.load(index_file)
	#
	# cityURLList = [
	# 	"https://s3-eu-west-1.amazonaws.com/static.madlan.co.il/widgets/ynetHPIWidget/1521504000000/%s.json" % quote(city)
	# 	for city in cities]
	#
	# cityData = {}
	#
	# for i, (city, cityURL) in enumerate(zip(cities, cityURLList)):
	# 	print("fetching data %s/%s about %s" % (i+1, len(cities), city))
	#
	# 	cityData[city] = json.loads(requests.get(cityURL).content.decode('utf-8')[2:-1])
	#
	# with open(output_path, 'w') as json_file:
	# 	json.dump(cityData, json_file)




	# neig_list = []
	# for city in cities_data:
	# 	city_data = cities_data[city]
	# 	if 'heatmap' in city_data:
	# 		city_neig_data = city_data['heatmap']['polys']
	# 		for city_neig in city_neig_data:
	# 			if 'id' in city_neig:
	# 				areaName = city_neig['id']
	# 				popupRawData = city_neig['popupContent'].split('</div>')
	# 				for line in popupRawData:
	# 					try:
	# 						found = re.search('"bold">(.+?)</span>', line).group(1)
	# 						if "מדד מדלן" in line:
	# 							areaPI = int(found.replace(",",""))
	# 						if "מדד למ\"ר" in line:
	# 							areaPPM = int(found.replace(",",""))
	# 					except AttributeError:
	# 						pass
	# 				neig_list.append({"areaName":areaName,
	# 								  "areaPI":areaPI,
	# 								  "areaPPM":areaPPM})
	# 	else:
	# 		neig_list.append({"areaName":city,
	# 						  "areaPI":city_data['priceIndexes']['priceIndex'],
	# 						  "areaPPM":city_data['priceIndexes']['PPMIndex']})

	# for i, neig in enumerate(neig_list):
	# 	if "coordinate" not in neig:
	# 		searchedArea = neig["areaName"].replace("/",",").replace(" ","+")
	# 		print("Searching for %s/%s location of %s" % (i+1,len(neig_list), searchedArea))
	# 		ans = requests.get("https://www.google.com/maps/search/%s" % searchedArea).content.decode('utf-8')
	# 		# sleep(30)
	# 		try:
	# 			regex = re.search('@3(.+?)/', ans)
	# 			if regex is not None:
	# 				print("found")
	# 				found = regex.group(1)
	# 				coordinate = "3" + found
	# 				coordinate = coordinate.split(",")
	# 				lat = coordinate[0]
	# 				lng = coordinate[1]
	# 				neig["coordinate"] = [lat, lng]
	# 			# else:
	# 				# regex = re.search('robot', ans)
	# 				# if regex is not None:
	# 				# 	break
	# 		except AttributeError as e:
	# 			s = str(e)
	# 			print("error searching for %s !" % searchedArea)
	# 			break
	#
	# coor_neig_list = [neig for neig in neig_list if "coordinate" in neig]

	# """ create map of israel from list of neighborhood data """

	# input_path = 'data/madlanDataDump/NeighDataWithCoordinates.json'
	# output_path = 'data/madlanDataDump/IsrealMap.json'
	# with open(input_path) as cities_data_file:
	# 	neig_list = json.load(cities_data_file)

	# israelMap = coor_to_list(neig_list, "areaPPM")
	# with open(output_path,"w") as neigh_data_file:
	# 	json.dump(israelMap, neigh_data_file)
	# print("done")
	# input_path = 'data/originalMaps/IsraelMap.txt'
	# with open(input_path,'rb') as mapfile:
	# 	a = pickle.load(mapfile)
	# plt.imshow(a, cmap='hot', interpolation='nearest')
	# plt.show()
	#
	#
	# input_path = 'data/originalMaps/newzealand_forests_2D_low_res.txt'
	# with open(input_path,'rb') as mapfile:
	# 	a = pickle.load(mapfile)
	# plt.imshow(a, cmap='hot', interpolation='nearest')
	# plt.show()
	#
	#

	plot_partition_from_path('results/luna/newZealandMaps06_results_full/logs/1281_EvenPazSquarePiece.csv')

	input_path = 'data/IsraelMaps02HS/0_valueMap_noise0.2.txt'
	with open(input_path, 'rb') as mapfile:
		a = pickle.load(mapfile)
	plt.imshow(a, cmap='hot', interpolation='nearest')
	plt.show()
	input_path = 'data/IsraelMaps02HS/1_valueMap_noise0.2.txt'
	with open(input_path, 'rb') as mapfile:
		a = pickle.load(mapfile)
	plt.imshow(a, cmap='hot', interpolation='nearest')
	plt.show()
	input_path = 'data/IsraelMaps06HS/0_valueMap_noise0.6.txt'
	with open(input_path, 'rb') as mapfile:
		a = pickle.load(mapfile)
	plt.imshow(a, cmap='hot', interpolation='nearest')
	plt.show()

	NTASKS = 10

	# folders_list = ['results/2019-02-10T10-13-22/IsraelMaps02_2019-02-10T20-29-15_NoiseProportion_0.2_50_exp',
	# 				'results/2019-02-10T10-13-22/newZealandLowResAgents02_2019-02-10T15-04-23_NoiseProportion_0.2_50_exp',
	# 				'results/2019-02-10T10-13-22/randomMaps02_2019-02-10T10-13-40_NoiseProportion_0.2_50_exp']
	folders_list = ['results/2019-02-19T15-40-36/IsraelMaps04_2019-02-19T15-40-48_NoiseProportion_0.4_50_exp',
					'results/2019-02-19T15-40-36/IsraelMaps06_2019-02-19T20-04-52_NoiseProportion_0.6_50_exp']
	for input_path in folders_list:
		log_folder = input_path+"/logs/"

		# cur_log_file = log_folder+"41_AssessorHighestScatter.csv"
		results = []
		log_file_list = os.listdir(log_folder)
		log_file_list = [os.path.join(log_folder, log_file) for log_file in log_file_list]

		p = mp.Pool(NTASKS)

		results = p.map(parseResults, log_file_list)
		p.close()
		p.join()

		del p

		write_results_to_folder(input_path+'/', "LargestEnvyCalculationFix", results)
	print("all done")
