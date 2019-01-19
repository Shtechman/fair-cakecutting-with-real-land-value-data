#!python3
import json
import re
import urllib
from time import sleep
from urllib.parse import quote,unquote
from xml.etree import ElementTree

import requests


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

	""" create map of israel from list of neighborhood data """

	input_path = 'data/madlanDataDump/NeighDataWithCoordinates.json'
	output_path = 'data/madlanDataDump/IsrealMap.json'
	with open(input_path) as cities_data_file:
		neig_list = json.load(cities_data_file)

	israelMap = coor_to_list(neig_list, "areaPPM")
	with open(output_path,"w") as neigh_data_file:
		json.dump(israelMap, neigh_data_file)
	print("done")
