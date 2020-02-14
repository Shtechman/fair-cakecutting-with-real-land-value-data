#!python3
import csv
import json
import os
import pickle
import re
import sys
import urllib
from statistics import stdev
from time import sleep
from urllib.parse import quote, unquote
from xml.etree import ElementTree

from utils.simulation_environment import SimulationEnvironment as SimEnv
from utils.mapfile_handler import plot_partition_from_path
from utils.measurements import Measurements as Measure
import multiprocessing as mp
import ast

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# import requests

from utils.agent import Agent
from utils.allocated_piece import AllocatedPiece
from utils.top_trading_cycle import top_trading_cycles
from utils.types import AggregationType
from utils.simulation_log import SimulationLog


def coor_to_list(coor_value_list, valueKey):
    cols = 1000
    rows = 1150
    westLine = 34.2
    eastLine = 35.92
    northLine = 33.42
    southLine = 29.46
    cellWidth = (eastLine - westLine) / cols
    cellHeight = (northLine - southLine) / rows
    israel_map = [[0 for _ in range(cols)] for _ in range(rows)]
    index_range = [x - 10 for x in range(21)]

    for entry in coor_value_list:
        coor = entry["coordinate"]
        lat = float(coor[0])
        lng = float(coor[1])
        mid_i = int((lat - southLine) / cellHeight)
        mid_j = int((lng - westLine) / cellWidth)
        i_list = [min(max(x + mid_i, 0), rows - 1) for x in index_range]
        j_list = [min(max(x + mid_j, 0), cols - 1) for x in index_range]
        for i in i_list:
            for j in j_list:
                israel_map[i][j] = int(entry[valueKey])

    return israel_map


def measure_largest_envy(
    numberOfAgents, noiseProportion, method, experiment, partition
):
    largestEnvy = Measure.get_largest_envy(partition)
    if "Assessor" in method:
        algName = "Assessor"
        method = method.replace(algName, "")
    else:
        algName = "EvenPaz"
        method = method.replace(algName, "")
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


def readLogFile(cur_log_file):
    with open(cur_log_file) as csv_log_file:
        csv_reader = csv.reader(csv_log_file, delimiter=",")
        log_dict = {}
        for row in csv_reader:
            log_dict[row[0]] = row[1]
    numberOfAgents = int(log_dict["Number of Agents"])
    noise = log_dict["Noise"]
    method = log_dict["Method"]
    experiment = log_dict["Experiment"]
    cut_pattern = log_dict["Method"].split("_")[-1]
    alg_name = log_dict["Method"].split("_")[0]
    agent_mapfiles_list = (
        log_dict["Agent Files"]
        .replace("'", "")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .split(",")
    )
    cuts = (
        log_dict["Partition"]
        .replace("'", "")
        .replace("receives [", "$")
        .replace("] -", "$")
        .replace("[", "")
        .replace("]", "")
        .replace("Anonymous(", "#")
        .replace("Dishonest(", "#")
        .replace(") $", "# $")
    )
    partitions = log_dict["Partition"]
    return (
        numberOfAgents,
        noise,
        method,
        experiment,
        cut_pattern,
        alg_name,
        agent_mapfiles_list,
        cuts,
        partitions,
    )


def parseResults(result_to_parse):
    cur_log_file = result_to_parse[0]
    agent_list = result_to_parse[1]
    print("parsing", cur_log_file)
    (
        numberOfAgents,
        noise,
        method,
        experiment,
        cut_pattern,
        alg_name,
        agent_mapfiles_list,
        cuts,
        partitions,
    ) = readLogFile(cur_log_file)

    def _parsePartition(p):
        matchObj = re.match(r"#([^#]*)# \$([^\$]*)\$[^\(]* ", p, re.M | re.I)
        return matchObj.group(1), matchObj.group(2)

    def _getDishonestAgent(p_string):
        try:
            return (
                p_string.replace("'", "")
                .replace("receives [", "$")
                .replace("] -", "$")
                .replace("[", "")
                .replace("]", "")
                .replace("Dishonest(", "@@@")
                .replace(") $", "# $")
                .split("@@@")[1]
                .split("#")[0]
            )
        except:
            return None

    dishonest_agent = _getDishonestAgent(partitions)

    cuts_list = [_parsePartition(p) for p in cuts.split("), ")]

    agent_piece_list = []
    for p in cuts_list:
        for agent in agent_list:
            if p[0] in agent.get_map_file_number():
                agent_piece_list = agent_piece_list + [[agent, p[1]]]

    def _allocatePiece(agent_piece):
        indexes = [float(i) for i in agent_piece[1].split(",")]
        return AllocatedPiece(
            agent_piece[0], indexes[0], indexes[1], indexes[2], indexes[3]
        )

    partition = list(map(_allocatePiece, agent_piece_list))

    valueGains = {
        ap.get_agent().get_map_file_number(): ap.get_relative_value()
        for ap in partition
    }

    return {
        AggregationType.NumberOfAgents.name: numberOfAgents,
        AggregationType.NoiseProportion.name: noise,
        "Algorithm": alg_name,
        "Method": method,
        "experiment": experiment,
        "cut_pattern": cut_pattern,
        "dishonest": dishonest_agent,
        "partition": valueGains,
    }


def _get_piece_of_agent(partition, agent_num):
    piece = [
        p
        for p in partition
        if p.get_agent().get_map_file_number() == agent_num
    ][0]
    return piece


def _get_dishonest_gain(
    honest_partition, dishonest_partition, dishonest_agent
):
    dis_value = dishonest_partition[dishonest_agent]
    hon_value = honest_partition[dishonest_agent]

    return dis_value - hon_value, (dis_value - hon_value) / hon_value


def get_dishonest_gain(rlogs):
    experiment_list = list(set([rlog["experiment"] for rlog in rlogs]))
    cut_pattern_list = list(set([rlog["cut_pattern"] for rlog in rlogs]))
    num_agents = list(
        set([rlog[AggregationType.NumberOfAgents.name] for rlog in rlogs])
    )

    result = {numA: {} for numA in num_agents}

    for exp in experiment_list:
        numA = [rlog for rlog in rlogs if rlog["experiment"] == exp][0][
            AggregationType.NumberOfAgents.name
        ]
        result[numA][exp] = {}
        for cp in cut_pattern_list:
            result[numA][exp][cp] = []
            relevant_logs = [
                rlog
                for rlog in rlogs
                if (rlog["experiment"] == exp and rlog["cut_pattern"] == cp)
            ]
            dishonest_partitions = {
                rlog["dishonest"]: rlog["partition"]
                for rlog in relevant_logs
                if rlog["dishonest"] is not None
            }
            honest = [
                rlog for rlog in relevant_logs if rlog["dishonest"] is None
            ][0]
            honest_partitions = honest["partition"]
            for agent in dishonest_partitions:
                v, p = _get_dishonest_gain(
                    honest_partitions, dishonest_partitions[agent], agent
                )

                result[numA][exp][cp].append(
                    {"agent": agent, "agent_gain": v, "agent_gain_per": p}
                )
    return result


def _write_dishonest_results_to_csv(dis_data, csv_path):
    agg_dis_data = {}
    for numOfAgent in dis_data:
        agg_dis_data[numOfAgent] = {}
        for exp in dis_data[numOfAgent].values():
            for cut_pattern in exp:
                try:
                    agg_dis_data[numOfAgent][cut_pattern]
                except:
                    agg_dis_data[numOfAgent][cut_pattern] = []
                exp[cut_pattern] = {
                    "sgi": exp[cut_pattern],
                    "sgAvg": np.average(
                        [sgi["agent_gain"] for sgi in exp[cut_pattern]]
                    ),
                    "sgAvg_per": np.average(
                        [sgi["agent_gain_per"] for sgi in exp[cut_pattern]]
                    ),
                    "sgMax": max(
                        [sgi["agent_gain"] for sgi in exp[cut_pattern]]
                    ),
                    "sgMax_per": max(
                        [sgi["agent_gain_per"] for sgi in exp[cut_pattern]]
                    ),
                }
                agg_dis_data[numOfAgent][cut_pattern].append(exp[cut_pattern])

        for cut_pattern in agg_dis_data[numOfAgent]:
            exp_list = agg_dis_data[numOfAgent][cut_pattern]
            exp_sgAvg = np.average([exp["sgAvg"] for exp in exp_list])
            exp_sgAvg_per = np.average([exp["sgAvg_per"] for exp in exp_list])
            exp_sgAvg_StDev = stdev([exp["sgAvg"] for exp in exp_list])
            exp_sgMax = np.average([exp["sgMax"] for exp in exp_list])
            exp_sgMax_per = np.average([exp["sgMax_per"] for exp in exp_list])
            exp_sgMax_StDev = stdev([exp["sgMax"] for exp in exp_list])
            agg_dis_data[numOfAgent][cut_pattern] = {
                "sgAvg": exp_sgAvg,
                "sgAvg_per": exp_sgAvg_per,
                "sgAvgStdev": exp_sgAvg_StDev,
                "sgMax": exp_sgMax,
                "sgMax_per": exp_sgMax_per,
                "sgMaxStdev": exp_sgMax_StDev,
            }

    # with open(in_path+'_sgRes', 'w') as json_file:
    # 	json.dump(agg_dis_data, json_file)

    with open(csv_path, "w", newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file)
        for numOfAgent in agg_dis_data:
            csv_file_writer.writerow([numOfAgent, "agents"])
            csv_file_writer.writerow(
                [
                    "Cut Pattern",
                    "sgAvg",
                    "sgAvg Improv(%)",
                    "sgAvg stdev",
                    "sgMax",
                    "sgMax Improv(%)",
                    "sgMax stdev",
                ]
            )
            for cp in agg_dis_data[numOfAgent]:
                csv_file_writer.writerow(
                    [
                        cp,
                        agg_dis_data[numOfAgent][cp]["sgAvg"],
                        agg_dis_data[numOfAgent][cp]["sgAvg_per"],
                        agg_dis_data[numOfAgent][cp]["sgAvgStdev"],
                        agg_dis_data[numOfAgent][cp]["sgMax"],
                        agg_dis_data[numOfAgent][cp]["sgMax_per"],
                        agg_dis_data[numOfAgent][cp]["sgMaxStdev"],
                    ]
                )
            csv_file_writer.writerow([])


def get_agents_for_exp(log_file):
    _, _, _, _, _, _, agent_mapfiles_list, _, _ = readLogFile(log_file)
    return list(map(Agent, agent_mapfiles_list))


def get_TTC_results_from_log(agents, log):
    SE = SimEnv(
        log.iSimulation,
        log.noiseProportion,
        agents,
        [],
        [],
        log.result_folder,
        [],
    )
    org_alloc = log.recreate_allocation()
    result = SE.parse_results_from_partition(
        log.algName, log.method, org_alloc, log.run_duration, "", False
    )
    for p in org_alloc:
        p.clear()
        del p
    del SE
    return result


if __name__ == "__main__":

    # in_path = 'results/2019-08-27T19-45-25/IsraelMaps06_2019-08-29T10-08-22_NoiseProportion_0.6_30_exp/dishonest_data.json'
    # with open(in_path, encoding="utf8") as in_file:
    #     dis_data = json.load(in_file)
    #
    # _write_dishonest_results_to_csv(dis_data, in_path + "dis_data.csv")
    #
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

    # plot_partition_from_path('results/luna/newZealandMaps06_results_full/logs/1281_EvenPazSquarePiece.csv')

    # input_path = 'data/newZealand_nonZuniform/1_valueMap_noise0.6.txt'
    # with open(input_path, 'rb') as mapfile:
    # 	a = pickle.load(mapfile)
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show()
    # input_path = 'data/IsraelMaps02HS/1_valueMap_noise0.2.txt'
    # with open(input_path, 'rb') as mapfile:
    # 	a = pickle.load(mapfile)
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show()
    # input_path = 'data/IsraelMaps06HS/0_valueMap_noise0.6.txt'
    # with open(input_path, 'rb') as mapfile:
    # 	a = pickle.load(mapfile)
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show()

    # NTASKS = 4

    # folders_list = ['results/2019-02-10T10-13-22/IsraelMaps02_2019-02-10T20-29-15_NoiseProportion_0.2_50_exp',
    # 				'results/2019-02-10T10-13-22/newZealandLowResAgents02_2019-02-10T15-04-23_NoiseProportion_0.2_50_exp',
    # 				'results/2019-02-10T10-13-22/randomMaps02_2019-02-10T10-13-40_NoiseProportion_0.2_50_exp']
    # folders_list = ['results/2019-09-01T21-16-33/IsraelMaps02_2019-09-01T21-16-37_NoiseProportion_0.2_2_exp']
    # for input_path in folders_list:
    #     log_folder = input_path + "/logs/"
    #
    #     # cur_log_file = log_folder+"41_AssessorHighestScatter.csv"
    #     results = []
    #     log_file_list = os.listdir(log_folder)
    #     log_exp_list = list(set([log_name.split('_')[0] for log_name in log_file_list]))
    #
    #     print("Sort logs to experiments...")
    #     log_list_per_exp = {
    #     exp: [os.path.join(log_folder, log_file) for log_file in log_file_list if exp == log_file.split('_')[0]]
    #     for exp in log_exp_list}
    #     rlogs = []
    #     for exp in log_list_per_exp:
    #         print("parsing logs in experiment %s" % exp)
    #         if len(log_list_per_exp[exp]) < 1:
    #             continue
    #
    #         agents = get_agents_for_exp(log_list_per_exp[exp][0])
    #         results_to_parse = [(log, agents) for log in log_list_per_exp[exp]]
    #         p = mp.Pool(NTASKS)
    #         rlogs = rlogs + p.map(parseResults, results_to_parse)
    #         p.close()
    #         p.join()
    #
    #         del p
    #
    #     results = get_dishonest_gain(rlogs)
    #
    #     data_output = input_path + '/dishonest_data.json'
    #     csv_output = input_path + '/dishonest_data_summary.csv'
    #
    #     with open(data_output, 'w') as json_file:
    #         json.dump(results, json_file)
    #
    #     _write_dishonest_results_to_csv(results, csv_output)

    # exp_folders = [
    #     # 'results/2019-08-27T18-35-00/IsraelMaps02_2019-08-27T18-35-05_NoiseProportion_0.2_2_exp/',
    #     # 'results/2019-05-03T15-41-32/newZealandLowResAgents06_2019-05-03T15-41-52_NoiseProportion_0.6_50_exp/',
    #     # 'results/2019-08-27T19-45-25/newZealandLowResAgents06_2019-08-27T19-45-44_NoiseProportion_0.6_30_exp',
    #     # 'results/2019-08-27T19-45-25/randomMaps06_2019-08-31T00-11-38_NoiseProportion_0.6_30_exp',
    #     'results/2019-05-05T08-10-28/newZealandLowResAgents06HS_2019-05-05T08-10-46_NoiseProportion_0.6_15_exp/',
    # ]
    #
    # for exp_folder in exp_folders:
    #     log_folder = exp_folder + 'logs/'
    #     ttc_out_folder = exp_folder + 'ttc_post_process_results/'
    #     logs = SimulationLog.create_logs_from_csv_folder(log_folder)
    #     results = []
    #     for idx, sim_log in enumerate(logs):
    #         print("=== %s/%s ===" % (idx, len(logs)))
    #         agents = logs[sim_log][0].recreate_agent_list()
    #         results += [
    #             get_TTC_results_from_log(agents, log)
    #             for log in logs[sim_log] if "Dishonest" not in log.method
    #         ]
    #         for agent in agents:
    #             agent.cleanMemory()
    #             del agent
    #     if not os.path.exists(ttc_out_folder):
    #         os.makedirs(ttc_out_folder)
    #     write_results_to_folder(ttc_out_folder, "ttc_post_process_results", results)

    # exp_folder = 'results/EPnLD/combined_results.json'
    #
    # files = [os.path.join(exp_folder, file_name) for file_name in os.listdir(exp_folder)]
    #
    # results = []
    # for file in files:
    #     with open(file) as json_file:
    #         results += json.load(json_file)
    #
    # output_file = os.path.join(exp_folder, "combined_results.json")
    #
    # with open(output_file, "w") as json_file:
    #     json.dump(results, json_file)

    print("all done")
