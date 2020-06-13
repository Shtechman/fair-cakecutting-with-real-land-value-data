#!python3
import csv
import json
import pickle
import re
from statistics import stdev

import numpy
from imageio import imread
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.patches import Circle
from shapely.geometry import Polygon
from descartes import PolygonPatch
from skimage.color import rgb2gray
from utils.simulation.simulation_environment import SimulationEnvironment as SimEnv
from utils.simulation.measurements import Measurements as Measure

import numpy as np

# import requests

from utils.simulation.agent import Agent
from utils.simulation.allocated_piece import AllocatedPiece
from utils.simulation.cc_types import AggregationType


def coor_to_list(coor_value_list, valueKey):
    cols = 1000
    rows = 1000
    westLine = 34.74
    eastLine = 34.85
    northLine = 32.15
    southLine = 32.04
    cellWidth = (eastLine - westLine) / cols
    cellHeight = (northLine - southLine) / rows
    tel_aviv_map = [[0 for _ in range(cols)] for _ in range(rows)]
    index_range = [x - 10 for x in range(21)]  # todo: calibrate "radius"

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
                tel_aviv_map[i][j] = int(entry[valueKey])

    return tel_aviv_map


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
    #exp 649 2020-05-28T23-05-58
    tmp = {'1014': 0.016516411983331186, '102': 0.020145682758285422, '132': 0.016784747713970025, '181': 0.019652427537338903, '182': 0.022546120531969159, '192': 0.021046343567998699, '200': 0.02355152298168306, '207': 0.023564094470375305, '215': 0.022424975477559995, '22': 0.016117401079521487, '228': 0.016715757895470237, '237': 0.021695844072901156, '266': 0.021194200032166249, '272': 0.018965854476892648, '276': 0.016430683417267469, '300': 0.021060685435338938, '319': 0.019769519931564623, '327': 0.018961219566701375, '357': 0.022014620456511595, '386': 0.02158860421612499, '389': 0.023140972870544872, '409': 0.016178009965255327, '437': 0.019937818454706359, '44': 0.018987245022697059, '452': 0.020650329931642182, '459': 0.022139048318260468, '47': 0.021753377067502053, '479': 0.02128103709532194, '505': 0.022577676114692146, '537': 0.016748099589210828, '547': 0.022739497213902699, '566': 0.019967759982472747, '588': 0.016638033007255321, '620': 0.023110117281450619, '630': 0.023615896471636318, '641': 0.023068557498541236, '672': 0.02139687435309676, '673': 0.016252856016096585, '678': 0.016359971381994548, '685': 0.016989847784585281, '695': 0.01996125295371666, '703': 0.019382309481196108, '715': 0.017114063598575022, '721': 0.016428618003536188, '746': 0.016559127966795695, '776': 0.020650329931642022, '797': 0.016441255249064132, '806': 0.021393461833462805, '810': 0.021707756029081378, '818': 0.021360505413733716, '838': 0.022047335143089302, '840': 0.019716984430077075, '843': 0.012006028102646166, '868': 0.019096238285795927, '872': 0.02275677429386181, '902': 0.025080439598393879, '925': 0.018626713240661931, '929': 0.018338568943304975, '931': 0.019325376706654199, '94': 0.023340315692814775, '957': 0.022195135278516603, '973': 0.019715698337032121, '979': 0.023122349615638711, '997': 0.021673721630185865}
    # with open('../data/testGdn06/index.txt', encoding="utf8") as in_file:
    #     index = json.load(in_file)
    #     maps_paths = index["mapsPaths"]
    #     old_dir = maps_paths[0][:maps_paths[0].rindex('/')+1]
    #     new_map_paths = []
    #     for map in [old_dir+k+'_valueMap' for k in tmp]:
    #         map_path = next((path for path in maps_paths if path.startswith(map)), None)
    #         new_map_paths.append(map_path.replace(old_dir,'./data/testGdn06/'))
    #     index["mapsPaths"] = new_map_paths
    # with open('../data/testGdn06/index.txt', 'w') as in_file:
    #     json.dump(index,in_file)

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

    # tlvNeibuf = imread('../data/madlanDataDump/tlvRealEstateMap.png')
    # tlvNeibuf = rgb2gray(tlvNeibuf)
    # tlvNeiMap = tlvNeibuf.tolist()
    output_path = '../data/tlvRealEstate06/0_valueMap_noise0.6_HS169_131.txt'
    # with open(output_path, 'wb') as tlv_re_file:
    #     pickle.dump(tlvNeiMap, tlv_re_file)
    cen = output_path.replace(".txt", "").split('_HS')[1].split("_")
    xy = (int(cen[1]),int(cen[0]))
    with open(output_path, 'rb') as tlv_re_file:
        a = pickle.load(tlv_re_file)
    ax = plt.gca()
    plt.imshow(a, cmap='hot', interpolation='nearest')
    ax.add_patch(Circle(xy, fc='b', ec='none'))
    plt.show()
    # """ create map of israel from list of neighborhood data """
    input_path = '../data/madlanDataDump/tlvGardensPolys.json'
    with open(input_path, 'rb') as gardens_data_file:
        raw_gardens_data = json.load(gardens_data_file)

    fig = plt.figure()
    axs = plt.gca()
    patch_list = []
    for feature in raw_gardens_data["features"]:
        name = feature["attributes"]["shem_gan"]
        area = feature["attributes"]["ms_area"]
        if feature["geometry"]["rings"]:
            coords = feature["geometry"]["rings"][0]
            gan_poly = Polygon(coords)
            xc, yc = gan_poly.centroid.xy
            # todo: get interesting data about the garden
            patch_list.append([gan_poly, area, xc[0], yc[0], name])
            # axs.text(xc[0],yc[0],name)

    max_ppm = max([gan[1] for gan in patch_list])
    [
        axs.add_patch(PolygonPatch(gan[0], fc=str(gan[1]), ec='none', alpha=1, zorder=0, label=gan[4]))
        for gan in patch_list
    ]
    axs.set_facecolor('black')
    axs.axis('scaled')
    plt.savefig('../data/madlanDataDump/tlvRealEstateMap.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    # fig.canvas.draw()
    # w, h = fig.canvas.get_width_height()
    # buf = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    # buf.shape = (w, h, 3)
    # buf = rgb2gray(buf)

    # input_path = '../data/madlanDataDump/tlvNeighPolys.json'
    # with open(input_path,'rb') as cities_data_file:
    #     raw_city_data = json.load(cities_data_file)
    # input_path = '../data/madlanDataDump/NeighDataWithCoordinates.json'
    # with open(input_path) as cities_data_file:
    #     neig_list = json.load(cities_data_file)
    # tlv_nei = [nei for nei in neig_list if "תל אביב" in nei["areaName"]]
    #
    # fig = plt.figure()
    # axs = plt.gca()
    # patch_list = []
    # for feature in raw_city_data["features"]:
    #     name = feature["attributes"]["shem_shchuna"].replace("-"," ")\
    #                                                 .replace("ככר","כיכר")\
    #                                                 .replace("נוה","נווה")\
    #                                                 .replace("תקוה","תקווה")\
    #                                                 .replace("לבנה,","לבנה ו")\
    #                                                 .replace("הצפון הישן   ","הצפון הישן ")\
    #                                                 .replace("הצפון הישן החלק הדרומי","הצפון הישן החלק המרכזי")\
    #                                                 .replace("הצפון החדש   ","הצפון החדש ")\
    #                                                 .replace("רמת אביב ג'","ר-א-ג")\
    #                                                 .replace("נאות אפקה א","נ-א-א")\
    #                                                 .replace("נאות אפקה ב","נ-א-ב")
    #     coords = feature["geometry"]["rings"][0]
    #     nei_poly = Polygon(coords)
    #     xc, yc = nei_poly.centroid.xy
    #     # xs, ys = nei_poly.exterior.xy
    #     # axs.fill(xs, ys, alpha=0.5, fc='r', ec='b', label=name)
    #     nei = next((nei for nei in tlv_nei if name in nei["areaName"].replace("-"," ").replace("נוה","נווה").replace("תקוה","תקווה").replace("רמת אביב ג","ר-א-ג")
    #                 or nei["areaName"].replace("תל אביב יפו/","").replace("-"," ").replace("תוכנית","תכנית").replace("נוה","נווה").replace("תקוה","תקווה").replace("נאות אפקה א","נ-א-א").replace("נאות אפקה ב","נ-א-ב") in name), None)
    #     if nei:
    #         ppm = max(0.0, float(nei["areaPPM"]))
    #     else:
    #         print('X - could not find ppm for', name)
    #         ppm = -1.0
    #     patch_list.append([nei_poly,ppm,xc[0],yc[0],name])
    #     # axs.text(xc[0],yc[0],name)
    #
    # max_ppm = max([nei[1] for nei in patch_list])
    #
    # for nei in patch_list:
    #     if nei[1] == -1:
    #         nei[1] = max_ppm/2
    # [
    # axs.add_patch(PolygonPatch(nei[0], fc=str(nei[1]/max_ppm), ec='none', alpha=1, zorder=0, label=nei[4]))
    #     for nei in patch_list
    # ]
    # axs.set_facecolor('black')
    # axs.axis('scaled')
    # plt.savefig('../data/madlanDataDump/tlvRealEstateMap.png', bbox_inches='tight', pad_inches=0)
    # fig.canvas.draw()
    # w, h = fig.canvas.get_width_height()
    # tlvNeibuf = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    # tlvNeibuf.shape = (h, w, 3)
    # tlvNeibuf = rgb2gray(tlvNeibuf)
    # tlvNeiMap = tlvNeibuf.tolist()
    # output_path = '../data/madlanDataDump/tlvRealEstateMap.txt'
    # with open(output_path,'wb') as tlv_re_file:
    #     pickle.dump(tlvNeiMap, tlv_re_file)
    # numpy.savetxt(output_path, tlvNeiMap, delimiter=",")



    input_path = '../data/madlanDataDump/NeighDataWithCoordinates.json'
    with open(input_path) as cities_data_file:
    	neig_list = json.load(cities_data_file)

    output_path = '../data/madlanDataDump/TLVMap.json'
    tlvMap = coor_to_list([nei for nei in neig_list if "תל אביב" in nei["areaName"]], "areaPPM")
    with open(output_path,"w") as neigh_data_file:
    	json.dump(tlvMap, neigh_data_file)
    print("done")
    input_path = '../data/madlanDataDump/TLVMap.json'
    with open(input_path,'rb') as mapfile:
    	a = json.load(mapfile)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()
    #
    #
    input_path = 'data/originalMaps/newzealand_forests_2D_low_res.txt'
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
