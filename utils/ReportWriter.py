import csv
import json
import os
from datetime import datetime
import numpy as np
from statistics import stdev
from utils.ReportGenerator import preprocess_results, EXTENDED_KEYS


def write_graph_method_per_measure_report_csv(jsonfilename, graph_method_results_per_measure):
    directory_path = os.path.dirname(jsonfilename)
    graph_report_path = os.path.join(directory_path, 'measurements_graphs')
    if not os.path.exists(graph_report_path):
        os.makedirs(graph_report_path)
    for measure in graph_method_results_per_measure:
        measure_results = graph_method_results_per_measure[measure]
        if not graph_method_results_per_measure[measure]:
            continue
        report_file_path = os.path.join(graph_report_path, measure+'.csv')
        with open(report_file_path, "w", newline='') as csv_file:
            csv_file_writer = csv.writer(csv_file)
            csv_file_writer.writerow([jsonfilename])
            table_header = ['NumberOfAgents', 'Honest', 'Assessor', 'Dishonest',  'Selling', 'Hon_Conf', 'As_Conf', 'Dis_Conf']
            for method in measure_results:
                if not measure_results[method]:
                    continue
                csv_file_writer.writerow([method])
                csv_file_writer.writerow(table_header)
                for method_entry in measure_results[method]:
                    if not method_entry:
                        continue
                    csv_file_writer.writerow(method_entry)
                csv_file_writer.writerow([""])
                csv_file_writer.writerow([""])
                csv_file_writer.writerow([""])


def write_dishonest_gain_results(jsonfilename, dishonest_gain):
    agg_dis_data = {}
    res_found = False
    for numOfAgent in dishonest_gain:
        agg_dis_data[numOfAgent] = {}
        for exp in dishonest_gain[numOfAgent].values():
            for cut_pattern in exp:
                if not exp[cut_pattern]:
                    continue
                else:
                    res_found = True
                try:
                    agg_dis_data[numOfAgent][cut_pattern]
                except:
                    agg_dis_data[numOfAgent][cut_pattern] = []
                exp[cut_pattern] = {"sgi": exp[cut_pattern],
                                    "sgAvg": np.average([sgi['agent_gain'] for sgi in exp[cut_pattern]]),
                                    "sgAvg_per": np.average([sgi['agent_gain_per'] for sgi in exp[cut_pattern]]),
                                    "sgMax": max([sgi['agent_gain'] for sgi in exp[cut_pattern]]),
                                    "sgMax_per": max([sgi['agent_gain_per'] for sgi in exp[cut_pattern]])}
                agg_dis_data[numOfAgent][cut_pattern].append(exp[cut_pattern])

        for cut_pattern in agg_dis_data[numOfAgent]:
            exp_list = agg_dis_data[numOfAgent][cut_pattern]
            exp_sgAvg = np.average([exp['sgAvg'] for exp in exp_list])
            exp_sgAvg_per = np.average([exp['sgAvg_per'] for exp in exp_list])
            exp_sgAvg_StDev = stdev([exp['sgAvg'] for exp in exp_list])
            exp_sgMax = np.average([exp['sgMax'] for exp in exp_list])
            exp_sgMax_per = np.average([exp['sgMax_per'] for exp in exp_list])
            exp_sgMax_StDev = stdev([exp['sgMax'] for exp in exp_list])
            agg_dis_data[numOfAgent][cut_pattern] = {'sgAvg': exp_sgAvg,
                                                     'sgAvg_per': exp_sgAvg_per,
                                                     'sgAvgStdev': exp_sgAvg_StDev,
                                                     'sgMax': exp_sgMax,
                                                     'sgMax_per': exp_sgMax_per,
                                                     'sgMaxStdev': exp_sgMax_StDev}

    # with open(in_path+'_sgRes', 'w') as json_file:
    # 	json.dump(agg_dis_data, json_file)
    if not res_found:
        return
    with open(jsonfilename + '_DishonestGain.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        for numOfAgent in agg_dis_data:
            csv_file_writer.writerow([numOfAgent, "agents"])
            csv_file_writer.writerow(
                ["Cut Pattern", "sgAvg", "sgAvg Improv(%)", "sgAvg stdev", "sgMax", "sgMax Improv(%)",
                 "sgMax stdev"])
            for cp in agg_dis_data[numOfAgent]:
                csv_file_writer.writerow([cp,
                                          agg_dis_data[numOfAgent][cp]['sgAvg'],
                                          agg_dis_data[numOfAgent][cp]['sgAvg_per'],
                                          agg_dis_data[numOfAgent][cp]['sgAvgStdev'],
                                          agg_dis_data[numOfAgent][cp]['sgMax'],
                                          agg_dis_data[numOfAgent][cp]['sgMax_per'],
                                          agg_dis_data[numOfAgent][cp]['sgMaxStdev']])
            csv_file_writer.writerow([])


def generate_reports(jsonfilename):
    # filepath_elements = jsonfilename.split('_')
    # aggText = filepath_elements[2]
    # aggParam = filepath_elements[3]
    # experiments_per_cell = int(filepath_elements[4])
    # dataParamType = AggregationType.NumberOfAgents
    with open(jsonfilename) as json_file:
        results = json.load(json_file)

    avg_results_per_method, \
    sum_honest_results_per_groupsize, \
    sum_dishonest_results_per_groupsize, \
    avg_honest_results_per_measurement,\
    avg_assessor_results_per_measurement, \
    avg_dishonest_results_per_measurement, \
    graph_method_results_per_measure, \
    groupsizes, \
    dishonest_gain, \
    bruteForce_gain = preprocess_results(results)

    write_graphtables_report_csv(jsonfilename, avg_honest_results_per_measurement, groupsizes)
    write_graphtables_report_csv(jsonfilename, avg_assessor_results_per_measurement, groupsizes, "Assessor")
    # write_graphtables_report_csv(jsonfilename, avg_dishonest_results_per_measurement, groupsizes, "Dishonest")

    write_summary_report_csv(jsonfilename, sum_honest_results_per_groupsize)
    # write_summary_report_csv(jsonfilename, sum_dishonest_results_per_groupsize, "Dishonest")

    write_graph_method_per_measure_report_csv(jsonfilename, graph_method_results_per_measure)

    write_extended_report_csv(jsonfilename, avg_results_per_method)

    write_dishonest_gain_results(jsonfilename, dishonest_gain)

    write_bruteforce_gain_results(jsonfilename, bruteForce_gain)


def write_bruteforce_gain_results(jsonfilename, bruteForce_gain):
    if not bruteForce_gain:
        return
    with open(jsonfilename + '_BruteForceGain.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        for groupsize in bruteForce_gain:
            csv_file_writer.writerow([groupsize, "agents"])
            csv_file_writer.writerow(["Measure", "Avg", "stdev"])
            for key_data in bruteForce_gain[groupsize]:
                csv_file_writer.writerow(key_data)
            csv_file_writer.writerow([])


def write_extended_report_csv(jsonfilename, avg_results_per_method):

    with open(jsonfilename + '_results.csv', "w", newline='') as csv_file:

        csv_file_writer = csv.writer(csv_file)
        keys_list = []
        csv_file_writer.writerow(keys_list)

        def _write_to_csv(keys, alg):
            r = avg_results_per_method[m][n][alg]
            if r:
                if not keys:
                    keys = list(r.keys())
                    csv_file_writer.writerow(keys)
                csv_file_writer.writerow([r[key] for key in keys])
            return keys

        for m in avg_results_per_method:
            for n in avg_results_per_method[m]:
                for key_report in EXTENDED_KEYS:
                    keys_list = _write_to_csv(keys_list, key_report)


def write_summary_report_csv(jsonfilename, sum_results_per_groupsize, label="Honest"):
    with open(jsonfilename + '_' + label + '_summary.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow([jsonfilename])
        first_headline = ["Cut Heuristic", "egalitarianGain", "", "", "ttc_egalitarianGain", "", "", "utilitarianGain",
                          "", "", "ttc_utilitarianGain", "", "", "averageFaceRatio", "", "", "smallestFaceRatio", "",
                          "", "largestEnvy", "", "", "ttc_largestEnvy", "", "", "runDuration(sec)", ""]
        second_headline = ["",
                           "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev",
                           "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev",
                           "AverageRatio", "Improv(%)", "StDev", "AverageRatio", "Improv(%)", "StDev",
                           "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev",
                           "Average Time", "StDev"]
        for groupSize in sum_results_per_groupsize:
            if not sum_results_per_groupsize[groupSize]:
                continue
            csv_file_writer.writerow([groupSize])
            csv_file_writer.writerow(first_headline)
            csv_file_writer.writerow(second_headline)
            for method_entry in sum_results_per_groupsize[groupSize]:
                if not method_entry:
                    continue
                csv_file_writer.writerow(method_entry)
            csv_file_writer.writerow([""])
            csv_file_writer.writerow([""])


def write_graphtables_report_csv(jsonfilename, results_per_measuement, groupsizes, label="Honest"):
    with open(jsonfilename + '_' + label + '_graphtables.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow([jsonfilename])
        headline = [""]+groupsizes
        for measure in results_per_measuement:
            if not results_per_measuement[measure]:
                continue
            csv_file_writer.writerow([label,measure])
            csv_file_writer.writerow(headline)
            for method in results_per_measuement[measure]:
                if not method:
                    continue
                if not results_per_measuement[measure][method]:
                    continue
                if not [x for x in results_per_measuement[measure][method] if x]:
                    continue
                csv_file_writer.writerow([method]+results_per_measuement[measure][method])
            csv_file_writer.writerow([""])
            csv_file_writer.writerow([""])


def create_exp_folder(run_folder, exp_name_string):
    result_folder = run_folder + exp_name_string + "/"
    result_log_folder = result_folder + "logs/"
    # result_ttc_folder = result_folder + "ttc/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        os.makedirs(result_log_folder)
        # os.makedirs(result_ttc_folder)
    return result_folder


def create_run_folder():
    run_folder = "./results/" + datetime.now().isoformat(timespec='seconds').replace(":", "-") + "/"
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    return run_folder


def generate_exp_name(aggParam, aggText, experiments_per_cell):
    timestring = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    file_name_string = timestring + "_" + aggText + "_" + str(aggParam) + "_" + str(experiments_per_cell) + "_exp"
    return file_name_string


def write_results_to_folder(result_folder, file_name_string, results):
    json_file_path = write_results_to_json(result_folder, file_name_string, results)
    generate_reports(json_file_path)
    write_results_to_csv(result_folder, file_name_string, results)


def write_results_to_csv(result_folder, file_name_string, results):
    csvfilename = result_folder + file_name_string + ".csv"
    with open(csvfilename, "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        keys_list = results[0].keys()
        data = [[result[key] for key in keys_list] for result in results]
        csv_file_writer.writerow(keys_list)
        for data_entry in data:
            csv_file_writer.writerow(data_entry)


def write_results_to_json(result_folder, file_name_string, results):
    jsonfilename = result_folder + file_name_string + ".json"
    with open(jsonfilename, "w") as json_file:
        json.dump(results, json_file)
    return jsonfilename


if __name__ == '__main__':

    """ generate report of experiment results from json file """


    files_to_import = [#'D:/MSc/Thesis/CakeCutting/results/luna/israelMaps02HS_results/IsraelMaps02HS_2019-05-05T15-16-06_NoiseProportion_0.2_15_exp.json',
                       #'D:/MSc/Thesis/CakeCutting/results/luna/israelMaps04HS_results/IsraelMaps04HS_2019-05-05T14-04-45_NoiseProportion_0.4_15_exp.json',
                       #'D:/MSc/Thesis/CakeCutting/results/luna/israelMaps06HS_results/IsraelMaps06HS_2019-05-05T12-54-11_NoiseProportion_0.6_15_exp.json',
                       #'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps02HS_results/newZealandLowResAgents02HS_2019-05-05T11-19-43_NoiseProportion_0.2_15_exp.json',
                       #'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps04HS_results/newZealandLowResAgents04HS_2019-05-05T09-46-34_NoiseProportion_0.4_15_exp.json',
                       #'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps06HS_results/newZealandLowResAgents06HS_2019-05-05T08-10-46_NoiseProportion_0.6_15_exp.json',
                       #'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps06_results_full/newZealandLowResAgents06_2019-03-29T07-50-19_NoiseProportion_0.6_50_exp.json',
                        'D:/MSc/Thesis/CakeCutting/results/2019-05-05T08-10-28/ttc_post_process_results for newZ HS06/ttc_post_process_results.json',
                        # 'D:/MSc/Thesis/CakeCutting/results/2019-08-27T19-45-25/newZealandLowResAgents06_2019-08-27T19-45-44_NoiseProportion_0.6_30_exp/newZealandLowResAgents06_2019-08-27T19-45-44_NoiseProportion_0.6_30_exp.json',
                        #'D:/MSc/Thesis/CakeCutting/results/2019-07-08T21-55-10/randomMaps06_2019-07-12T06-16-05_NoiseProportion_0.6_30_exp/randomMaps06_Dis_NoiseProportion_0.6_30_exp.json',
    ]

    for jsonfilename in files_to_import:
        generate_reports(jsonfilename)
