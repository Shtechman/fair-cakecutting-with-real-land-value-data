import csv
import itertools
import json
import math
from functools import reduce
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as pyplot

from utils.Types import AggregationType, AlgType, CutPattern

def calculate_avg_result(result_list):
    keys_to_average = ['egalitarianGain', 'utilitarianGain', 'averageFaceRatio', 'largestFaceRatio', 'largestEnvy']
    if result_list:
        result = {}
        for key in result_list[0]:
            if key in keys_to_average:
                key_list_values = list(map(lambda res: res[key], result_list))
                avg_key = key+'_Avg'
                std_key = key + '_StDev'
                result[avg_key] = mean(key_list_values)
                result[std_key] = stdev(key_list_values)
            else:
                result[key] = result_list[-1][key]
                if key == "Method":
                    result[key] = result[key].replace(result_list[-1]["Algorithm"], "")
        return result
    else:
        return {}


def calculate_int_result(EvenPaz_res, Assessor_res):
    keys_to_integrate = ['egalitarianGain_Avg', 'utilitarianGain_Avg', 'averageFaceRatio_Avg', 'largestFaceRatio_Avg', 'largestEnvy_Avg']
    if EvenPaz_res:
        result = {}
        for key in EvenPaz_res:
            if key in keys_to_integrate:
                if Assessor_res[key] == 0:
                    result[key] = "INF"
                else:
                    result[key] = (EvenPaz_res[key] - Assessor_res[key])/Assessor_res[key]
            else:
                result[key] = EvenPaz_res[key]
                if key == "Method":
                    result[key] = result[key].replace(EvenPaz_res["Algorithm"], "")
                if key == "Algorithm":
                    result[key] = "Integrated"
        return result
    else:
        return {}
    pass


def parse_sum_data_entry(orig_data_entry,sum_data_entry):

    if not sum_data_entry:
        return []
    dict_to_parse = {}

    for key in orig_data_entry:
        dict_to_parse[key] = orig_data_entry[key]

    dict_to_parse['egalitarianGain_Imp'] = sum_data_entry["egalitarianGain_Avg"]
    dict_to_parse['utilitarianGain_Imp'] = sum_data_entry["utilitarianGain_Avg"]
    dict_to_parse['averageFaceRatio_Imp'] = sum_data_entry["averageFaceRatio_Avg"]
    dict_to_parse['largestFaceRatio_Imp'] = sum_data_entry["largestFaceRatio_Avg"]
    dict_to_parse['largestEnvy_Imp'] = sum_data_entry ["largestEnvy_Avg"]

    key_order = ['Method',
         'egalitarianGain_Avg', 'egalitarianGain_Imp', 'egalitarianGain_StDev',
         'utilitarianGain_Avg', 'utilitarianGain_Imp', 'utilitarianGain_StDev',
         'averageFaceRatio_Avg', 'averageFaceRatio_Imp', 'averageFaceRatio_StDev',
         'largestFaceRatio_Avg', 'largestFaceRatio_Imp', 'largestFaceRatio_StDev',
         'largestEnvy_Avg', 'largestEnvy_Imp', 'largestEnvy_StDev']

    return [dict_to_parse[key] for key in key_order]


def generate_reports(jsonfilename):
    filepath_elements = jsonfilename.split('_')
    aggText = filepath_elements[1]
    aggParam = filepath_elements[2]
    experiments_per_cell = int(filepath_elements[3])
    dataParamType = AggregationType.NumberOfAgents
    with open(jsonfilename) as json_file:
        results = json.load(json_file)
    results_per_method = {
    method: [r for r in results if r["Method"].replace("EvenPaz", "").replace("Assessor", "") == method] for method in
    [m.name for m in CutPattern]}
    for m in results_per_method:
        results_per_method[m] = {n: [r for r in results_per_method[m] if r['NumberOfAgents'] == n] for n in
                                 [4, 8, 16, 32, 64, 128]}  # todo make dynamic to group size
        for n in results_per_method[m]:
            results_per_method[m][n] = {a: [r for r in results_per_method[m][n] if r['Algorithm'] == a] for a in
                                        ["EvenPaz", "Assessor"]}
    avg_results_per_method = {}
    sum_results_per_groupsize = {4: [], 8: [], 16: [], 32: [], 64: [], 128: []}  # todo make dynamic to group size
    for m in results_per_method:
        method_avg_results = {}
        for n in results_per_method[m]:
            method_avg_results[n] = {a: calculate_avg_result(results_per_method[m][n][a]) for a in
                                     results_per_method[m][n]}
            EvenPaz_res = method_avg_results[n]["EvenPaz"]
            Assessor_res = method_avg_results[n]["Assessor"]
            method_avg_results[n]["Integrated"] = calculate_int_result(EvenPaz_res, Assessor_res)
            sum_results_per_groupsize[n].append(
                parse_sum_data_entry(method_avg_results[n]["EvenPaz"], method_avg_results[n]["Integrated"]))
        avg_results_per_method[m] = method_avg_results
    with open(jsonfilename + '_summary.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow([jsonfilename])
        first_headline = ["Cut Heuristic", "egalitarianGain", "", "", "utilitarianGain", "", "", "averageFaceRatio", "",
                          "", "largestFaceRatio", "", "", "largestEnvy", "", ""]
        second_headline = ["", "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev",
                           "AverageRatio", "Improv(%)", "StDev", "AverageRatio", "Improv(%)", "StDev",
                           "AverageGain", "Improv(%)", "StDev"]
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
    with open(jsonfilename + '_results.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        keys_list = []
        csv_file_writer.writerow(keys_list)
        for m in avg_results_per_method:
            for n in avg_results_per_method[m]:
                r = avg_results_per_method[m][n]["EvenPaz"]
                if not r:
                    continue
                if not keys_list:
                    keys_list = list(r.keys())
                    csv_file_writer.writerow(keys_list)
                csv_file_writer.writerow([r[key] for key in keys_list])
                r = avg_results_per_method[m][n]["Assessor"]
                csv_file_writer.writerow([r[key] for key in keys_list])
                r = avg_results_per_method[m][n]["Integrated"]
                csv_file_writer.writerow([r[key] for key in keys_list])


if __name__ == '__main__':

    """ generate report of experiment results from json file """

    files_to_import = ['D:/MSc/Thesis/CakeCutting/results/2019-01-28T15-12-12_NoiseProportion_random_10_exp_Random.json',
                       'D:/MSc/Thesis/CakeCutting/results/2019-01-28T19-24-55_NoiseProportion_0.2_10_exp_NewZealand.json',
                       'D:/MSc/Thesis/CakeCutting/results/2019-01-29T03-06-01_NoiseProportion_0.2_10_exp_Israel.json']

    for jsonfilename in files_to_import:
        generate_reports(jsonfilename)
