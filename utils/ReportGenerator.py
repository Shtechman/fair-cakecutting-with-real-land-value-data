import csv
import itertools
import json
import math
import os
from datetime import datetime
from functools import reduce
from statistics import mean, stdev


from utils.Types import AggregationType, CutPattern

def calculate_avg_result(result_list, keys_to_average):

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


def calculate_int_result(EvenPaz_res, Assessor_res,keys_to_integrate):
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


def parse_sum_data_entry(orig_data_entry, sum_data_entry):

    if not sum_data_entry:
        return []
    dict_to_parse = {}

    for key in orig_data_entry:
        dict_to_parse[key] = orig_data_entry[key]

    dict_to_parse['egalitarianGain_Imp'] = sum_data_entry["egalitarianGain_Avg"]
    dict_to_parse['utilitarianGain_Imp'] = sum_data_entry["utilitarianGain_Avg"]
    dict_to_parse['averageFaceRatio_Imp'] = sum_data_entry["averageFaceRatio_Avg"]
    dict_to_parse['largestFaceRatio_Imp'] = sum_data_entry["largestFaceRatio_Avg"]
    dict_to_parse['smallestFaceRatio_Imp'] = sum_data_entry["smallestFaceRatio_Avg"]
    dict_to_parse['largestInheritanceGain_Imp'] = sum_data_entry["largestInheritanceGain_Avg"]
    dict_to_parse['averageInheritanceGain_Imp'] = sum_data_entry["averageInheritanceGain_Avg"]
    dict_to_parse['largestEnvy_Imp'] = sum_data_entry["largestEnvy_Avg"]

    key_order = ['Method',
         'egalitarianGain_Avg', 'egalitarianGain_Imp', 'egalitarianGain_StDev',
         'utilitarianGain_Avg', 'utilitarianGain_Imp', 'utilitarianGain_StDev',
         'averageFaceRatio_Avg', 'averageFaceRatio_Imp', 'averageFaceRatio_StDev',
         'largestFaceRatio_Avg', 'largestFaceRatio_Imp', 'largestFaceRatio_StDev',
         'smallestFaceRatio_Avg', 'smallestFaceRatio_Imp', 'smallestFaceRatio_StDev',
         'largestInheritanceGain_Avg', 'largestInheritanceGain_StDev',
         'averageInheritanceGain_Avg', 'averageInheritanceGain_StDev',
         'largestEnvy_Avg', 'largestEnvy_Imp', 'largestEnvy_StDev',
         'experimentDurationSec_Avg', 'experimentDurationSec_StDev']

    return [dict_to_parse[key] for key in key_order]


def create_table_summary_line(groupsize_sum_results):
    max_idx_dict = {'EGA': 1,
           'EGP': 2,
           'UGA': 4,
           'UGP': 5,
           'AFA': 7,
           'AFP': 8,
           'LFA': 10,
           'LFP': 11,
           'SFA': 13,
           'SFP': 14,
           'LIA': 16,
           'AIA': 18}
    min_idx_dict = {
                'LEA': 20,
                'LEP': 21,
                'RDA': 23}
    sum_result = [' ']*len(groupsize_sum_results[0])
    if sum_result:
        sum_result[0] = "Summary"

        for idx in max_idx_dict.values():
            sum_result[idx] = next((r[0] for r in groupsize_sum_results if
                                    r and (r[idx] == max([r[idx] for r in groupsize_sum_results if r]))), " ")

        for idx in min_idx_dict.values():
            sum_result[idx] = next((r[0] for r in groupsize_sum_results if
                                    r and (r[idx] == min([r[idx] for r in groupsize_sum_results if r]))), " ")
    return sum_result


def generate_reports(jsonfilename):
    # filepath_elements = jsonfilename.split('_')
    # aggText = filepath_elements[2]
    # aggParam = filepath_elements[3]
    # experiments_per_cell = int(filepath_elements[4])
    # dataParamType = AggregationType.NumberOfAgents
    with open(jsonfilename) as json_file:
        results = json.load(json_file)

    avg_results_per_method,\
    sum_results_per_groupsize,\
    avg_evenpaz_results_per_measurement,\
    avg_assessor_results_per_measurement,\
    groupsizes = preprocess_results(results)

    write_graphtables_report_csv(jsonfilename, avg_evenpaz_results_per_measurement, groupsizes)
    write_graphtables_report_csv(jsonfilename, avg_assessor_results_per_measurement, groupsizes, "Assessor")
    write_summary_report_csv(jsonfilename, sum_results_per_groupsize)
    write_extended_report_csv(jsonfilename, avg_results_per_method)


def preprocess_results(results):
    keys_to_average = ['egalitarianGain', 'utilitarianGain', 'averageFaceRatio', 'largestFaceRatio',
                       'smallestFaceRatio',
                       'largestInheritanceGain', 'averageInheritanceGain', 'largestEnvy', 'experimentDurationSec']
    keys_to_integrate = [key +'_Avg' for key in keys_to_average]
    available_groupsizes = [4, 8, 16, 32, 64, 128]  # todo make dynamic to group size

    res_per_m = { # m - method
        method: [r for r in results if r["Method"].replace("EvenPaz", "").replace("Assessor", "") == method] for method
    in
        [m.name for m in CutPattern]}

    res_per_gs_per_m = {}  # gs - groupsize, m - method
    res_per_a_per_gs_per_m = {}  # a - algorithm, gs - groupsize, m - method

    for method in res_per_m:
        res_per_gs_per_m[method] = {n: [r for r in res_per_m[method]
                                                        if r['NumberOfAgents'] == n] for n in available_groupsizes}
        res_per_a_per_gs_per_m[method] = {n: [] for n in available_groupsizes}
        for groupsize in res_per_gs_per_m[method]:
            res_per_a_per_gs_per_m[method][groupsize] = {a: [r for r in res_per_gs_per_m[method][groupsize]
                                                             if r['Algorithm'] == a] for a in ["EvenPaz", "Assessor"]}
    avg_results_per_method = {}
    sum_results_per_groupsize = {n: [] for n in available_groupsizes}
    for method in res_per_a_per_gs_per_m:
        method_avg_results = {}
        for groupsize in res_per_a_per_gs_per_m[method]:
            method_avg_results[groupsize] = {a: calculate_avg_result(res_per_a_per_gs_per_m[method][groupsize][a],
                                                                     keys_to_average)
                                             for a in res_per_a_per_gs_per_m[method][groupsize]}
            EvenPaz_res = method_avg_results[groupsize]["EvenPaz"]
            Assessor_res = method_avg_results[groupsize]["Assessor"]
            method_avg_results[groupsize]["Integrated"] = calculate_int_result(EvenPaz_res, Assessor_res,keys_to_integrate)
            sum_results_per_groupsize[groupsize].append(
                parse_sum_data_entry(method_avg_results[groupsize]["EvenPaz"], method_avg_results[groupsize]["Integrated"]))
        avg_results_per_method[method] = method_avg_results

    avg_evenpaz_results_per_measurement = {measure:
                                       {method:
                                            [avg_results_per_method[method][groupsize]["EvenPaz"][measure]
                                             if avg_results_per_method[method][groupsize]["EvenPaz"] else ""
                                             for groupsize in avg_results_per_method[method]]
                                        for method in avg_results_per_method}
                                   for measure in keys_to_integrate}

    avg_assessor_results_per_measurement = {measure:
                                       {method:
                                            [avg_results_per_method[method][groupsize]["Assessor"][measure]
                                            if avg_results_per_method[method][groupsize]["Assessor"] else ""
                                             for groupsize in avg_results_per_method[method]]
                                        for method in avg_results_per_method}
                                   for measure in keys_to_integrate}

    for groupSize in sum_results_per_groupsize:
        if not sum_results_per_groupsize[groupSize]:
            continue
        sum_results_per_groupsize[groupSize].append(create_table_summary_line(sum_results_per_groupsize[groupSize]))
    return avg_results_per_method,\
           sum_results_per_groupsize,\
           avg_evenpaz_results_per_measurement,\
           avg_assessor_results_per_measurement,\
           available_groupsizes


def write_extended_report_csv(jsonfilename, avg_results_per_method):
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


def write_summary_report_csv(jsonfilename, sum_results_per_groupsize):
    with open(jsonfilename + '_summary.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow([jsonfilename])
        first_headline = ["Cut Heuristic", "egalitarianGain", "", "", "utilitarianGain", "", "", "averageFaceRatio", "",
                          "", "largestFaceRatio", "", "", "smallestFaceRatio", "", "", "largestInheritanceGain", "",
                          "averageInheritanceGain", "", "largestEnvy", "", "", "runDuration(sec)", ""]
        second_headline = ["", "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev",
                           "AverageRatio", "Improv(%)", "StDev", "AverageRatio", "Improv(%)", "StDev",
                           "AverageGain", "Improv(%)", "StDev", "AverageGain", "StDev",
                           "AverageGain", "StDev", "AverageGain", "Improv(%)", "StDev", "Average Time", "StDev"]
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

def write_graphtables_report_csv(jsonfilename, results_per_measuement, groupsizes, label="EvenPaz"):
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
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        os.makedirs(result_log_folder)
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

    files_to_import = ['D:/MSc/Thesis/CakeCutting/results/2019-01-28T15-12-12_NoiseProportion_random_10_exp_Random.json',
                       'D:/MSc/Thesis/CakeCutting/results/2019-01-28T19-24-55_NoiseProportion_0.2_10_exp_NewZealand.json',
                       'D:/MSc/Thesis/CakeCutting/results/2019-01-29T03-06-01_NoiseProportion_0.2_10_exp_Israel.json']

    for jsonfilename in files_to_import:
        generate_reports(jsonfilename)
