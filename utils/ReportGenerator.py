import csv
import itertools
import json
import math
import os
from datetime import datetime
from functools import reduce
from statistics import mean, stdev
from utils.Types import AggregationType, CutPattern
import numpy as np

try:
    from scipy.stats import t
    def calculate_confidence_interval(confidence, stdev, n):
        return stdev * t.ppf((1 + confidence) / 2., n - 1)
except:
    def calculate_confidence_interval(confidence, stdev, n):
        return -1





def calculate_avg_result(result_list, keys_to_average, groupsize):
    if result_list:
        result = {}
        for key in result_list[0]:
            if key in keys_to_average:
                key_list_values = list(map(lambda res: res[key], result_list))
                avg_key = key+'_Avg'
                std_key = key + '_StDev'
                interval_key = key + '_interval'
                result[avg_key] = mean(key_list_values)
                result[std_key] = stdev(key_list_values)
                result[interval_key] = calculate_confidence_interval(0.95, result[std_key], groupsize)
            else:
                result[key] = result_list[-1][key]
                if key == "Method":
                    result[key] = result[key].split("_")[-1]
        return result
    else:
        return {}


def calculate_int_result(Algorithm_res, Assessor_res, keys_to_integrate):
    if Algorithm_res and Assessor_res:
        result = {}
        for key in Algorithm_res:
            if key in keys_to_integrate:
                if Assessor_res[key] == 0:
                    result[key] = "INF"
                else:
                    if "ttc_" in key:
                        assessor_key = key.replace("ttc_", "")
                    else:
                        assessor_key = key
                    result[key] = (Algorithm_res[key] - Assessor_res[assessor_key]) / Assessor_res[assessor_key]
            else:
                result[key] = Algorithm_res[key]
                if key == "Method":
                    result[key] = result[key].split("_")[-1]
                if key == "Algorithm":
                    result[key] = "Integrated_" + Algorithm_res[key]
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
    dict_to_parse['ttc_egalitarianGain_Imp'] = sum_data_entry["ttc_egalitarianGain_Avg"]
    dict_to_parse['utilitarianGain_Imp'] = sum_data_entry["utilitarianGain_Avg"]
    dict_to_parse['ttc_utilitarianGain_Imp'] = sum_data_entry["ttc_utilitarianGain_Avg"]
    dict_to_parse['averageFaceRatio_Imp'] = sum_data_entry["averageFaceRatio_Avg"]
    dict_to_parse['largestFaceRatio_Imp'] = sum_data_entry["largestFaceRatio_Avg"]
    dict_to_parse['smallestFaceRatio_Imp'] = sum_data_entry["smallestFaceRatio_Avg"]
    dict_to_parse['largestInheritanceGain_Imp'] = sum_data_entry["largestInheritanceGain_Avg"]
    dict_to_parse['averageInheritanceGain_Imp'] = sum_data_entry["averageInheritanceGain_Avg"]
    dict_to_parse['largestEnvy_Imp'] = -1 * sum_data_entry["largestEnvy_Avg"]
    dict_to_parse['ttc_largestEnvy_Imp'] = -1 * sum_data_entry["ttc_largestEnvy_Avg"]

    key_order = ['Method',
         'egalitarianGain_Avg', 'egalitarianGain_Imp', 'egalitarianGain_StDev',
         'ttc_egalitarianGain_Avg', 'ttc_egalitarianGain_Imp', 'ttc_egalitarianGain_StDev',
         'utilitarianGain_Avg', 'utilitarianGain_Imp', 'utilitarianGain_StDev',
         'ttc_utilitarianGain_Avg', 'ttc_utilitarianGain_Imp', 'ttc_utilitarianGain_StDev',
         'averageFaceRatio_Avg', 'averageFaceRatio_Imp', 'averageFaceRatio_StDev',
         'largestFaceRatio_Avg', 'largestFaceRatio_Imp', 'largestFaceRatio_StDev',
         'smallestFaceRatio_Avg', 'smallestFaceRatio_Imp', 'smallestFaceRatio_StDev',
         'largestInheritanceGain_Avg', 'largestInheritanceGain_StDev',
         'averageInheritanceGain_Avg', 'averageInheritanceGain_StDev',
         'largestEnvy_Avg', 'largestEnvy_Imp', 'largestEnvy_StDev',
         'ttc_largestEnvy_Avg', 'ttc_largestEnvy_Imp', 'ttc_largestEnvy_StDev',
         'experimentDurationSec_Avg', 'experimentDurationSec_StDev']

    return [dict_to_parse[key] for key in key_order]


def create_table_summary_line(groupsize_sum_results):
    max_idx_dict = {'EGA': 1,
           'EGP': 2,
           'ttcEGA': 4,
           'ttcEGP': 5,
           'UGA': 7,
           'UGP': 8,
           'ttcUGA': 10,
           'ttcUGP': 11,
           'AFA': 13,
           'AFP': 14,
           'LFA': 16,
           'LFP': 17,
           'SFA': 19,
           'SFP': 20,
           'LIA': 22,
           'AIA': 24}
    min_idx_dict = {
                'LEA': 26,
                'LEP': 27,
                'ttcLEA': 29,
                'ttcLEP': 30,
                'RDA': 32}
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
    with open(jsonfilename + '_dishonestGain.csv', "w", newline='') as csv_file:
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
    write_graphtables_report_csv(jsonfilename, avg_dishonest_results_per_measurement, groupsizes, "Dishonest")

    write_summary_report_csv(jsonfilename, sum_honest_results_per_groupsize)
    write_summary_report_csv(jsonfilename, sum_dishonest_results_per_groupsize, "Dishonest")

    write_graph_method_per_measure_report_csv(jsonfilename, graph_method_results_per_measure)

    write_extended_report_csv(jsonfilename, avg_results_per_method)

    write_dishonest_gain_results(jsonfilename, dishonest_gain)

    write_bruteforce_gain_results(jsonfilename, bruteForce_gain)


def parse_method_over_measure_data_entry(measure, method, available_groupsizes, hon_avg, hon_interval, as_avg, as_interval, dis_avg, dis_interval, selling = 1.):
    #header = ['NumberOfAgents', 'Honest', 'Assessor', 'Dishonest',  'Selling', 'Hon_Conf', 'As_Conf', 'Dis_Conf']
    data_entry = []
    for idx,group in enumerate(available_groupsizes):
        data_entry.append([group, hon_avg[idx], as_avg[idx], dis_avg[idx], selling,
                           hon_interval[idx], as_interval[idx], dis_interval[idx]])
    return data_entry


def preprocess_results(results):
    keys_to_average = ['egalitarianGain', 'ttc_egalitarianGain', 'ttc_utilitarianGain','utilitarianGain', 'averageFaceRatio', 'largestFaceRatio',
                       'smallestFaceRatio',
                       'largestInheritanceGain', 'averageInheritanceGain', 'largestEnvy', 'ttc_largestEnvy', 'experimentDurationSec']
    keys_to_integrate = [key + '_Avg' for key in keys_to_average]
    interval_keys = [key + '_interval' for key in keys_to_average]
    available_groupsizes = [4, 8, 16, 32, 64, 128]  # todo make dynamic to group size

    def _fix_parse(res):
        m = res['Method']
        if "Assessor" in m:
            res['Method'] = res['Method'].replace('Assessor','Assessor_EvenPaz_')
            res['Algorithm'] = 'Assessor_EvenPaz'
        if "Honest" in m:
            res['Algorithm'] = 'Honest_EvenPaz'
        return res

    results = [_fix_parse(r) for r in results]

    dishonestGain = get_dishonest_gain(results)


    honest = "Honest_EvenPaz"
    dishonest = "Dishonest_EvenPaz"
    assessor = "Assessor_EvenPaz"
    integratedH = "Integrated_Honest_EvenPaz"
    integratedD = "Integrated_Dishonest_EvenPaz"
    bruteForce = "BruteForce"

    res_per_m = { # m - method
        method: [r for r in results if r["Method"].split("_")[-1] == method] for method
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
                                                             if r['Algorithm'] == a] for a in [honest, assessor, dishonest]}
    avg_results_per_method = {}
    sum_honest_results_per_groupsize = {n: [] for n in available_groupsizes}
    sum_dishonest_results_per_groupsize = {n: [] for n in available_groupsizes}

    best_result_per_gs = {}
    if bruteForce in res_per_a_per_gs_per_m:
        keys = [('egalitarianGain',max),
                ('ttc_egalitarianGain',max),
                ('ttc_utilitarianGain',max),
                ('utilitarianGain',max),
                ('averageFaceRatio',max),
                ('smallestFaceRatio',max),
                ('largestEnvy',min),
                ('ttc_largestEnvy',min),
                ('experimentDurationSec',min)]
        for groupsize in res_per_a_per_gs_per_m[bruteForce]:
            if res_per_a_per_gs_per_m[bruteForce][groupsize][honest]:
                uniqe_exp_id = list(set([r['experiment'] for r in res_per_a_per_gs_per_m[bruteForce][groupsize][honest]]))
                best_result_per_gs[groupsize] = []
                for key in keys:
                    cur_key_list = []
                    for exp_id in uniqe_exp_id:
                        exp_results = [r for r in res_per_a_per_gs_per_m[bruteForce][groupsize][honest]
                                       if r['experiment'] == exp_id]
                        cur_key_list.append(key[1](exp_results, key=lambda r: r[key[0]])[key[0]])

                    best_result_per_gs[groupsize].append([key[0], mean(cur_key_list), stdev(cur_key_list)])

    for method in res_per_a_per_gs_per_m:
        method_avg_results = {}
        
        for groupsize in res_per_a_per_gs_per_m[method]:
            method_avg_results[groupsize] = {a: calculate_avg_result(res_per_a_per_gs_per_m[method][groupsize][a],
                                                                     keys_to_average, groupsize)
                                             for a in res_per_a_per_gs_per_m[method][groupsize]}
            Honest_res = method_avg_results[groupsize][honest]
            Assessor_res = method_avg_results[groupsize][assessor]
            Dishonest_res = method_avg_results[groupsize][dishonest]
            method_avg_results[groupsize][integratedH] = calculate_int_result(Honest_res, Assessor_res,
                                                                             keys_to_integrate)
            method_avg_results[groupsize][integratedD] = calculate_int_result(Dishonest_res, Assessor_res,
                                                                             keys_to_integrate)
            sum_honest_results_per_groupsize[groupsize].append(
                parse_sum_data_entry(method_avg_results[groupsize][honest], method_avg_results[groupsize][integratedH]))
            sum_dishonest_results_per_groupsize[groupsize].append(
                parse_sum_data_entry(method_avg_results[groupsize][dishonest], method_avg_results[groupsize][integratedD]))

        # todo - implement parser for graph summary file
        avg_results_per_method[method] = method_avg_results

    avg_honest_results_per_measurement = {measure:
                                       {method:
                                            [avg_results_per_method[method][groupsize][honest][measure]
                                             if avg_results_per_method[method][groupsize][honest] else ""
                                             for groupsize in avg_results_per_method[method]]
                                        for method in avg_results_per_method}
                                   for measure in keys_to_integrate}

    interval_honest_results_per_measurement = {measure:
                                               {method:
                                                    [avg_results_per_method[method][groupsize][honest][measure]
                                                     if avg_results_per_method[method][groupsize][honest] else ""
                                                     for groupsize in avg_results_per_method[method]]
                                                for method in avg_results_per_method}
                                           for measure in interval_keys}

    avg_assessor_results_per_measurement = {measure:
                                       {method:
                                            [avg_results_per_method[method][groupsize][assessor][measure]
                                            if avg_results_per_method[method][groupsize][assessor] else ""
                                             for groupsize in avg_results_per_method[method]]
                                        for method in avg_results_per_method}
                                   for measure in keys_to_integrate}


    interval_assessor_results_per_measurement = {measure:
                                                {method:
                                                     [avg_results_per_method[method][groupsize][assessor][measure]
                                                      if avg_results_per_method[method][groupsize][assessor] else ""
                                                      for groupsize in avg_results_per_method[method]]
                                                 for method in avg_results_per_method}
                                            for measure in interval_keys}

    avg_dishonest_results_per_measurement = {measure:
                                                {method:
                                                     [avg_results_per_method[method][groupsize][dishonest][measure]
                                                      if avg_results_per_method[method][groupsize][dishonest] else ""
                                                      for groupsize in avg_results_per_method[method]]
                                                 for method in avg_results_per_method}
                                            for measure in keys_to_integrate}

    interval_dishonest_results_per_measurement = {measure:
                                                     {method:
                                                          [avg_results_per_method[method][groupsize][dishonest][measure]
                                                           if avg_results_per_method[method][groupsize][
                                                              dishonest] else ""
                                                           for groupsize in avg_results_per_method[method]]
                                                      for method in avg_results_per_method}
                                                 for measure in interval_keys}
    
    graph_method_results_per_measure = {measure: {} for measure in keys_to_average}
    for measure in graph_method_results_per_measure:
        avg_key = measure + '_Avg'
        ass_avg_key = avg_key.replace('ttc_', '')
        interval_key = measure + '_interval'
        ass_int_key = interval_key.replace('ttc_','')
        honest_avg = avg_honest_results_per_measurement[avg_key]
        assessor_avg = avg_assessor_results_per_measurement[ass_avg_key]
        dishonest_avg = avg_dishonest_results_per_measurement[avg_key]
        honest_interval = interval_honest_results_per_measurement[interval_key]
        assessor_interval = interval_assessor_results_per_measurement[ass_int_key]
        dishonest_interval = interval_dishonest_results_per_measurement[interval_key]
        for method in avg_results_per_method:
            graph_method_results_per_measure[measure][method] = parse_method_over_measure_data_entry(measure,method,
                                                                                                     available_groupsizes,
                                                                                                     honest_avg[method],honest_interval[method],
                                                                                                     assessor_avg[method],assessor_interval[method],
                                                                                                     dishonest_avg[method],dishonest_interval[method])


    for groupSize in sum_honest_results_per_groupsize:
        if not sum_honest_results_per_groupsize[groupSize]:
            continue
        sum_honest_results_per_groupsize[groupSize].append(create_table_summary_line(sum_honest_results_per_groupsize[groupSize]))

    for groupSize in sum_dishonest_results_per_groupsize:
        if not sum_dishonest_results_per_groupsize[groupSize]:
            continue
        sum_dishonest_results_per_groupsize[groupSize].append(create_table_summary_line(sum_dishonest_results_per_groupsize[groupSize]))

    return avg_results_per_method, \
           sum_honest_results_per_groupsize, \
           sum_dishonest_results_per_groupsize, \
           avg_honest_results_per_measurement,\
           avg_assessor_results_per_measurement, \
           avg_dishonest_results_per_measurement, \
           graph_method_results_per_measure, \
           available_groupsizes, \
           dishonestGain, \
           best_result_per_gs


def write_bruteforce_gain_results(jsonfilename, bruteForce_gain):
    with open(jsonfilename + '_BruteForceGain.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        for groupsize in bruteForce_gain:
            csv_file_writer.writerow([groupsize, "agents"])
            csv_file_writer.writerow(["Measure", "Avg", "stdev"])
            for key_data in bruteForce_gain[groupsize]:
                csv_file_writer.writerow(key_data)
            csv_file_writer.writerow([])


def write_extended_report_csv(jsonfilename, avg_results_per_method):
    honest = "Honest_EvenPaz"
    assessor = "Assessor_EvenPaz"
    dishonest = "Dishonest_EvenPaz"
    integratedH = "Integrated_Honest_EvenPaz"
    integratedD = "Integrated_Dishonest_EvenPaz"

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
                keys_list = _write_to_csv(keys_list, honest)
                keys_list = _write_to_csv(keys_list, assessor)
                keys_list = _write_to_csv(keys_list, integratedH)
                keys_list = _write_to_csv(keys_list, integratedD)
                keys_list = _write_to_csv(keys_list, dishonest)


def write_summary_report_csv(jsonfilename, sum_results_per_groupsize, label="Honest"):
    with open(jsonfilename + '_' + label + '_summary.csv', "w", newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow([jsonfilename])
        first_headline = ["Cut Heuristic", "egalitarianGain", "", "", "ttc_egalitarianGain", "", "", "utilitarianGain",
                          "", "", "ttc_utilitarianGain", "", "", "averageFaceRatio", "",
                          "", "largestFaceRatio", "", "", "smallestFaceRatio", "", "", "largestInheritanceGain", "",
                          "averageInheritanceGain", "", "largestEnvy", "", "", "ttc_largestEnvy", "", "", "runDuration(sec)", ""]
        second_headline = ["", "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev","AverageGain", "Improv(%)", "StDev",
                           "AverageRatio", "Improv(%)", "StDev", "AverageRatio", "Improv(%)", "StDev",
                           "AverageGain", "Improv(%)", "StDev", "AverageGain", "StDev",
                           "AverageGain", "StDev", "AverageGain", "Improv(%)", "StDev", "AverageGain", "Improv(%)", "StDev", "Average Time", "StDev"]
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


def get_dishonest_gain(results):
    experiment_list = list(set([rlog["experiment"] for rlog in results]))
    cut_pattern_list = list(set([rlog["Method"].replace("Dishonest_","").replace("Honest_","") for rlog in results]))
    num_agents = list(set([rlog[AggregationType.NumberOfAgents.name] for rlog in results]))

    result = {numA: {} for numA in num_agents}

    def _get_dishonest_gain(honest_partition, dishonest_partition, dishonest_agent):
        dis_value = dishonest_partition[dishonest_agent]
        hon_value = honest_partition[dishonest_agent]

        return dis_value - hon_value, (dis_value - hon_value) / hon_value

    for exp in experiment_list:
        numA = [rlog for rlog in results if rlog["experiment"] == exp][0][AggregationType.NumberOfAgents.name]
        result[numA][exp] = {}
        for cp in cut_pattern_list:
            result[numA][exp][cp] = []
            relevant_logs = [rlog for rlog in results if (rlog["experiment"] == exp and rlog["Method"].replace("Dishonest_","").replace("Honest_","") == cp)]
            dishonest_partitions = {rlog["dishonestAgent"]: rlog["relativeValues"] for rlog in relevant_logs if rlog["dishonestAgent"] is not None}
            honest = [rlog for rlog in relevant_logs if rlog["dishonestAgent"] is None][0]
            honest_partitions = honest["relativeValues"]
            for agent in dishonest_partitions:
                v,p = _get_dishonest_gain(honest_partitions, dishonest_partitions[agent], agent)

                result[numA][exp][cp].append(
                {
                "agent": agent,
                "agent_gain": v,
                "agent_gain_per": p
                })

    return result


def create_exp_folder(run_folder, exp_name_string):
    result_folder = run_folder + exp_name_string + "/"
    result_log_folder = result_folder + "logs/"
    result_ttc_folder = result_folder + "ttc/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        os.makedirs(result_log_folder)
        os.makedirs(result_ttc_folder)
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
                        'D:/MSc/Thesis/CakeCutting/results/2019-05-03T15-41-32/newZealandLowResAgents06_2019-05-03T15-41-52_NoiseProportion_0.6_50_exp/ttc_post_process_results_st16/ttc_post_process_results_st16.json',
                        # 'D:/MSc/Thesis/CakeCutting/results/2019-08-27T19-45-25/newZealandLowResAgents06_2019-08-27T19-45-44_NoiseProportion_0.6_30_exp/newZealandLowResAgents06_2019-08-27T19-45-44_NoiseProportion_0.6_30_exp.json',
                        #'D:/MSc/Thesis/CakeCutting/results/2019-07-08T21-55-10/randomMaps06_2019-07-12T06-16-05_NoiseProportion_0.6_30_exp/randomMaps06_Dis_NoiseProportion_0.6_30_exp.json',
    ]

    for jsonfilename in files_to_import:
        generate_reports(jsonfilename)
