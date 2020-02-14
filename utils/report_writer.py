import csv
import json
import os
from datetime import datetime
from statistics import stdev
import numpy as np
from utils.report_generator import preprocess_results


def write_graph_method_per_measure_report_csv(
    path, json_file_name, graph_method_results_per_measure
):
    graph_report_path = os.path.join(path, "measurements_graphs")
    alg_name = os.path.basename(path)
    if not os.path.exists(graph_report_path):
        os.makedirs(graph_report_path)
    for measure in graph_method_results_per_measure:
        measure_results = graph_method_results_per_measure[measure]
        if not graph_method_results_per_measure[measure]:
            continue
        report_file_path = os.path.join(
            graph_report_path, measure + "_" + alg_name + ".csv"
        )
        with open(report_file_path, "w", newline="") as csv_file:
            csv_file_writer = csv.writer(csv_file)
            csv_file_writer.writerow([json_file_name])
            table_header = [
                "NumberOfAgents",
                alg_name,
                "Assessor",
                alg_name + "TTC",
                "Selling",
                alg_name + "_Conf",
                "As_Conf",
                alg_name + "TTC_Conf",
            ]
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


def write_dishonest_gain_results(path, dishonest_gain, label=""):
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
            exp_strategy_gain_avg = np.average(
                [exp["sgAvg"] for exp in exp_list]
            )
            exp_strategy_percentage_gain_avg = np.average(
                [exp["sgAvg_per"] for exp in exp_list]
            )
            exp_strategy_gain_avg_stdev = stdev(
                [exp["sgAvg"] for exp in exp_list]
            )
            exp_strategy_gain_max = np.average(
                [exp["sgMax"] for exp in exp_list]
            )
            exp_strategy_percentage_gain_max = np.average(
                [exp["sgMax_per"] for exp in exp_list]
            )
            exp_strategy_gain_max_stdev = stdev(
                [exp["sgMax"] for exp in exp_list]
            )
            agg_dis_data[numOfAgent][cut_pattern] = {
                "sgAvg": exp_strategy_gain_avg,
                "sgAvg_per": exp_strategy_percentage_gain_avg,
                "sgAvgStdev": exp_strategy_gain_avg_stdev,
                "sgMax": exp_strategy_gain_max,
                "sgMax_per": exp_strategy_percentage_gain_max,
                "sgMaxStdev": exp_strategy_gain_max_stdev,
            }

    if not res_found:
        return

    with open(
        os.path.join(path, label + "_DishonestGain.csv"), "w", newline=""
    ) as csv_file:
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


def generate_reports(json_file_name):
    with open(json_file_name) as json_file:
        results = json.load(json_file)

    different_algorithm = list(
        set(
            [
                k.split("_")[-1]
                for k in list(set([r["Algorithm"] for r in results]))
            ]
        )
    )
    assessor_algorithm = next(
        a for a in different_algorithm if "NoPattern" in a
    )
    different_algorithm.remove(assessor_algorithm)
    results_by_algorithm = {
        a: [r for r in results if a in r["Algorithm"]]
        for a in different_algorithm
    }
    assessor_results = [r for r in results if "Assessor" in r["Algorithm"]]

    for algorithm in results_by_algorithm:
        generate_algorithm_report(
            algorithm,
            json_file_name,
            results_by_algorithm[algorithm],
            assessor_results,
        )


def generate_algorithm_report(
    algorithm, json_file_name, results, assessor_results
):
    (
        avg_results_per_method,
        sum_honest_results_per_groupsize,
        sum_dishonest_results_per_groupsize,
        avg_honest_results_per_measurement,
        avg_assessor_results_per_measurement,
        avg_dishonest_results_per_measurement,
        graph_method_results_per_measure,
        groupsizes,
        dishonest_gain,
        bruteForce_gain,
    ) = preprocess_results(results, assessor_results)

    base_dir = os.path.dirname(json_file_name)
    json_file_name = os.path.basename(json_file_name)
    algorithm_res_path = os.path.join(base_dir, algorithm)
    if not os.path.exists(algorithm_res_path):
        os.makedirs(algorithm_res_path)

    write_graphtables_report_csv(
        algorithm_res_path,
        json_file_name,
        avg_honest_results_per_measurement,
        groupsizes,
        algorithm,
    )
    write_graphtables_report_csv(
        algorithm_res_path,
        json_file_name,
        avg_assessor_results_per_measurement,
        groupsizes,
        "Assessor",
    )
    # write_graphtables_report_csv(
    #   algorithm_res_path, jsonfilename, avg_dishonest_results_per_measurement, groupsizes, "Dishonest"
    #   )
    write_summary_report_csv(
        algorithm_res_path,
        json_file_name,
        sum_honest_results_per_groupsize,
        algorithm,
    )
    # write_summary_report_csv(
    #   algorithm_res_path, jsonfilename, sum_dishonest_results_per_groupsize, "Dishonest"
    #   )
    write_graph_method_per_measure_report_csv(
        algorithm_res_path, json_file_name, graph_method_results_per_measure
    )
    write_extended_report_csv(
        algorithm_res_path, json_file_name, avg_results_per_method, algorithm
    )
    write_dishonest_gain_results(algorithm_res_path, dishonest_gain, algorithm)
    write_bruteforce_gain_results(
        algorithm_res_path, bruteForce_gain, algorithm
    )


def write_bruteforce_gain_results(path, brute_force_gain, label=""):
    if not brute_force_gain:
        return
    with open(
        os.path.join(path, label + "_BruteForceGain.csv"), "w", newline=""
    ) as csv_file:
        csv_file_writer = csv.writer(csv_file)
        for groupsize in brute_force_gain:
            csv_file_writer.writerow([groupsize, "agents"])
            csv_file_writer.writerow(["Measure", "Avg", "stdev"])
            for key_data in brute_force_gain[groupsize]:
                csv_file_writer.writerow(key_data)
            csv_file_writer.writerow([])


def write_extended_report_csv(
    path, json_file_name, avg_results_per_method, label="Honest"
):
    json_file_name = os.path.basename(json_file_name)
    with open(
        os.path.join(path, json_file_name + "_" + label + "_results.csv"),
        "w",
        newline="",
    ) as csv_file:

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
                keys_to_report = avg_results_per_method[m][n].keys()
                for key_report in keys_to_report:
                    keys_list = _write_to_csv(keys_list, key_report)


def write_summary_report_csv(
    path, json_file_name, sum_results_per_groupsize, label="Honest"
):
    with open(
        os.path.join(path, label + "_summary.csv"), "w", newline=""
    ) as csv_file:
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow([json_file_name])
        first_headline = [
            "Cut Heuristic",
            "egalitarianGain",
            "",
            "",
            "ttc_egalitarianGain",
            "",
            "",
            "utilitarianGain",
            "",
            "",
            "ttc_utilitarianGain",
            "",
            "",
            "averageFaceRatio",
            "",
            "",
            "smallestFaceRatio",
            "",
            "",
            "largestEnvy",
            "",
            "",
            "ttc_largestEnvy",
            "",
            "",
            "runDuration(sec)",
            "",
        ]
        second_headline = [
            "",
            "AverageGain",
            "Improv(%)",
            "StDev",
            "AverageGain",
            "Improv(%)",
            "StDev",
            "AverageGain",
            "Improv(%)",
            "StDev",
            "AverageGain",
            "Improv(%)",
            "StDev",
            "AverageRatio",
            "Improv(%)",
            "StDev",
            "AverageRatio",
            "Improv(%)",
            "StDev",
            "AverageGain",
            "Improv(%)",
            "StDev",
            "AverageGain",
            "Improv(%)",
            "StDev",
            "Average Time",
            "StDev",
        ]
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


def write_graphtables_report_csv(
    path, json_file_name, results_per_measuement, groupsizes, label="Honest"
):
    with open(
        os.path.join(path, label + "_graphtables.csv"), "w", newline=""
    ) as csv_file:
        csv_file_writer = csv.writer(csv_file)
        csv_file_writer.writerow([json_file_name])
        headline = [""] + groupsizes
        for measure in results_per_measuement:
            if not results_per_measuement[measure]:
                continue
            csv_file_writer.writerow([label, measure])
            csv_file_writer.writerow(headline)
            for method in results_per_measuement[measure]:
                if not method:
                    continue
                if not results_per_measuement[measure][method]:
                    continue
                if not [
                    x for x in results_per_measuement[measure][method] if x
                ]:
                    continue
                csv_file_writer.writerow(
                    [method] + results_per_measuement[measure][method]
                )
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
    run_folder = (
        "./results/"
        + datetime.now().isoformat(timespec="seconds").replace(":", "-")
        + "/"
    )
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    return run_folder


def generate_exp_name(
    aggregated_param, aggregated_param_text, experiments_per_cell
):
    time_string = (
        datetime.now().isoformat(timespec="seconds").replace(":", "-")
    )
    file_name_string = (
        time_string
        + "_"
        + aggregated_param_text
        + "_"
        + str(aggregated_param)
        + "_"
        + str(experiments_per_cell)
        + "_exp"
    )
    return file_name_string


def write_results_to_folder(result_folder, file_name_string, results):
    json_file_path = write_results_to_json(
        result_folder, file_name_string, results
    )
    generate_reports(json_file_path)
    write_results_to_csv(result_folder, file_name_string, results)


def write_results_to_csv(result_folder, file_name_string, results):
    csv_filen_ame = result_folder + file_name_string + ".csv"
    with open(csv_filen_ame, "w", newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file)
        keys_list = results[0].keys()
        data = [[result[key] for key in keys_list] for result in results]
        csv_file_writer.writerow(keys_list)
        for data_entry in data:
            csv_file_writer.writerow(data_entry)


def write_results_to_json(result_folder, file_name_string, results):
    json_file_name = result_folder + file_name_string + ".json"
    with open(json_file_name, "w") as json_file:
        json.dump(results, json_file)
    return json_file_name


if __name__ == "__main__":

    """ generate report of experiment results from simulation output json file """
    files_to_import = [
        # 'D:/MSc/Thesis/CakeCutting/results/luna/israelMaps02HS_results/IsraelMaps02HS_2019-05-05T15-16-06_NoiseProportion_0.2_15_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/luna/israelMaps04HS_results/IsraelMaps04HS_2019-05-05T14-04-45_NoiseProportion_0.4_15_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/luna/israelMaps06HS_results/IsraelMaps06HS_2019-05-05T12-54-11_NoiseProportion_0.6_15_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps02HS_results/newZealandLowResAgents02HS_2019-05-05T11-19-43_NoiseProportion_0.2_15_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps04HS_results/newZealandLowResAgents04HS_2019-05-05T09-46-34_NoiseProportion_0.4_15_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps06HS_results/newZealandLowResAgents06HS_2019-05-05T08-10-46_NoiseProportion_0.6_15_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/luna/newZealandMaps06_results_full/newZealandLowResAgents06_2019-03-29T07-50-19_NoiseProportion_0.6_50_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/paper_results/newZealand/06hs/newZealandLowResAgents06HS_2019-12-24T17-49-04_NoiseProportion_0.6_50_exp.json',
        # 'D:/MSc/Thesis/CakeCutting/results/paper_results/newZealand/06uni/newZealandLowResAgents06_2019-12-24T17-50-27_NoiseProportion_0.6_50_exp.json',
        "D:/MSc/Thesis/CakeCutting/results/2020-02-08T19-02-59/newZealandLowResAgents06_2020-02-08T19-03-01_NoiseProportion_0.6_50_exp/newZealandLowResAgents06_2020-02-08T19-03-01_NoiseProportion_0.6_50_exp.json"
    ]

    for json_file_name in files_to_import:
        generate_reports(json_file_name)
