from statistics import mean, stdev

from utils.simulation.cc_types import AggregationType

try:
    from scipy.stats import t

    def calculate_confidence_interval(confidence, stdev, n):
        return stdev * t.ppf((1 + confidence) / 2.0, n - 1)


except:
    # we do this because our remote server does not run scipy.stats
    # we then locally re-run Report Generation on our local machine
    def calculate_confidence_interval(confidence, stdev, n):
        return -1


BRUTEFORCE_KEY = "BruteForce"


def calculate_avg_result(result_list, keys_to_average, groupsize):
    if result_list:
        result = {}
        for key in result_list[0]:
            if key in keys_to_average:
                key_list_values = list(map(lambda res: res[key], result_list))
                avg_key = key + "_Avg"
                std_key = key + "_StDev"
                interval_key = key + "_interval"
                result[avg_key] = mean(key_list_values)
                result[std_key] = stdev(key_list_values)
                result[interval_key] = calculate_confidence_interval(
                    0.95, result[std_key], groupsize
                )
            else:
                result[key] = result_list[-1][key]
                if key == "Method":
                    result[key] = result[key].split("_")[-1]
        return result
    else:
        return {}


def calculate_int_result(algorithm_res, assessor_res, keys_to_integrate):
    if algorithm_res and assessor_res:
        result = {}
        for key in algorithm_res:
            if key in keys_to_integrate:
                if assessor_res[key] == 0:
                    result[key] = "INF"
                else:
                    if "ttc_" in key:
                        assessor_key = key.replace("ttc_", "")
                    else:
                        assessor_key = key
                    result[key] = (
                        algorithm_res[key] - assessor_res[assessor_key]
                    ) / assessor_res[assessor_key]
            else:
                result[key] = algorithm_res[key]
                if key == "Method":
                    result[key] = result[key].split("_")[-1]
                if key == "Algorithm":
                    result[key] = "Integrated_" + algorithm_res[key]
        return result
    else:
        return {}


def parse_sum_data_entry(orig_data_entry, sum_data_entry):
    if not sum_data_entry or not orig_data_entry:
        return []
    dict_to_parse = {}

    for key in orig_data_entry:
        dict_to_parse[key] = orig_data_entry[key]

    dict_to_parse["egalitarianGain_Imp"] = sum_data_entry[
        "egalitarianGain_Avg"
    ]
    dict_to_parse["ttc_egalitarianGain_Imp"] = sum_data_entry[
        "ttc_egalitarianGain_Avg"
    ]
    dict_to_parse["utilitarianGain_Imp"] = sum_data_entry[
        "utilitarianGain_Avg"
    ]
    dict_to_parse["ttc_utilitarianGain_Imp"] = sum_data_entry[
        "ttc_utilitarianGain_Avg"
    ]
    dict_to_parse["averageFaceRatio_Imp"] = sum_data_entry[
        "averageFaceRatio_Avg"
    ]
    dict_to_parse["smallestFaceRatio_Imp"] = sum_data_entry[
        "smallestFaceRatio_Avg"
    ]
    dict_to_parse["largestEnvy_Imp"] = -1 * sum_data_entry["largestEnvy_Avg"]
    dict_to_parse["ttc_largestEnvy_Imp"] = (
        -1 * sum_data_entry["ttc_largestEnvy_Avg"]
    )

    key_order = [
        "Method",
        "egalitarianGain_Avg",
        "egalitarianGain_Imp",
        "egalitarianGain_StDev",
        "ttc_egalitarianGain_Avg",
        "ttc_egalitarianGain_Imp",
        "ttc_egalitarianGain_StDev",
        "utilitarianGain_Avg",
        "utilitarianGain_Imp",
        "utilitarianGain_StDev",
        "ttc_utilitarianGain_Avg",
        "ttc_utilitarianGain_Imp",
        "ttc_utilitarianGain_StDev",
        "averageFaceRatio_Avg",
        "averageFaceRatio_Imp",
        "averageFaceRatio_StDev",
        "smallestFaceRatio_Avg",
        "smallestFaceRatio_Imp",
        "smallestFaceRatio_StDev",
        "largestEnvy_Avg",
        "largestEnvy_Imp",
        "largestEnvy_StDev",
        "ttc_largestEnvy_Avg",
        "ttc_largestEnvy_Imp",
        "ttc_largestEnvy_StDev",
        "experimentDurationSec_Avg",
        "experimentDurationSec_StDev",
    ]

    return [dict_to_parse[key] for key in key_order]


def create_table_summary_line(groupsize_sum_results):
    max_idx_dict = {
        "EGA": 1,
        "EGP": 2,
        "ttcEGA": 4,
        "ttcEGP": 5,
        "UGA": 7,
        "UGP": 8,
        "ttcUGA": 10,
        "ttcUGP": 11,
        "AFA": 13,
        "AFP": 14,
        "SFA": 16,
        "SFP": 17,
    }
    min_idx_dict = {
        "LEA": 19,
        "LEP": 20,
        "ttcLEA": 22,
        "ttcLEP": 23,
        "RDA": 25,
    }
    groupsize_sum_results = [r for r in groupsize_sum_results if r]
    sum_result = [" "] * 50
    if groupsize_sum_results:
        sum_result[0] = "Summary"

        for idx in max_idx_dict.values():
            sum_result[idx] = next(
                (
                    r[0]
                    for r in groupsize_sum_results
                    if r
                    and (
                        r[idx]
                        == max([r[idx] for r in groupsize_sum_results if r])
                    )
                ),
                " ",
            )

        for idx in min_idx_dict.values():
            sum_result[idx] = next(
                (
                    r[0]
                    for r in groupsize_sum_results
                    if r
                    and (
                        r[idx]
                        == min([r[idx] for r in groupsize_sum_results if r])
                    )
                ),
                " ",
            )
    return sum_result


def parse_method_over_measure_data_entry(
    measure,
    method,
    available_groupsizes,
    hon_avg,
    hon_interval,
    as_avg,
    as_interval,
    ttc_avg,
    ttc_interval,
    selling=1.0,
):
    """ header is ['NumberOfAgents', 'Honest', 'Assessor', 'TTC',  'Selling', 'Hon_Conf', 'As_Conf', 'TTC_Conf'] """
    data_entry = []
    for idx, group in enumerate(available_groupsizes):
        data_entry.append(
            [
                group,
                hon_avg[idx],
                as_avg[idx],
                ttc_avg[idx],
                selling,
                hon_interval[idx],
                as_interval[idx],
                ttc_interval[idx],
            ]
        )
    return data_entry


def preprocess_results(results, assessor_results, highestbidder_results):
    keys_to_average = [
        "egalitarianGain",
        "ttc_egalitarianGain",
        "ttc_utilitarianGain",
        "utilitarianGain",
        "averageFaceRatio",
        "smallestFaceRatio",
        "largestEnvy",
        "ttc_largestEnvy",
        "experimentDurationSec",
    ]
    keys_to_integrate = [key + "_Avg" for key in keys_to_average]
    interval_keys = [key + "_interval" for key in keys_to_average]
    available_groupsizes = [
        4,
        8,
        16,
        32,
        64,
        128,
    ]  # todo make dynamic to group size

    # # for old versions format
    # def _fix_parse(res):
    #     m = res['Method'].replace('ValuableRemain', 'ValuableMargin').replace('RemainRange', 'Margin')
    #     if "Assessor" in m:
    #         res['Method'] = m.replace('Assessor', 'Assessor_EvenPaz_')
    #         res['Algorithm'] = 'Assessor_EvenPaz'
    #     if "Honest" in m:
    #         res['Method'] = m
    #         res['Algorithm'] = 'Honest_EvenPaz'
    #     return res
    #
    # results = [_fix_parse(r) for r in results]

    dishonest_gain = get_dishonest_gain(results)
    highestbidder_gain = get_highestbidder_gain(results, highestbidder_results)

    method_list = [r["Method"].split("_")[-1] for r in results]
    res_per_m = {  # m - method
        method: [r for r in results if r["Method"].split("_")[-1] == method]
        for method in method_list
    }

    hbidder_key, assessor_key, dishonest_key, honest_key,\
    int_dishonest_key, int_honest_key, run_types_keys = extract_keys(results)

    assessor_res_per_gs = group_results_by_groupsize(available_groupsizes,assessor_results)

    hbidder_res_per_gs = group_results_by_groupsize(available_groupsizes, highestbidder_results)

    res_per_a_per_gs_per_m = group_results_per_method(available_groupsizes, res_per_m, run_types_keys)

    best_result_per_gs = extract_specific_results_by_groupsize(res_per_a_per_gs_per_m, BRUTEFORCE_KEY)

    avg_results_per_method = average_results_per_method(hbidder_key, assessor_key, dishonest_key, honest_key,
                                                        int_dishonest_key, int_honest_key,
                                                        keys_to_average, keys_to_integrate,
                                                        hbidder_res_per_gs,
                                                        assessor_res_per_gs,
                                                        res_per_a_per_gs_per_m)

    sum_honest_results_per_groupsize = sum_groupsize_results(avg_results_per_method,
                                                             honest_key, int_honest_key)

    sum_dishonest_results_per_groupsize = sum_groupsize_results(avg_results_per_method,
                                                                dishonest_key, int_dishonest_key)

    avg_honest_results_per_measurement = reorder_results_per_measure(avg_results_per_method, honest_key,
                                                                     keys_to_integrate)

    interval_honest_results_per_measurement = reorder_results_per_measure(avg_results_per_method, honest_key,
                                                                          interval_keys)

    avg_assessor_results_per_measurement = reorder_results_per_measure(avg_results_per_method, assessor_key,
                                                                       keys_to_integrate)

    interval_assessor_results_per_measurement = reorder_results_per_measure(avg_results_per_method, assessor_key,
                                                                            interval_keys)

    avg_dishonest_results_per_measurement = reorder_results_per_measure(avg_results_per_method, dishonest_key,
                                                                        keys_to_integrate)

    interval_dishonest_results_per_measurement = reorder_results_per_measure(avg_results_per_method, dishonest_key,
                                                                             interval_keys)

    graph_method_results_per_measure = format_results_for_graph(available_groupsizes,
                                                                avg_assessor_results_per_measurement,
                                                                avg_honest_results_per_measurement,
                                                                avg_results_per_method,
                                                                interval_assessor_results_per_measurement,
                                                                interval_honest_results_per_measurement,
                                                                keys_to_average)

    return (
        avg_results_per_method,
        sum_honest_results_per_groupsize,
        sum_dishonest_results_per_groupsize,
        avg_honest_results_per_measurement,
        avg_assessor_results_per_measurement,
        avg_dishonest_results_per_measurement,
        graph_method_results_per_measure,
        available_groupsizes,
        dishonest_gain,
        best_result_per_gs,
        highestbidder_gain,
    )


def format_results_for_graph(available_groupsizes, avg_assessor_results_per_measurement,
                             avg_honest_results_per_measurement, avg_results_per_method,
                             interval_assessor_results_per_measurement, interval_honest_results_per_measurement,
                             keys_to_average):
    graph_method_results_per_measure = {
        measure: {} for measure in keys_to_average
    }
    for measure in graph_method_results_per_measure:
        if "ttc" in measure:
            continue
        avg_key = measure + "_Avg"
        ttc_avg_key = "ttc_" + avg_key
        ass_avg_key = avg_key
        interval_key = measure + "_interval"
        ttc_interval_key = "ttc_" + interval_key
        ass_int_key = interval_key
        honest_avg = avg_honest_results_per_measurement[avg_key]
        if ttc_avg_key in avg_honest_results_per_measurement:
            ttc_honest_avg = avg_honest_results_per_measurement[ttc_avg_key]
        else:
            ttc_honest_avg = {
                method: [""] * len(available_groupsizes)
                for method in avg_results_per_method
            }
        assessor_avg = avg_assessor_results_per_measurement[ass_avg_key]
        honest_interval = interval_honest_results_per_measurement[interval_key]
        if ttc_interval_key in interval_honest_results_per_measurement:
            ttc_honest_interval = interval_honest_results_per_measurement[
                ttc_interval_key
            ]
        else:
            ttc_honest_interval = {
                method: [""] * len(available_groupsizes)
                for method in avg_results_per_method
            }
        assessor_interval = interval_assessor_results_per_measurement[
            ass_int_key
        ]
        for method in avg_results_per_method:
            graph_method_results_per_measure[measure][
                method
            ] = parse_method_over_measure_data_entry(
                measure,
                method,
                available_groupsizes,
                honest_avg[method],
                honest_interval[method],
                assessor_avg[method],
                assessor_interval[method],
                ttc_honest_avg[method],
                ttc_honest_interval[method],
            )
    return graph_method_results_per_measure


def extract_keys(results):
    run_types_keys = list(set([r["Algorithm"] for r in results]))
    hbidder_key = "HighestBidder"
    assessor_key = "Assessor"
    honest_key = next(
        (key for key in run_types_keys if "Honest" in key), "HonestEmpty"
    )
    dishonest_key = next(
        (key for key in run_types_keys if "Dishonest" in key), "DishonestEmpty"
    )
    int_honest_key = "{}_{}".format("Integrated", honest_key)
    int_dishonest_key = "{}_{}".format("Integrated", dishonest_key)
    return hbidder_key, assessor_key, dishonest_key, honest_key, int_dishonest_key, int_honest_key, run_types_keys


def reorder_results_per_measure(avg_results_per_method, value_type_key, measures_keys):
    return {
        measure: {
            method: [
                avg_results_per_method[method][groupsize][value_type_key][measure]
                if avg_results_per_method[method][groupsize][value_type_key]
                else ""
                for groupsize in avg_results_per_method[method]
            ]
            for method in avg_results_per_method
        }
        for measure in measures_keys
    }


def sum_groupsize_results(avg_results_per_method, value_key, int_value_key):
    results = {groupsize: [
        parse_sum_data_entry(
            method_avg_results[groupsize][value_key],
            method_avg_results[groupsize][int_value_key],
        )
        for method_avg_results in avg_results_per_method.values()
    ]
        for groupsize in list(avg_results_per_method.values())[0]
    }

    for groupSize in results:
        if not results[groupSize]:
            continue
        results[groupSize].append(create_table_summary_line(results[groupSize]))

    return results


def average_results_per_method(hbidder_key, assessor_key, dishonest_key, honest_key, int_dishonest_key, int_honest_key,
                               keys_to_average, keys_to_integrate, hbidder_res_per_gs, assessor_res_per_gs, res_per_a_per_gs_per_m):
    avg_results_per_method = {}
    for method in res_per_a_per_gs_per_m:
        method_avg_results = {}

        for groupsize in res_per_a_per_gs_per_m[method]:
            method_avg_results[groupsize] = {
                a: calculate_avg_result(
                    res_per_a_per_gs_per_m[method][groupsize][a],
                    keys_to_average,
                    groupsize,
                )
                for a in res_per_a_per_gs_per_m[method][groupsize]
            }
            method_avg_results[groupsize][assessor_key] = calculate_avg_result(
                assessor_res_per_gs[groupsize], keys_to_average, groupsize
            )

            method_avg_results[groupsize][hbidder_key] = calculate_avg_result(
                hbidder_res_per_gs[groupsize], keys_to_average, groupsize
            )

            for key in [assessor_key, honest_key, dishonest_key, hbidder_key]:
                if key not in method_avg_results[groupsize]:
                    method_avg_results[groupsize][key] = {}

            assessor_res = method_avg_results[groupsize][assessor_key]

            honest_res = method_avg_results[groupsize][honest_key]
            method_avg_results[groupsize][
                int_honest_key
            ] = calculate_int_result(
                honest_res, assessor_res, keys_to_integrate
            )

            dishonest_res = method_avg_results[groupsize][dishonest_key]
            method_avg_results[groupsize][
                int_dishonest_key
            ] = calculate_int_result(
                dishonest_res, assessor_res, keys_to_integrate
            )

        # todo - implement parser for graph summary file
        avg_results_per_method[method] = method_avg_results
    return avg_results_per_method


def extract_specific_results_by_groupsize(res_per_a_per_gs_per_m, method_key):
    specific_result_per_gs = {}
    if method_key in res_per_a_per_gs_per_m:
        keys = [
            ("egalitarianGain", max),
            ("ttc_egalitarianGain", max),
            ("ttc_utilitarianGain", max),
            ("utilitarianGain", max),
            ("averageFaceRatio", max),
            ("smallestFaceRatio", max),
            ("largestEnvy", min),
            ("ttc_largestEnvy", min),
            ("experimentDurationSec", min),
        ]
        for groupsize in res_per_a_per_gs_per_m[method_key]:
            honest_key = next(
                (
                    key
                    for key in res_per_a_per_gs_per_m[method_key][
                    groupsize
                ].keys()
                    if "Honest" in key
                ),
                None,
            )
            if not honest_key:
                continue
            if res_per_a_per_gs_per_m[method_key][groupsize][honest_key]:
                uniqe_exp_id = list(
                    set(
                        [
                            r["experiment"]
                            for r in res_per_a_per_gs_per_m[method_key][
                            groupsize
                        ][honest_key]
                        ]
                    )
                )
                specific_result_per_gs[groupsize] = []
                for key in keys:
                    cur_key_list = []
                    for exp_id in uniqe_exp_id:
                        exp_results = [
                            r
                            for r in res_per_a_per_gs_per_m[method_key][
                                groupsize
                            ][honest_key]
                            if r["experiment"] == exp_id
                        ]
                        cur_key_list.append(
                            key[1](exp_results, key=lambda r: r[key[0]])[
                                key[0]
                            ]
                        )

                    specific_result_per_gs[groupsize].append(
                        [key[0], mean(cur_key_list), stdev(cur_key_list)]
                    )
    return specific_result_per_gs


def group_results_per_method(available_groupsizes, res_per_m, run_types_keys):
    res_per_gs_per_m = {}  # gs - groupsize, m - method
    res_per_a_per_gs_per_m = {}  # a - algorithm, gs - groupsize, m - method
    for method in res_per_m:
        res_per_gs_per_m[method] = group_results_by_groupsize(available_groupsizes, res_per_m[method])

        res_per_a_per_gs_per_m[method] = {n: [] for n in available_groupsizes}
        for groupsize in res_per_gs_per_m[method]:
            res_per_a_per_gs_per_m[method][groupsize] = {
                a: [
                    r
                    for r in res_per_gs_per_m[method][groupsize]
                    if r["Algorithm"] == a
                ]
                for a in run_types_keys
            }
    return res_per_a_per_gs_per_m


def group_results_by_groupsize(available_groupsizes, res_per_m):
    return {
        n: [r for r in res_per_m if r["NumberOfAgents"] == n]
        for n in available_groupsizes
    }


def _get_gain_over_fair(fair_partition, unfair_partition, agent_of_interest, loss=False):
    unfair_value = unfair_partition[agent_of_interest]
    fair_value = fair_partition[agent_of_interest]
    return unfair_value - fair_value, _get_percentage_gain(fair_value, unfair_value, loss)


def _get_percentage_gain(fair_value, unfair_value, loss):
    ref_value = unfair_value if loss and unfair_value > 0 else fair_value
    # todo: when "loss" calculation is requested and unfair_value is 0, we return gain instead to avoid dividing by 0.
    return (unfair_value - fair_value) / ref_value


# todo: refactor, this function is almost the same as get_dishonest_gain..
def get_highestbidder_gain(results, highestbidder_results):
    experiment_list = list(set([rlog["experiment"] for rlog in results]))
    cut_pattern_list = list(
        set(
            [
                rlog["Method"].replace("Dishonest_", "").replace("Honest_", "")
                for rlog in results
            ]
        )
    )
    num_agents = list(
        set([rlog[AggregationType.NumberOfAgents.name] for rlog in results])
    )

    result = {numA: {} for numA in num_agents}
    if not highestbidder_results:
        return result

    hb_rlogs = {exp: [rlog for rlog in highestbidder_results if rlog["experiment"] == exp][0]
                for exp in experiment_list}

    for exp in experiment_list:
        numA = [rlog for rlog in results if rlog["experiment"] == exp][0][
            AggregationType.NumberOfAgents.name
        ]
        result[numA][exp] = {}
        for cp in cut_pattern_list:
            result[numA][exp][cp] = []
            relevant_logs = [
                rlog
                for rlog in results
                if (
                    rlog["experiment"] == exp
                    and rlog["Method"]
                    .replace("Dishonest_", "")
                    .replace("Honest_", "")
                    == cp
                )
            ]

            highestbidder_partition = hb_rlogs[exp]["relativeValues"]
            highestbidder_uv = hb_rlogs[exp]["ttc_utilitarianGain"]
            honest = [
                rlog
                for rlog in relevant_logs
                if rlog["dishonestAgent"] is None
            ][0]
            honest_partitions = honest["relativeValues"]
            honest_uv = honest["ttc_utilitarianGain"]
            # for agent in highestbidder_partition:
            #     v, p = _get_gain_over_fair(
            #         honest_partitions, highestbidder_partition, agent
            #     )
            #
            #     result[numA][exp][cp].append(
            #         {"agent": agent, "agent_gain": v, "agent_gain_per": p}
            #     )
            pof = _get_percentage_gain(honest_uv, highestbidder_uv, True)
            result[numA][exp][cp].append(
                {"agent": "POF", "agent_gain": highestbidder_uv-honest_uv, "agent_gain_per": pof}
            )

    return result


def calc_dishonest_worth(func):
    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)

        for numA in result:
            results_per_exp = result[numA]
            num_experiments = len(results_per_exp)*1.0
            loss_count = {}
            for exp in results_per_exp:
                for cp in results_per_exp[exp]:
                    for dishonest_agent in results_per_exp[exp][cp]:
                        if cp not in loss_count:
                            loss_count[cp] = 0
                        if dishonest_agent['agent_gain'] < 0:
                            loss_count[cp] += 1
            print("For %s agents - average number of loses while playing dishonest:" % numA)
            for cp in loss_count:
                print(
                    "%s, %s" % (cp.split("_")[-1], loss_count[cp]/num_experiments))

        return result

    return wrap


@calc_dishonest_worth
def get_dishonest_gain(results):
    experiment_list = list(set([rlog["experiment"] for rlog in results]))
    cut_pattern_list = list(
        set(
            [
                rlog["Method"].replace("Dishonest_", "").replace("Honest_", "")
                for rlog in results
            ]
        )
    )
    num_agents = list(
        set([rlog[AggregationType.NumberOfAgents.name] for rlog in results])
    )

    result = {numA: {} for numA in num_agents}

    for exp in experiment_list:
        numA = [rlog for rlog in results if rlog["experiment"] == exp][0][
            AggregationType.NumberOfAgents.name
        ]
        result[numA][exp] = {}
        for cp in cut_pattern_list:
            result[numA][exp][cp] = []
            relevant_logs = [
                rlog
                for rlog in results
                if (
                    rlog["experiment"] == exp
                    and rlog["Method"]
                    .replace("Dishonest_", "")
                    .replace("Honest_", "")
                    == cp
                )
            ]
            dishonest_partitions = {
                rlog["dishonestAgent"]: rlog["relativeValues"]
                for rlog in relevant_logs
                if rlog["dishonestAgent"] is not None
            }
            honest = [
                rlog
                for rlog in relevant_logs
                if rlog["dishonestAgent"] is None
            ][0]
            honest_partitions = honest["relativeValues"]
            for agent in dishonest_partitions:
                v, p = _get_gain_over_fair(
                    honest_partitions, dishonest_partitions[agent], agent
                )

                result[numA][exp][cp].append(
                    {"agent": agent, "agent_gain": v, "agent_gain_per": p}
                )

    return result
