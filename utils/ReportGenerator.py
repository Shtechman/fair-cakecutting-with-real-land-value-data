from statistics import mean, stdev
from utils.Types import AggregationType, CutPattern

try:
    from scipy.stats import t

    def calculate_confidence_interval(confidence, stdev, n):
        return stdev * t.ppf((1 + confidence) / 2., n - 1)
except:
    # we do this because our remote server does not run scipy.stats
    # we then locally re-run Report Generation on our local machine
    def calculate_confidence_interval(confidence, stdev, n):
        return -1


# todo - modify to support multiple algorithm types and not just 'Even and Paz'
HONEST_KEY = "Honest_EvenPaz"
DISHONEST_KEY = "Dishonest_EvenPaz"
ASSESSOR_KEY = "Assessor_EvenPaz"
INT_HONEST_KEY = "Integrated_Honest_EvenPaz"
INT_DISHONEST_KEY = "Integrated_Dishonest_EvenPaz"
BRUTEFORCE_KEY = "BruteForce"
EXTENDED_KEYS = [HONEST_KEY,DISHONEST_KEY,ASSESSOR_KEY,INT_HONEST_KEY,INT_DISHONEST_KEY]


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
                    result[key] = (algorithm_res[key] - assessor_res[assessor_key]) / assessor_res[assessor_key]
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

    # # for old versions format
    # def _fix_parse(res):
    #     m = res['Method']
    #     if "Assessor" in m:
    #         res['Method'] = res['Method'].replace('Assessor','Assessor_EvenPaz_')
    #         res['Algorithm'] = 'Assessor_EvenPaz'
    #     if "Honest" in m:
    #         res['Algorithm'] = 'Honest_EvenPaz'
    #     return res
    #
    # results = [_fix_parse(r) for r in results]

    dishonestGain = get_dishonest_gain(results)

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
                                                             if r['Algorithm'] == a] for a in [HONEST_KEY, ASSESSOR_KEY, DISHONEST_KEY]}
    avg_results_per_method = {}
    sum_honest_results_per_groupsize = {n: [] for n in available_groupsizes}
    sum_dishonest_results_per_groupsize = {n: [] for n in available_groupsizes}

    best_result_per_gs = {}
    if BRUTEFORCE_KEY in res_per_a_per_gs_per_m:
        keys = [('egalitarianGain',max),
                ('ttc_egalitarianGain',max),
                ('ttc_utilitarianGain',max),
                ('utilitarianGain',max),
                ('averageFaceRatio',max),
                ('smallestFaceRatio',max),
                ('largestEnvy',min),
                ('ttc_largestEnvy',min),
                ('experimentDurationSec',min)]
        for groupsize in res_per_a_per_gs_per_m[BRUTEFORCE_KEY]:
            if res_per_a_per_gs_per_m[BRUTEFORCE_KEY][groupsize][HONEST_KEY]:
                uniqe_exp_id = list(set([r['experiment'] for r in res_per_a_per_gs_per_m[BRUTEFORCE_KEY][groupsize][HONEST_KEY]]))
                best_result_per_gs[groupsize] = []
                for key in keys:
                    cur_key_list = []
                    for exp_id in uniqe_exp_id:
                        exp_results = [r for r in res_per_a_per_gs_per_m[BRUTEFORCE_KEY][groupsize][HONEST_KEY]
                                       if r['experiment'] == exp_id]
                        cur_key_list.append(key[1](exp_results, key=lambda r: r[key[0]])[key[0]])

                    best_result_per_gs[groupsize].append([key[0], mean(cur_key_list), stdev(cur_key_list)])

    for method in res_per_a_per_gs_per_m:
        method_avg_results = {}
        
        for groupsize in res_per_a_per_gs_per_m[method]:
            method_avg_results[groupsize] = {a: calculate_avg_result(res_per_a_per_gs_per_m[method][groupsize][a],
                                                                     keys_to_average, groupsize)
                                             for a in res_per_a_per_gs_per_m[method][groupsize]}
            Honest_res = method_avg_results[groupsize][HONEST_KEY]
            Assessor_res = method_avg_results[groupsize][ASSESSOR_KEY]
            Dishonest_res = method_avg_results[groupsize][DISHONEST_KEY]
            method_avg_results[groupsize][INT_HONEST_KEY] = calculate_int_result(Honest_res, Assessor_res,
                                                                             keys_to_integrate)
            method_avg_results[groupsize][INT_DISHONEST_KEY] = calculate_int_result(Dishonest_res, Assessor_res,
                                                                             keys_to_integrate)
            sum_honest_results_per_groupsize[groupsize].append(
                parse_sum_data_entry(method_avg_results[groupsize][HONEST_KEY], method_avg_results[groupsize][INT_HONEST_KEY]))
            sum_dishonest_results_per_groupsize[groupsize].append(
                parse_sum_data_entry(method_avg_results[groupsize][DISHONEST_KEY], method_avg_results[groupsize][INT_DISHONEST_KEY]))

        # todo - implement parser for graph summary file
        avg_results_per_method[method] = method_avg_results

    avg_honest_results_per_measurement = {measure:
                                       {method:
                                            [avg_results_per_method[method][groupsize][HONEST_KEY][measure]
                                             if avg_results_per_method[method][groupsize][HONEST_KEY] else ""
                                             for groupsize in avg_results_per_method[method]]
                                        for method in avg_results_per_method}
                                   for measure in keys_to_integrate}

    interval_honest_results_per_measurement = {measure:
                                               {method:
                                                    [avg_results_per_method[method][groupsize][HONEST_KEY][measure]
                                                     if avg_results_per_method[method][groupsize][HONEST_KEY] else ""
                                                     for groupsize in avg_results_per_method[method]]
                                                for method in avg_results_per_method}
                                           for measure in interval_keys}

    avg_assessor_results_per_measurement = {measure:
                                       {method:
                                            [avg_results_per_method[method][groupsize][ASSESSOR_KEY][measure]
                                            if avg_results_per_method[method][groupsize][ASSESSOR_KEY] else ""
                                             for groupsize in avg_results_per_method[method]]
                                        for method in avg_results_per_method}
                                   for measure in keys_to_integrate}


    interval_assessor_results_per_measurement = {measure:
                                                {method:
                                                     [avg_results_per_method[method][groupsize][ASSESSOR_KEY][measure]
                                                      if avg_results_per_method[method][groupsize][ASSESSOR_KEY] else ""
                                                      for groupsize in avg_results_per_method[method]]
                                                 for method in avg_results_per_method}
                                            for measure in interval_keys}

    avg_dishonest_results_per_measurement = {measure:
                                                {method:
                                                     [avg_results_per_method[method][groupsize][DISHONEST_KEY][measure]
                                                      if avg_results_per_method[method][groupsize][DISHONEST_KEY] else ""
                                                      for groupsize in avg_results_per_method[method]]
                                                 for method in avg_results_per_method}
                                            for measure in keys_to_integrate}

    interval_dishonest_results_per_measurement = {measure:
                                                     {method:
                                                          [avg_results_per_method[method][groupsize][DISHONEST_KEY][measure]
                                                           if avg_results_per_method[method][groupsize][
                                                              DISHONEST_KEY] else ""
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

