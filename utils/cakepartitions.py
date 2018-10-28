#!python3

import math
# from utils.ExpMeasures import ExpMeasures
# from evenpaz.allocation import Allocation
# from utils import numbersUtil as Numbersutil
# from utils import calcOperations

__author__ = 'gabib3b'


def agent_value(values, from_index, to_index):
    return calcOperations.sum_values(values, from_index, to_index)

def agent_value_for_indexes(allocation, from_index, to_index):
    return allocation.agent_value_for_indexes(from_index, to_index)

def alg_division_to_measures(division, validate = True):

    egalitarianGain = Numbersutil.normalizedEgalitarianValue(division) -1

    if  validate and  egalitarianGain <- 0.001:
        raise  Exception ("In proportional division, normalized egalitarian gain must be at least 0; got "+egalitarianGain)

    utilitarianGain = Numbersutil.utilitarianValue(division) -1

    if  validate and  utilitarianGain <- 0.001:
        raise  Exception ("In proportional division, utilitarian gain must be at least 0; got "+utilitarianGain)

    envy = largest_envy(division)

    return ExpMeasures(egalitarianGain, utilitarianGain, envy)
