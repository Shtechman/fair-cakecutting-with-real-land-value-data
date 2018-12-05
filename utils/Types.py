#!python3

from enum import Enum


class AggregationType(Enum):
    NoiseProportion = 1,
    NumberOfAgents = 2


class AlgType(Enum):
    EvenPaz = 1,
    Assessor = 2


class CutDirection(Enum):
    Horizontal = 1,
    Vertical = 2

class CutPattern(Enum):
    Hor = 1,
    Ver = 2,
    HorVer = 3,
    VerHor = 4
