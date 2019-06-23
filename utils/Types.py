#!python3

from enum import Enum


class AggregationType(Enum):
    NoiseProportion = 1,
    NumberOfAgents = 2


class AlgType(Enum):
    EvenPaz = 1


class RunType(Enum):
    Honest = 1,
    Dishonest = 2,
    Assessor = 3


class CutDirection(Enum):
    Horizontal = 1,
    Vertical = 2,
    Both = 3

class CutPattern(Enum):
    Hor = 1,
    Ver = 2,
    HorVer = 3,
    VerHor = 4,
    SmallestHalfCut = 5,
    SmallestPiece = 6,
    LongestDim = 7,
    ShortestDim = 8,
    LargestRemainRange = 9,
    LargestAvgRemainRange = 10,
    LargestRemainArea = 11,
    MostValuableRemain = 12,
    MixedValuableRemain = 13,
    HighestScatter = 14,
    SquarePiece = 15,

