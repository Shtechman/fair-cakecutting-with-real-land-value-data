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
    LargestMargin = 9,
    LargestAvgMargin = 10,
    LargestMarginArea = 11,
    MostValuableMargin = 12,
    HighestScatter = 13,
    SquarePiece = 14,
    BruteForce = 15,

