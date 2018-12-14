from utils.Types import CutPattern, CutDirection
import numpy as np


class Cutter:

    def __init__(self, cut_pattern, cut_query=None, cut_direction=None):
        self.cut_pattern = cut_pattern
        self.cut_query = cut_query
        self.cut_direction = cut_direction

    def __copy__(self):
        return Cutter(self.cut_pattern, self.cut_query, self.cut_direction)

    def allocate_cuts(self, allocations, number_of_agents):
        number_of_agents_in_first_partition = int(np.ceil(number_of_agents/2))
        proportion_of_first_partition = number_of_agents_in_first_partition / float(number_of_agents)

        self._set_next_query_direction()

        # Ask all agents a "cut" query - cut the cake in proportionOfFirstPartition (half or near-half):
        for allocation in allocations:
            value = proportion_of_first_partition*allocation.getValue()
            self._set_relevant_halfcuts(value, allocation)

        self._set_next_cutting_direction(allocations)

        # Calculate the median of the agents' half-cuts: this will be our cut location.
        allocations.sort(key=lambda alloc: alloc.halfcuts[self.cut_direction])
        end_of_first_part = allocations[number_of_agents_in_first_partition-1].halfcuts[self.cut_direction]
        start_of_second_part = allocations[number_of_agents_in_first_partition].halfcuts[self.cut_direction]
        cut_location = (end_of_first_part+start_of_second_part)/2

        # Divide the agents to two groups of nearly the same size, based on their half-cut locations:
        first_part_allocations = []
        second_part_allocations = []
        #print("cut allocation for %s agents" % number_of_agents)
        for i in range(0, number_of_agents_in_first_partition):
            iFrom = allocations[i].getDirectionaliFrom(self.cut_direction)
            iTo = cut_location
            oppIndexes = allocations[i].getOppositeDirectionalRange(self.cut_direction)
            #print("\t%s cutting piece %s at [%s,%s]-(opp[%s,%s])" % (self.cut_direction, i, iFrom, iTo, oppIndexes[0], oppIndexes[1]))
            first_part_allocations.append(allocations[i].subCut(iFrom, iTo, self.cut_direction))

        for i in range(number_of_agents_in_first_partition,  number_of_agents):
            iFrom = cut_location
            iTo = allocations[i].getDirectionaliTo(self.cut_direction)
            oppIndexes = allocations[i].getOppositeDirectionalRange(self.cut_direction)
            #print("\t%s cutting piece %s at [%s,%s]-(opp[%s,%s])" % (self.cut_direction, i, iFrom, iTo, oppIndexes[0], oppIndexes[1]))
            second_part_allocations.append(allocations[i].subCut(iFrom, iTo, self.cut_direction))

        return first_part_allocations, second_part_allocations

    def _set_relevant_halfcuts(self, value, allocation):
        allocation.halfcuts = {}
        if (self.cut_query is CutDirection.Horizontal) or (self.cut_query is CutDirection.Both):
            allocation.halfcuts[CutDirection.Horizontal] = allocation.markQuery(value, CutDirection.Horizontal)
        if (self.cut_query is CutDirection.Vertical) or (self.cut_query is CutDirection.Both):
            allocation.halfcuts[CutDirection.Vertical] = allocation.markQuery(value, CutDirection.Vertical)


    def _initial_cut_direction(self):
        switcher = {
            CutPattern.Hor: CutDirection.Horizontal,
            CutPattern.Ver: CutDirection.Vertical,
            CutPattern.HorVer: CutDirection.Horizontal,
            CutPattern.VerHor: CutDirection.Vertical,
        }

        return switcher.get(self.cut_pattern, CutDirection.Both)

    def _cut_pattern_same_query_direction(self):
        return self.cut_query

    def _cut_pattern_opp_query_direction(self):
        switcher = {
            CutDirection.Horizontal: CutDirection.Vertical,
            CutDirection.Vertical: CutDirection.Horizontal,
        }

        return switcher.get(self.cut_direction, CutDirection.Both)

    def _get_query_direction_iterator_func(self):
        switcher = {
            CutPattern.HorVer: self._cut_pattern_opp_query_direction,
            CutPattern.VerHor: self._cut_pattern_opp_query_direction,
        }

        return switcher.get(self.cut_pattern, self._cut_pattern_same_query_direction)

    def _set_next_query_direction(self):
        if self.cut_query is None:
            self.cut_query = self._initial_cut_direction()
        else:
            dir_iter_func = self._get_query_direction_iterator_func()
            self.cut_query = dir_iter_func()

    def _set_next_cutting_direction(self, allocations):
        if self.cut_pattern is CutPattern.SmallestHalfCut:
            first_hor_cut = min(allocations, key=lambda alloc: alloc.halfcuts[CutDirection.Horizontal])
            first_ver_cut = min(allocations, key=lambda alloc: alloc.halfcuts[CutDirection.Vertical])
            first_hor_cut_halfcut = first_hor_cut.halfcuts[CutDirection.Horizontal]
            first_ver_cut_halfcut = first_ver_cut.halfcuts[CutDirection.Vertical]
            if first_hor_cut_halfcut < first_ver_cut_halfcut:
                self.cut_direction = CutDirection.Horizontal
                #print("%s because - %s < %s" % (self.cut_direction, first_hor_cut_halfcut, first_ver_cut_halfcut))
            else:
                self.cut_direction = CutDirection.Vertical
                #print("%s because - %s < %s" % (self.cut_direction, first_ver_cut_halfcut, first_hor_cut_halfcut))
            return
        if self.cut_pattern is CutPattern.SmallestPiece:
            first_hor_cut = min(allocations, key=lambda alloc: alloc.halfcuts[CutDirection.Horizontal])
            first_ver_cut = min(allocations, key=lambda alloc: alloc.halfcuts[CutDirection.Vertical])
            first_hor_cut_halfcut = first_hor_cut.halfcuts[CutDirection.Horizontal]
            first_ver_cut_halfcut = first_ver_cut.halfcuts[CutDirection.Vertical]
            start_index_hor = first_hor_cut.getDirectionaliFrom(CutDirection.Horizontal)
            start_index_ver = first_ver_cut.getDirectionaliFrom(CutDirection.Vertical)

            hor_cut_size = (first_hor_cut_halfcut-start_index_hor)*first_ver_cut.getDimensions()[CutDirection.Vertical]
            ver_cut_size = (first_ver_cut_halfcut-start_index_ver)*first_hor_cut.getDimensions()[CutDirection.Horizontal]

            if hor_cut_size < ver_cut_size:
                self.cut_direction = CutDirection.Horizontal
                #print("%s because - %s < %s" % (self.cut_direction, hor_cut_size, ver_cut_size))
            else:
                self.cut_direction = CutDirection.Vertical
                #print("%s because - %s < %s" % (self.cut_direction, ver_cut_size, hor_cut_size))
            return
        if self.cut_pattern is CutPattern.LongestDim:
            allocation = allocations[0]
            allocation_dimensions = allocation.getDimensions()

            if allocation_dimensions[CutDirection.Vertical] < allocation_dimensions[CutDirection.Horizontal]:
                self.cut_direction = CutDirection.Horizontal
                #print("%s because - %s < %s" % (self.cut_direction, allocation_dimensions[CutDirection.Vertical],
                #                            allocation_dimensions[CutDirection.Horizontal]))
            else:
                self.cut_direction = CutDirection.Vertical
                #print("%s because - %s < %s" % (self.cut_direction, allocation_dimensions[CutDirection.Horizontal],
                #                            allocation_dimensions[CutDirection.Vertical]))
            return
        if self.cut_pattern is CutPattern.ShortestDim:
            allocation = allocations[0]
            allocation_dimensions = allocation.getDimensions()

            if allocation_dimensions[CutDirection.Horizontal] < allocation_dimensions[CutDirection.Vertical]:
                self.cut_direction = CutDirection.Horizontal
                #print("%s because - %s < %s" % (self.cut_direction, allocation_dimensions[CutDirection.Horizontal],
                #                            allocation_dimensions[CutDirection.Vertical]))
            else:
                self.cut_direction = CutDirection.Vertical
                #print("%s because - %s < %s" % (self.cut_direction, allocation_dimensions[CutDirection.Vertical],
                #                            allocation_dimensions[CutDirection.Horizontal]))
            return
        else:
            self.cut_direction = self.cut_query
