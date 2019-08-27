import sys

from utils.Types import CutPattern, CutDirection
import numpy as np

SMALLEST_NUMBER = 0.0000000000001


class Cutter:

    def __init__(self, cut_pattern, cut_query=None, cut_direction=None, next_dir_index=-1):
        self.cut_pattern = cut_pattern
        self.next_dir_index = next_dir_index
        self.cut_query = cut_query
        self.cut_direction = cut_direction
        self.freeplay_mode = type(cut_pattern) is list

    def __copy__(self):
        return Cutter(self.cut_pattern, self.cut_query, self.cut_direction, self.next_dir_index)

    def allocate_cuts(self, allocations, number_of_agents):
        number_of_agents_in_first_partition = int(np.ceil(number_of_agents/2))
        proportion_of_first_partition = number_of_agents_in_first_partition / float(number_of_agents)

        self._set_next_query_direction()

        # Ask all agents a "cut" query - cut the cake in proportionOfFirstPartition (half or near-half):
        dishonestIdx = -1
        for idx, allocation in enumerate(allocations):
            if dishonestIdx < 0 and allocation.getAgent().isDishonest():
                dishonestIdx = idx
            else:
                value = proportion_of_first_partition*allocation.getValue()
                self._set_relevant_halfcuts(value, allocation)

        if dishonestIdx > -1:
            allocation = allocations[dishonestIdx]
            honestAllocations = allocations[:dishonestIdx]+allocations[dishonestIdx+1:]
            self._best_halfcuts(allocation, honestAllocations)

        self._set_next_cutting_direction(allocations, number_of_agents)

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

    def _best_halfcuts(self, allocation, honestAllocations):
        allocation.halfcuts = {}
        if (self.cut_query is CutDirection.Horizontal) or (self.cut_query is CutDirection.Both):
            horizontalHosnestHalfcuts = [allocation.halfcuts[CutDirection.Horizontal] for allocation in
                                         honestAllocations]
            allocation.halfcuts[CutDirection.Horizontal] = self._calculate_best_halfcut(allocation, CutDirection.Horizontal,
                                                                                        horizontalHosnestHalfcuts)
        if (self.cut_query is CutDirection.Vertical) or (self.cut_query is CutDirection.Both):
            verticalHosnestHalfcuts = [allocation.halfcuts[CutDirection.Vertical] for allocation in
                                       honestAllocations]
            allocation.halfcuts[CutDirection.Vertical] = self._calculate_best_halfcut(allocation, CutDirection.Vertical,
                                                                                      verticalHosnestHalfcuts)

    def _initial_cut_direction(self):

        if self.freeplay_mode:
            self.next_dir_index = 0
            return self.cut_pattern[0]
        else:
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

    def _get_next_query_from_list(self):
        self.next_dir_index = self.next_dir_index + 1 if self.next_dir_index + 1 < len(self.cut_pattern) else 0
        return self.cut_pattern[self.next_dir_index]

    def _get_query_direction_iterator_func(self):
        if self.freeplay_mode:
            return self._get_next_query_from_list
        else:
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

    def _calculate_best_halfcut(self, allocation, query_direction, hosnest_halfcuts):
        if len(hosnest_halfcuts) % 2 > 0:
            middleIdx = int(np.floor(len(hosnest_halfcuts) / 2))
        else:
            raise ValueError("Odd number of agents yet to be supported") # todo: add support for odd num of agents

        hosnest_halfcuts.sort()
        query_halfcut = hosnest_halfcuts[middleIdx]
        low_half = allocation.subCut(allocation.getDirectionaliFrom(query_direction), query_halfcut, query_direction)
        high_half = allocation.subCut(query_halfcut, allocation.getDirectionaliTo(query_direction), query_direction)

        if low_half.getValue() > high_half.getValue():
            return query_halfcut - SMALLEST_NUMBER
        else:
            return query_halfcut + SMALLEST_NUMBER

    def _set_next_cutting_direction(self, allocations, number_of_agents):
        _get_horizontal_halfcut = lambda alloc: alloc.halfcuts[CutDirection.Horizontal]
        _get_vertical_halfcut = lambda alloc: alloc.halfcuts[CutDirection.Vertical]

        if self.cut_pattern is CutPattern.SmallestHalfCut:
            first_hor_cut = min(allocations, key=_get_horizontal_halfcut)
            first_ver_cut = min(allocations, key=_get_vertical_halfcut)
            first_hor_cut_halfcut = _get_horizontal_halfcut(first_hor_cut)
            first_ver_cut_halfcut = _get_vertical_halfcut(first_ver_cut)
            if first_hor_cut_halfcut < first_ver_cut_halfcut:
                self.cut_direction = CutDirection.Horizontal
                #print("%s because - %s < %s" % (self.cut_direction, first_hor_cut_halfcut, first_ver_cut_halfcut))
            else:
                self.cut_direction = CutDirection.Vertical
                #print("%s because - %s < %s" % (self.cut_direction, first_ver_cut_halfcut, first_hor_cut_halfcut))
            return
        if self.cut_pattern is CutPattern.SmallestPiece:
            first_hor_cut = min(allocations, key=_get_horizontal_halfcut)
            first_ver_cut = min(allocations, key=_get_vertical_halfcut)
            first_hor_cut_halfcut = _get_horizontal_halfcut(first_hor_cut)
            first_ver_cut_halfcut = _get_vertical_halfcut(first_ver_cut)
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
        if self.cut_pattern is CutPattern.LargestRemainRange:
            sorted_horizontal_cuts = list(map(_get_horizontal_halfcut, sorted(allocations, key=_get_horizontal_halfcut)))
            sorted_vertical_cuts = list(map(_get_vertical_halfcut, sorted(allocations, key=_get_vertical_halfcut)))
            middle_index = int(np.ceil(number_of_agents/2))
            horizontal_remain = sorted_horizontal_cuts[middle_index] - sorted_horizontal_cuts[middle_index - 1]
            vertical_remain = sorted_vertical_cuts[middle_index] - sorted_vertical_cuts[middle_index - 1]

            if vertical_remain < horizontal_remain:
                self.cut_direction = CutDirection.Horizontal
                # print("%s because - %s < %s" % (self.cut_direction, vertical_remain, horizontal_remain))
            else:
                self.cut_direction = CutDirection.Vertical
                # print("%s because - %s < %s" % (self.cut_direction, horizontal_remain, vertical_remain))
            return
        if self.cut_pattern is CutPattern.LargestAvgRemainRange:
            sorted_horizontal_cuts = list(map(_get_horizontal_halfcut, sorted(allocations, key=_get_horizontal_halfcut)))
            sorted_vertical_cuts = list(map(_get_vertical_halfcut, sorted(allocations, key=_get_vertical_halfcut)))
            middle_index = int(np.ceil(number_of_agents/2))
            horizontal_remain = sorted_horizontal_cuts[middle_index] - sorted_horizontal_cuts[middle_index - 1]
            hor_avg_remain = np.average([np.abs(halfcut-horizontal_remain) for halfcut in sorted_horizontal_cuts])
            vertical_remain = sorted_vertical_cuts[middle_index] - sorted_vertical_cuts[middle_index - 1]
            ver_avg_remain = np.average([np.abs(halfcut-vertical_remain) for halfcut in sorted_vertical_cuts])

            if ver_avg_remain < hor_avg_remain:
                self.cut_direction = CutDirection.Horizontal
                # print("%s because - %s < %s" % (self.cut_direction, vertical_remain, horizontal_remain))
            else:
                self.cut_direction = CutDirection.Vertical
                # print("%s because - %s < %s" % (self.cut_direction, horizontal_remain, vertical_remain))
            return
        if self.cut_pattern is CutPattern.LargestRemainArea:
            sorted_horizontal_cuts = list(map(_get_horizontal_halfcut, sorted(allocations, key=_get_horizontal_halfcut)))
            sorted_vertical_cuts = list(map(_get_vertical_halfcut, sorted(allocations, key=_get_vertical_halfcut)))
            middle_index = int(np.ceil(number_of_agents/2))
            allocation_dimensions = allocations[0].getDimensions()
            horizontal_remain = (sorted_horizontal_cuts[middle_index] - sorted_horizontal_cuts[middle_index - 1]) * \
                                allocation_dimensions[CutDirection.Vertical]
            vertical_remain = (sorted_vertical_cuts[middle_index] - sorted_vertical_cuts[middle_index - 1]) * \
                              allocation_dimensions[CutDirection.Horizontal]

            if vertical_remain < horizontal_remain:
                self.cut_direction = CutDirection.Horizontal
                # print("%s because - %s < %s" % (self.cut_direction, vertical_remain, horizontal_remain))
            else:
                self.cut_direction = CutDirection.Vertical
                # print("%s because - %s < %s" % (self.cut_direction, horizontal_remain, vertical_remain))
            return
        if self.cut_pattern is CutPattern.MostValuableRemain:
            sorted_horizontal_cuts = list(map(_get_horizontal_halfcut, sorted(allocations, key=_get_horizontal_halfcut)))
            sorted_vertical_cuts = list(map(_get_vertical_halfcut, sorted(allocations, key=_get_vertical_halfcut)))
            middle_index = int(np.ceil(number_of_agents/2))
            hor_remain_iFrom = sorted_horizontal_cuts[middle_index - 1]
            hor_remain_iTo = sorted_horizontal_cuts[middle_index]
            hor_cut_option = (hor_remain_iFrom + hor_remain_iTo) / 2.0
            hor_remain_avg_value = np.average(list(map(lambda alloc:
                                                    alloc.getDirectionalValue(hor_cut_option, CutDirection.Horizontal),
                                                    allocations)))
            ver_remain_iFrom = sorted_vertical_cuts[middle_index - 1]
            ver_remain_iTo = sorted_vertical_cuts[middle_index]
            ver_cut_option = (ver_remain_iFrom + ver_remain_iTo) / 2.0
            ver_remain_avg_value = np.average(list(map(lambda alloc:
                                                    alloc.getDirectionalValue(ver_cut_option, CutDirection.Vertical),
                                                    allocations)))

            if ver_remain_avg_value < hor_remain_avg_value:
                self.cut_direction = CutDirection.Horizontal
                # print("%s because - %s < %s" % (self.cut_direction, ver_remain_avg_value, hor_remain_avg_value))
            else:
                self.cut_direction = CutDirection.Vertical
                # print("%s because - %s < %s" % (self.cut_direction, hor_remain_avg_value, ver_remain_avg_value))
            return
        if self.cut_pattern is CutPattern.MixedValuableRemain:
            sorted_horizontal_cuts = list(map(_get_horizontal_halfcut, sorted(allocations, key=_get_horizontal_halfcut)))
            sorted_vertical_cuts = list(map(_get_vertical_halfcut, sorted(allocations, key=_get_vertical_halfcut)))
            middle_index = int(np.ceil(number_of_agents/2))
            hor_remain_iFrom = sorted_horizontal_cuts[middle_index - 1]
            hor_remain_iTo = sorted_horizontal_cuts[middle_index]
            hor_remain_avg_value = np.average(list(map(lambda alloc:
                                                    alloc.getDirectionalValue(hor_remain_iFrom, hor_remain_iTo,
                                                                                       CutDirection.Horizontal),
                                                    allocations)))

            ver_remain_iFrom = sorted_vertical_cuts[middle_index - 1]
            ver_remain_iTo = sorted_vertical_cuts[middle_index]
            ver_remain_avg_value = np.average(list(map(lambda alloc:
                                                    alloc.getDirectionalValue(ver_remain_iFrom, ver_remain_iTo,
                                                                                       CutDirection.Vertical),
                                                    allocations)))

            if (ver_remain_avg_value < hor_remain_avg_value):
                if len(allocations) > 2:
                    self.cut_direction = CutDirection.Horizontal
                else:
                    self.cut_direction = CutDirection.Vertical
                # print("%s because - %s < %s" % (self.cut_direction, ver_remain_avg_value, hor_remain_avg_value))
            else:
                if len(allocations) > 2:
                    self.cut_direction = CutDirection.Vertical
                else:
                    self.cut_direction = CutDirection.Horizontal
                # print("%s because - %s < %s" % (self.cut_direction, hor_remain_avg_value, ver_remain_avg_value))
            return
        if self.cut_pattern is CutPattern.SquarePiece:
            sorted_horizontal_cuts = list(map(_get_horizontal_halfcut, sorted(allocations, key=_get_horizontal_halfcut)))
            sorted_vertical_cuts = list(map(_get_vertical_halfcut, sorted(allocations, key=_get_vertical_halfcut)))
            middle_index = int(np.ceil(number_of_agents/2))
            hor_remain_iFrom = sorted_horizontal_cuts[middle_index - 1]
            hor_remain_iTo = sorted_horizontal_cuts[middle_index]
            hor_cut_option = (hor_remain_iFrom + hor_remain_iTo) / 2.0
            hor_face_ratio_avg_value = np.average(list(map(lambda alloc:
                                                    alloc.getDirectionalFaceRatio(hor_cut_option,
                                                                                       CutDirection.Horizontal),
                                                    allocations)))

            ver_remain_iFrom = sorted_vertical_cuts[middle_index - 1]
            ver_remain_iTo = sorted_vertical_cuts[middle_index]
            ver_cut_option = (ver_remain_iFrom + ver_remain_iTo) / 2.0
            ver_face_ratio_value = np.average(list(map(lambda alloc:
                                                    alloc.getDirectionalFaceRatio(ver_cut_option,
                                                                                       CutDirection.Vertical),
                                                    allocations)))

            if ver_face_ratio_value < hor_face_ratio_avg_value:
                self.cut_direction = CutDirection.Horizontal
            else:
                self.cut_direction = CutDirection.Vertical
            return
        if self.cut_pattern is CutPattern.HighestScatter:
            sorted_horizontal_cuts = list(map(_get_horizontal_halfcut, sorted(allocations, key=_get_horizontal_halfcut)))
            sorted_vertical_cuts = list(map(_get_vertical_halfcut, sorted(allocations, key=_get_vertical_halfcut)))
            neighbor_horizontal_cuts = list(zip(sorted_horizontal_cuts, np.roll(sorted_horizontal_cuts, -1)))[:-1]
            neighbor_vertical_cuts = list(zip(sorted_vertical_cuts, np.roll(sorted_vertical_cuts, -1)))[:-1]
            # print(neighbor_horizontal_cuts,"---",neighbor_vertical_cuts)
            hor_scatter_avg_value = np.average([b - a for (a, b) in neighbor_horizontal_cuts])
            ver_scatter_avg_value = np.average([b - a for (a, b) in neighbor_vertical_cuts])
            # print("\t",hor_scatter_avg_value, "---", ver_scatter_avg_value)

            if ver_scatter_avg_value < hor_scatter_avg_value:
                self.cut_direction = CutDirection.Horizontal
            else:
                self.cut_direction = CutDirection.Vertical
            return
        else:
            self.cut_direction = self.cut_query