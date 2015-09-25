#!/usr/bin/env python

''' analyse triples (U, S, V) SVD produced .
    Group closest together. Create list of grouped
    components.
'''

import numpy as np

class FFT_grouping(object):
    def __init__(self, config, v, s, u):
        '''
            @param conifg - config object
            @param u, s, v - SVD result
        '''
        self.log = logging.getLogger('FFT_grp')
        self.V = v
        self.S = s
        self.U = u

    def grouping(self):
        ''' group items of SVD result,
            return list of groups, like this: [[1,2], [3], [4,5,6]]
            Only neighbours can be grouped togheter.
        '''
        # separate groups by S values
        prelim_group_list = self.groups_by_S()
        # make groups using result above(based on correlation of fft).
        group_list = self.groups_by_fft(prelim_group_list)
        return group_list

