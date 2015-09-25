#!/usr/bin/env python
'''
    Utilities and operations
'''
import numpy as np

def fft_on_list(vec_list):
    ''' returns matrix of spektres
        @arg list of numpy vectors
    '''
    out_list = []
    for vec in vec_list:
        gx = vec
        spektr = abs(np.fft.rfft(gx))
        out_list.append(spektr)
    return np.array(out_list)


def calc_fft_correlations(in_matrix):
    ''' calculate correlation between matrix rows
        @param matrix
        @return vector of correlation coefficients.
        Length of vector is one less than in_matrix rows.
        value [i] = correlation of i, i+1 rows of in_matrix
    '''
    rows, c = in_matrix.shape
#    print "rows:%d, cols:%d" % (rows, c)
    if rows < 2:
        return np.array([])
    corr_vect = np.zeros(rows-1)
    for r in range(rows-1):
        # correlation between r and r+1 rows
        corr = np.sqrt( np.dot(in_matrix[r], in_matrix[r+1]) /
                (np.sum(in_matrix[r]) * np.sum(in_matrix[r+1])))
#        print "row[%d] %s" % (r, repr(in_matrix[r]))
#        print "row[%d] %s" % (r+1, repr(in_matrix[r+1]))
#        print "nominator[%d]: %f" % (r, np.dot(in_matrix[r], in_matrix[r+1]))
#        print "corr:", corr
        corr_vect[r] = corr
    return corr_vect

def funcs_by_data(x, y, deg):
    ''' find function which describes data in vector
        @param x    array-like X
        @param y    array-like Y
        @param deg  degree of polynom
        @return vector of coefficients
    '''
    return np.polyfit(x, y, deg)

