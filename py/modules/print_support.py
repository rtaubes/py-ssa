#!/usr/bin/env python

''' Print support for arrays and matrices, logging.INFO level.

    How to use with debug levels:

    def debug_matrix(self, m, title):
        if self.loglevel <= logging.DEBUG:
            self.log_matrix(m, title)
'''

import logging


def log_array(m, title):
    ''' print to log array values '''
    log = logging.getLogger('parray')
    log.info("%s (%d):" % (title, m.size))
    elems = 0
    pln = '['
    for c in m:
        if c > -10 and c < 10:
            s = "% 2.8f  " % (c)
        else:
            s = "% 2.6e  " % (c)
        pln += s
        elems += 1
        if elems == 5:
            log.info(pln)
            elems = 0
            pln = "   "
    log.info(pln + ']')

def log_matrix(m, title):
    ''' print to log matrix '''
    lns, cols = m.shape
    log = logging.getLogger('pmatrix')
    log.info("%s (%dx%d):" % (title, lns, cols))
    for ln in range(lns):
        pln = '['
        elems = 0
        for c in range(cols):
            if m[ln,c] > -10 and m[ln,c] < 10:
                s = "% 2.8f  " % (m[ln,c])
            else:
                s = "% 2.6e  " % (m[ln,c])
            pln += s
            elems += 1
            if elems == 5:
                log.info(pln)
                elems = 0
                pln = " "
        pln += ']'
        log.info(pln)

def matrix_shape_str(m):
    ''' matrix shape as string '''
    sh = m.shape
    if len(sh) == 1:
        return "%dx1" % sh[0]
    ret = ""
    for i in sh:
        ret += "%dx" % i
    return ret[:-1]

