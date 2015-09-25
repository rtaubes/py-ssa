#!/usr/bin/env python
'''
    Simple test for SSA
'''
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as ls
import statsmodels as sm

def src_matrix(y, N, L):
    # create hankel matrix from y
    K = N - L + 1
    ys = np.zeros((L, K))
    shift = 0
    for col in range(K):
        for ln in range(L):
            ys[ln, col] = y[shift + ln]
        shift += 1
    return ys

def lin_source(x):
    # return [1, 2]
    return np.linspace(1, x, x)

def square_source(x):
    y0 = np.linspace(1, x, x)
    return y0 * y0


def make_ssa(hm):
    # create source matrix by shifting 'y'
    u, s, v = ls.svd(hm, full_matrices=False)
    # from help of numpy.linalg.svd:
    # The SVD is commonly written as ``a = U S V.H``.  The `v` returned
    # by this function is ``V.H`` and ``u = U``.

    # If ``U`` is a unitary matrix, it means that it
    # satisfies ``U.H = inv(U)``.

    # The rows of `v` are the eigenvectors of ``a.H a``. The columns
    # of `u` are the eigenvectors of ``a a.H``.  For row ``i`` in
    # `v` and column ``i`` in `u`, the corresponding eigenvalue is
    # ``s[i]**2``.

    # If `a` is a `matrix` object (as opposed to an `ndarray`), then so
    # are all the return values.

    # eigenvector is a vector for which A.u = a.u, where
    # u - eigenvector, A - square matrix, a - scalar constant(eigen value).
    # eigenvectors can be found by solving 'A' as system of linear equations.

    # Reconstruction based on reduced SVD:
    # >>> U, s, V = np.linalg.svd(a, full_matrices=False)
    # >>> U.shape, V.shape, s.shape
    # ((9, 6), (6, 6), (6,))
    # >>> S = np.diag(s)
    # >>> np.allclose(a, np.dot(U, np.dot(S, V)))
    # True

    return u, s, v

def diag_average(y):
    ''' diagonal averaging of a matrix
        @param  matrix
        @return vector of averaged data
    '''
    L, K = y.shape
    Ks = max(K, L)
    Ls = min(K, L)
    N = L + K - 1
    print ("diag_average() L:%d, K:%d, Ks:%d, Ls:%d, N:%d" % (L,K,Ks,Ls,N))
    g = []
    for k in range(N):
        elem = 0
        if k < Ls-1:
            for m in range(1, k+2):
                if L < K:
                    elem += y[m-1, k-m+1]
                else:
                    elem += y[k-m+1, m-1]
            elem = elem / (k+1)
        elif k < Ks:
            for m in range(1, Ls+1):
                if L < K:
                    elem += y[m-1, k-m+1]
                else:
                    elem += y[k-m+1, m-1]
            elem = elem / Ls
        else:   # Ks <= k < N
            for m in range(k-Ks+2, N-Ks+2):
                if L < K:
                    elem += y[m-1, k-m+1]
                else:
                    elem += y[k-m+1, m-1]
            elem = elem /(N - k)
        g.append(elem)
    return g

def averaging(mlist):
    ''' averaging each matrix from list
        and return list of vectors, one for each matrix
        @param list of matrices
        @return list of vectors
    '''
    ret = []
    for m in mlist:
        v = diag_average(m)
        ret.append(v)
    return ret

def ssa_unit_matrices(u, s, v):
    ''' select triples from SVD result and create an unitx matrices
        @param s,v,d - result of SVD
        @return list of unit matrixes at the same order as triples
    '''
    ret = []
    for i in range(s.size):
        ui = u[:,i:i+1]
        vi = v[i:i+1,:]
        si = s[i]
        # here it is a place to analyze the SSA triangles
        xi = np.dot(ui, np.dot(si, vi))
        print "%d -> %s" % (i, xi)
        ret.append(xi)
    return ret

if __name__ == '__main__':
    N = 8
    L = 4
    x = np.array(range(N))
    y = lin_source(N)
    print "y:", y
    hm = src_matrix(y, N, L)
    print "hankel matrix:", hm, "\n"
    u, s, v = make_ssa(hm)
    print "u:", u, "\n"
    print "s:", s, "\n"
    print "v:", v, "\n"
    S = np.diag(s)
    print "S:", S, "\n"
    u2 = np.dot(u, s)
    print "u2 = u * s:", u2, "\n"

    # unit matrixes for unit vectors
    umatrix_list = ssa_unit_matrices(u, s, v)


    # list of vectors, one for each matrix
    vec_list = averaging(umatrix_list)
    print "vec_list", vec_list

    # make summary vector
    rv = np.zeros(len(vec_list[0]))
    for v in vec_list:
        rv += np.array(v)
    print "summary vector:", rv

    # calc coefficients
    d = 0 # rank
    bound = np.max(s) * 1e-6
    for si in s:
        print "si:%f, bound:%f" % (si, bound)
        if si <= bound:
            break
        d += 1
    print "\nrank:", d
    uu = u.T
    v2 = 0
#    R = np.zeros(d)
    R = None
    for r in range(d):
        # by rows - eigenvectors
        vec = uu[r]
        vsz = vec.size
        v2 += vec[vsz-1]**2
        print "vec:%s, vsz:%d, last elem:%f" % (repr(vec), vsz, vec[vsz-1])
#        R += vec[vsz-3 : vsz-1] * vec[vsz-1]
        if R is None:
            R = vec[:vsz-1] * vec[vsz-1]
        else:
            R += vec[:vsz-1] * vec[vsz-1]
        print "R:", R
    print "eq (4a): ", R
    R = R / (1 - v2)
    print "R - eq (4): ", R

