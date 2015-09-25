#!/usr/bin/env python
'''
    1) Make SSA analyze of input,
    2) create SSA distribution,
    3) create data parts from SSA as components
    of the original data
'''
# TODO: use only selected components in result

import numpy as np
import numpy.linalg as ls
import logging
import math
#import config
from print_support import log_array, log_matrix, matrix_shape_str


class SSAError(Exception):
    pass


class SSAErrorDataRange(Exception):
    pass


class SSA(object):
    ''' SSA processing of input vector '''

    def __init__(self, conf, loglevel):
        self.log = logging.getLogger('SSA_ALGO')
        self.loglevel = loglevel
        # conf parameters
        self.data_offset = 0    # offset within analized data
        self.N = 0      # size of analized data (N)
        self.L = 0    # size of analize window (L)
        self.conf = conf
        self.noise_level = 0    # noise level - all below are noice
        self.ydata = None       # data vector, size is data_size
        self.in_matrix = None
        self.unit_matrices = None
        self.partial_matrices = None  # produced data from groups
        # optimized by size vector s,u,v
        self.u_opt = None
        self.v_opt = None
        self.s_opt = None
        # full matrices s,u,v
        self.s_full = None          # 'S' vector from SVD. Unmodified.
        self.u_full = None
        self.v_full = None
        self.group_list = []        # list of groups. Ex: [[1], [2,3], [4], [5,6,7,8]]
        self.poly_roots = None      # roots of estimation polynom. Usually
                                    # polynom size is self.L - 1
        self.predict_poly = None    # prediction polynom
#        self.R = None
        self.main_poly = None
        self.calc_poly_roots = False    # set to True for caclulation of
                                        # polynom roots of result function.
        self.predicted = None   # array for calculation predicted data.
                                # Last value is current predicted.
        self.predict_x = 0      # 'x' for next predicted.

    def set_s_size(self, sz):
        ''' set optimized SSA size '''
        self.log.debug("set_s_size(): S=%s" % sz)
        self.s_opt = self.s_full.copy()
        self.u_opt = self.u_full.copy()
        self.v_opt = self.v_full.copy()
        self.u_opt = self.u_opt[:, :sz]
        self.v_opt = self.v_opt[:sz, :]
        self.s_opt = self.s_opt[:sz]
        self.log.debug("set_s_size(): svd: u_full: {}:".format(self.u_full))
        self.log.debug("set_s_size(): s_full: {}:".format(self.s_full))
        self.log.debug("set_s_size(): v_full: {}:".format(self.v_full))
        self.log.debug("set_s_size(): u_opt: {}".format(self.s_opt))
        self.log.debug("set_s_size(): s_opt: {}".format(self.s_opt))
        self.log.debug("set_s_size(): v_opt: {}".format(self.s_opt))

    def set_S(self, S):
        ''' set SSA 'S' optimized vector '''
        self.s_opt = S
        self.log.info("set_S(): S=%s" % self.s_opt)

    def get_poly_roots(self):
        ''' return np.array of polynom roots '''
        return self.poly_roots

    def get_N_data(self):
        x = range(self.data_offset, self.data_offset + self.N)
        y = self.ydata[self.data_offset:self.data_offset + self.N]
        return [x, y]

    def data(self, ulim=1e20, offset=0):
        upidx = self.ydata.size
        if ulim < self.ydata.size:
            upidx = ulim
        return [range(offset, upidx), self.ydata[offset:upidx]]

    def assign_groups(self, glist):
        ''' assign combination of eigenvectors(unit matrixes)
            to groups.
            @param list which elemenet of it is list of eigenvector indexes
        '''
        self.group_list = glist

    def u_eigen_vectors(self):
        ''' return array of U eigenvectors of SVD decomposition. Each
            row is an U vector '''
        if self.u_opt is None:
            return np.array([])
        return self.u_opt.transpose()

    def s_svd(self):
        ''' return non-optimized S of SVD decomposition. '''
        if self.s_full is None:
            return np.array([])
        return np.power(self.s_full, 2)

    def s_svd_opt(self):
        ''' return optimized S of SVD decomposition. '''
        if self.s_opt is None:
            return np.array([])
        return np.power(self.s_opt, 2)

    def v_eigen_vectors(self):
        ''' return array of V eigenvectors of SVD decomposition.
            Eacho row is an V vector '''
        if self.u_opt is None:
            return np.array([])
        return self.v_opt

    def debug_array(self, m, title):
        if self.loglevel <= logging.DEBUG:
            log_array(m, title)

    def debug_matrix(self, m, title):
        if self.loglevel <= logging.DEBUG:
            log_matrix(m, title)

    def configure(self, conf):
        self.L = conf.ssa.window_size
        self.N = conf.common.data_size
        self.data_offset = conf.common.data_offset
        self.noise_level = conf.ssa.noise_level

    def prepare_hankel_matrix(self):
        ''' create data matrix from data vector
            @param y - data vector with length N
            @return: Hankel matrix LxK, where K=N-L+1.
                Matrix created by shifting y for producing matrix column
        '''
        try:
            if self.L < 1 or self.L > self.N - 1:
                raise SSAError("index self.L not in range 1..%d" % (self.N-1))
            K = self.N - self.L + 1
            ys = np.zeros((self.L, K))
            shift = 0
            for col in range(K):
                for ln in range(self.L):
                    ys[ln, col] = self.ydata[self.data_offset + shift + ln]
                shift += 1
            return ys
        except IndexError:
            err = "prepare_hankel_matrix(): index out of range - data too small"
            self.log.error(err)
            raise SSAErrorDataRange(err)

    def make_ssa(self, hmatrix, delta=0):
        ''' make svd transofrm form LxK matrix
            @param hmatrix - hankel matrix
            @param delta   - low limit (zero)
            @return u - column of eigenvectors
                    s - vector of eigenvalues
                    v - rows of eigenvectors
                All u,s,v are truncated - zero values removed
        '''
        self.u_full, self.s_full, self.v_full = ls.svd(hmatrix, full_matrices=False)
        self.in_matrix = hmatrix
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

        # truncate zero values
        self.log.info("make_ssa()")
        self.log.debug("svd result:")
        self.debug_matrix(self.u_full, "svd: u_full")
        self.debug_array(self.s_full, "svd: s_full")
        self.debug_matrix(self.v_full, "svd: v_full")
        self.s_opt = self.s_full.copy()
        self.u_opt = self.u_full.copy()
        self.v_opt = self.v_full.copy()
        self.debug_matrix(self.u_full, "svd: u_opt")
        self.debug_array(self.s_full, "svd: s_opt")
        self.debug_matrix(self.v_full, "svd: v_opt")

        s_bound = np.max(self.s_opt) * delta
        real_sz = 0
        for si in self.s_opt:
            if si < s_bound:
                break
            real_sz += 1
        self.u_opt = self.u_opt[:, :real_sz]
        self.v_opt = self.v_opt[:real_sz, :]
        self.s_opt = self.s_opt[:real_sz]
        self.log.info("sizes of s,u,v truncated to %d" % len(self.s_opt))

    def create_unit_matrices(self, sz=-1):
        ''' create unit matrices for 'sz' members of SVD
            sz = -1 -> use whole data from matrix
        '''
        if sz > self.s_full.size:
            err = "create_group_matrices(): parameter"
            err += "({}) too long(more than S size {})".format(sz, self.s_full.size)
            self.log.error(err)
            raise SSAErrorDataRange(err)
        if sz < 0:
            sz = self.s_full.size
        self.debug_matrix(self.u_opt, "u")
        self.debug_matrix(self.v_opt, "v")
        ret = []
        xi = None
        for i in range(sz):
            ui = self.u_opt[:,i:i+1]
            vi = self.v_opt[i:i+1,:]
            si = self.s_full[i]
            # here it is a place to analyze the SSA triples
            xi = np.dot(ui, np.dot(si, vi))
            self.debug_matrix(ui, "u%d" % i)
            self.debug_matrix(vi, "v%d" % i)
            self.debug_matrix(xi, "x%d" % i)
            ret.append(xi)
        self.unit_matrices = np.array(ret)

    def create_group_matrices(self, sz):
        ''' select triples from SVD result and create an unit matrices.
            Grouping them according to groups
        '''
        if sz > self.s_full.size:
            err = "create_group_matrices(): parameter"
            err += "({}) too long(more than S size {})".format(sz, self.s_full.size)
            self.log.error(err)
            raise SSAErrorDataRange(err)
        self.create_unit_matrices(sz)
        msg = "%d unit matrices" % (sz)
        self.log.info(msg)
        # grouping them
        pm = []
        ln, rows = self.unit_matrices[0].shape
        for g in self.group_list:
            gm = np.zeros((ln, rows))
            # create partial group, gi is an index of unit matrix
            for gi in g:
                gm += self.unit_matrices[gi]
            pm.append(gm)
        self.partial_matrices = np.array(pm)

    def __diag_average_um(self, um):
        ''' diagonal averaging of matrix
            @param um - matrix
            @return vector of averaged values
        '''
        L, K = um.shape
        Ks = max(K, L)
        Ls = min(K, L)
        N = L + K - 1
        self.log.debug("L:%d, K:%d, Ks:%d, Ls:%d, N:%d" %(L,K,Ks,Ls,N))
        g = []
        for k in range(N):
            elem = 0
            if k < Ls-1:
                for m in range(1, k+2):
                    if L < K:
                        elem += um[m-1, k-m+1]
                    else:
                        elem += um[k-m+1, m-1]
                elem = elem / (k+1)
            elif k < Ks:
                for m in range(1, Ls+1):
                    if L < K:
                        elem += um[m-1, k-m+1]
                    else:
                        elem += um[k-m+1, m-1]
                elem = elem / Ls
            else:   # Ks <= k < N
                for m in range(k-Ks+2, N-Ks+2):
                    if L < K:
                        elem += um[m-1, k-m+1]
                    else:
                        elem += um[k-m+1, m-1]
                elem = elem /(N - k)
            g.append(elem)
        return np.array(g)

    def diag_average(self):
        ''' diagonal averaging each partial matrix
            @returns list of vectors, where each vector is separated
            component of input.
        '''
        ret = []
        for m in self.partial_matrices:
            v = self.__diag_average_um(m)
            ret.append(v)
        return ret

    def diag_average_unit_matrices(self):
        ''' diagonal averaging each unit matrix
            @returns list of vectors, where each vector is diagonalized
            unit matrix.
        '''
        ret = []
        for m in self.unit_matrices:
            v = self.__diag_average_um(m)
            ret.append(v)
        return ret

    def _w_production(self, f1, f2):
        ''' calculate omega-production of array-like f1, f2
            Vectors should have the same size.
            @param f1 - vector a
            @param f2 - vector b
        '''
        if len(f1) != len(f2):
            err = "calc_w_production(): data sizes not equal({} and {})".format(
                    len(f1), len(f2))
            self.log.error(err)
            raise SSAErrorDataRange(err)
        K = self.N - self.L + 1
        Ls = min(self.L, K)
        Ks = max(K, self.L)
        res = 0
        for i in range(len(f1)):
            if i < Ls - 1:
                w = i + 1
            elif i < Ks:
                w = Ls
            else:
                w = self.N - i
#            self.log.debug("_w_production(): w:{}".format(w))
            res += w * 1.0 * f1[i] * f2[i]
        return res

    def calc_w_correl(self, f1, f2):
        ''' calculate omega-correlation of array-like f1, f2
            Vectors should have the same size.
            @param f1 - vector a
            @param f2 - vector b
        '''
        if len(f1) != len(f2):
            err = "calc_w_correl(): data sizes not equal({} and {})".format(
                    len(f1), len(f2))
            self.log.error(err)
            raise SSAErrorDataRange(err)
        self.log.debug("calc_w_correl(): f1:{}".format(f1))
        self.log.debug("calc_w_correl(): f2:{}".format(f2))
        nomin = self._w_production(f1, f2)
        denomin = math.sqrt(self._w_production(f1, f1))
        denomin *= math.sqrt(self._w_production(f2, f2))
        self.log.debug("calc_w_correl(): nomin={}, denomin={}".format(
            nomin, denomin))
        if denomin < 1e-15:
            err = "calc_w_correl(): denominator {} too small".format(denomin)
            self.log.error(err)
            return 1e10
        return nomin / denomin

    def merge_main_groups(self):
        ''' merge main groups to one '''
        if len(self.partial_matrices) == 0:
            self.log.warn("Could not merge nothing in partial matrices")
            return
        r, c = self.partial_matrices[0].shape
        mm = np.zeros((r, c))
        for m in self.partial_matrices:
            mm += m
        self.partial_matrices = [ mm ]

    def v2(self, d):
        ''' calc v**2 for size d.
            When v**2 about or equal 1, prediction impossible
            @param d - U vectors count. Should be less or equal
                to self.window_size (L)
        '''
        if d > self.s_opt.size:
            err = "v2() calc error: parameter %d more than U size(L): %d" % \
                (d, self.s_opt.size)
            self.log.error(err)
            raise SSAError(err)
        v2 = 0
        uu =  self.u_opt.transpose()
        for r in range(d):
            v2 += uu[r][vsz-1] ** 2
        return v2

    def prepare_predict(self):
        ''' prepare prediction calculation for data without last group
            1. Calc polynom coeff.
            2. Creating Hankel matrix from separated part and diagonalization
            should be made before.
            4. Calc roots of polynom.
            5. Estimate roots.
            6. Recalc polynom from roots - minimal form of polynom.
        '''
        # eigenvectors in rows
        uu = self.u_opt.transpose()
        r, c = uu.shape
        R = None
        v2 = 0
        d = self.s_opt.size
        if d == self.L:
            self.log.info("d(%d) == L!. Let d=L-1" % d)
            d = self.L - 1

        self.log.info("Predict data: s size:%d, %d 'u' vectors(%d)" %(d, r, c))
        vsz = uu[0].size
        for r in range(d):
            vec = uu[r]
            v2 += vec[vsz-1] ** 2
            if R is None:
                R = vec[:vsz-1] * vec[vsz-1]
                self.log.debug("R[%d] = %s" % (r, R))
            else:
                self.log.debug("R[%d] = %s" % (r, vec[:vsz-1] * vec[vsz-1]))
                R += vec[:vsz-1] * vec[vsz-1]
        self.log.info("v2 = %f, R = %s" % (v2, R))
        if np.abs(v2 - 1.0) < 0.01:
            err = "v2 in R calculation close to 1. Prediction impossible"
            self.log.error(err)
            raise SSAError(err)
        if not R is None:
            self.predict_poly = R / (1 - v2)
        else:
            self.predict_poly = np.array([])
        self.log.debug("(eq 4): prediction poly:%s" % self.predict_poly)
        self.log.info("predict poly(predict_poly): %s" % (self.predict_poly))
        if self.calc_poly_roots:
            # np.roots parameter expects of poly members in that order:
            #  p0 * x^n + p1 * x^(n-1) + ...
            # i.e. highest degree first
            self.poly_roots = np.roots(self.predict_poly)
            for root in self.poly_roots:
                self.log.info("R root:%s (%f)" % (root, np.linalg.norm(root)))
            main_roots = self.poly_roots[:d]
            # check main_roots
            for root in main_roots:
                if np.linalg.norm(root) > 1:
                    self.log.warn("Warning: root (%s) has mod %f > 1"
                            % (root, np.linalg.norm(root)))
            self.main_poly = np.poly(main_roots)
            # main_poly has coefficients from highest to lowest degree
#            self.log.info("R: %s" % (self.R))
            self.log.info("Polynom from main roots:%s" % (self.main_poly))
        else:
            self.np_roots = None
        self.reset_predict()
        self.log.info("predict array: %s" % self.predicted)

    def reset_predict_by_F1(self, offset):
        ''' reset predicted data using F(1) values from offset.
            When data size too small, exception generated.
        '''
        f1_data = self.calc_F1()    # result is [x, y]
        f1_x = f1_data[0]
        f1_y = f1_data[1]
#        print ("reset_predict_by_F1(), fi_x[0]: {}".format(f1_x[0]))
        p0 = offset  # x[0] + offset
        p1 = p0 + self.predict_poly.size + 1 # 1 is for predicted value
        self.predicted = f1_y[p0:p1] * 1.0
#        print("reset_predict_by_F1((): p0: {}, p1: {}, predict sz: {}"
#                .format(p0, p1, len(self.predicted)))
        self.predict_x = f1_x[0] + p1

    def reset_predict(self):
        ''' reset predicted data.  '''
        p1 = self.data_offset + self.N # 'x' of predicted
        p0 = p1 - self.predict_poly.size - 1
        self.predicted = self.ydata[p0:p1] * 1.0
        self.predict_x = p1

    def report(self):
        ''' make report about currnet state '''
        log = logging.getLogger("SSA.Report")
        log.info("==============================")
        log.info("=== ssa calculation report ===")
        log.info("==============================")
        log.info("- input vector:%s" % (self.ydata))
        self.debug_matrix(self.in_matrix, "- input matrix")
        log.info("- data size:%d" % self.N)
        log.info("- window size:%d" % self.L)
        log.info("- data offset:%d" % self.data_offset)
        log.debug("- unix matrices:")
        if not self.unit_matrices is None:
            for i in range(len(self.unit_matrices)):
                log.debug("%d: %s" % (i, matrix_shape_str(self.unit_matrices[i])))
        log.debug("- partial matrices:")
        if not self.partial_matrices is None:
            for i in range(len(self.partial_matrices)):
                log.debug("%d: %s" % (i, matrix_shape_str(self.partial_matrices[i])))
        log.info(" - s_opt size:{}".format(len(self.s_opt)))
        self.debug_array(self.s_opt, "- S(opt)")
        self.debug_matrix(self.u_opt, "- U(opt)")
        self.debug_matrix(self.v_opt, "- V(opt)")
        a = ""
        for g in self.group_list:
            a += "%s " % g
        log.info("- group_list:%s" % (a))
        log.info("===============================")

    def predict(self):
        ''' Make one step prediction based on prepared polynom
            @return predicted value as (x, y)
        '''
        log = logging.getLogger("Predict")
        sz = self.predicted.size
        self.predicted = np.hstack((self.predicted[1:], np.array([0])))

#        print("predict(): sz:{}, poly sz: {}".format(sz, len(self.predict_poly)))

        self.predicted[sz-1] = np.dot(self.predicted[:sz-1], self.predict_poly)
        log.debug("predicted:%s" % self.predicted)
        cx = self.predict_x
        self.predict_x += 1
        return cx, self.predicted[sz-1]

    def calc_rest(self):
        ''' calculate difference between SSA approximation and real data/
            It is a rest of SSA decomposition don't used in predict.
            @return vector of rest values
        '''
        opt_sz = self.s_opt.size
        s = self.s_full[opt_sz:]
        u = self.u_full[:, opt_sz:]
        v = self.v_full[opt_sz:, :]
        self.debug_array(s, " - S(rest)")
        self.debug_matrix(u, " - U(rest)")
        self.debug_matrix(v, " - V(rest)")
        m = np.dot(u, np.dot(np.diagflat(s), v))
        y = self.__diag_average_um(m)
        self.log.debug("calc_rest(), y:{}".format(y))
        return [np.array(range(y.size)) + self.data_offset , y]

    def calc_F1(self):
        ''' calculate F1 in input range.
            @return vector of F(1) values.
        '''
        u = self.u_full[:, :self.s_opt.size]
        v = self.v_full[:self.s_opt.size, :]
        self.log.debug("calc_F1(): u:{}, self.s_opt:{}, v:{}"
                .format(u.shape, np.diagflat(self.s_opt).shape, v.shape))
        m = np.dot(u, np.dot(np.diagflat(self.s_opt), v))
        y = self.__diag_average_um(m)
        self.log.debug("calcF1(), y:{}".format(y))
        x = np.array(range(y.size)) + self.data_offset
        return [x, y]
