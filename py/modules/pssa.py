#!/usr/bin/env python
'''
    SSA algo:
    - configure by config file;
    - read or produce data;
    - some steps for L = [L0 .. L=N/2], L0 is configured:
        - process data by SSA;
        - create DFFT for produced parts;
    - choose better grouping for slow frequencies;
    - grouping data according config parameters;
    - create functions for describing grouped data;
'''

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSignal
import logging
from print_support import log_array, log_matrix
import ssa_algo
import utils
import data_source
import numpy as np


class PSSAError(Exception):
    pass


class PSSA(QtCore.QThread):
    SSA_W_OPT_STEPS = 11
    SSA_RANK_MAX = 15   # maximum interested rank
    W_MIN_DEPTH = 5     # w-correlation depth
    USE_W_CORR_WEIGHTS = True
    DATA_DECIMALS = 6   # decimals in data. Should be provided in
                        # data description.

    # Step of ssa calculation started.
    ssa_step_start = pyqtSignal(int, int, int) # offset, N, L
    # Step of ssa calculation finished.
    ssa_step_done = pyqtSignal(int, float) # rank, w-min
    # SSA calculation done.
    ssa_calc_finished = pyqtSignal(int, int, int) # offset, N, L
    # Just a message
    ssa_log_msg = pyqtSignal(str)
    ssa_log_error = pyqtSignal(str)

    def __init__(self, parent, config_obj, loglevel):
        super(QtCore.QThread, self).__init__(parent)
        self.log = logging.getLogger('PSSA')
        self.parent = parent
        self.loglevel = loglevel
        self.ssa = None
        self.u_fft = None
        self.u_fft_pw = None
        self.cfg = config_obj
        self.predict_possible = False
        self.data_src = None
        self.u_correl = None
        self.w_correl = None    # omega-correlation of unit matrices.
        self.predict_errors = None # list of [x], [y] for F(2) in calc range
        self.preferred_rank = 0 # for simple SSA calc - set rank manually.
        self.commands = {"calculate_ssa" : self.__calculate_ssa,
                "calculate_w_optimized_ssa" : self.__calculate_w_optimized_ssa}
        self.cmd = ''
        self.args = None
        self.ssa_step_start.connect(self.parent.sig_ssa_step_start)
        self.ssa_step_done.connect(self.parent.sig_ssa_step_done)
        self.ssa_calc_finished.connect(self.parent.sig_ssa_calc_finished)
        self.ssa_log_msg.connect(self.parent.sig_ssa_msg)
        self.ssa_log_error.connect(self.parent.sig_ssa_error)

    def do_cmd(self, cmd, kargs=None):
        self.cmd = cmd
        self.args = kargs

    def run(self):
        ''' QThread main function '''
        self.log.info("start ssa thread with cmd [{}]".format(self.cmd))
        if self.cmd not in self.commands.keys():
            e = "Unknown command [{}]".format(self.cmd)
            self.log.error(e)
            self.ssa_calc_finished.emit(0, 0, 0)
            return
        self.commands[self.cmd]()

    def debug_matrix(self, m, title):
        if self.loglevel <= logging.DEBUG:
            log_matrix(m, title)

    def debug_array(self, m, title):
        if self.loglevel <= logging.DEBUG:
            log_array(m, title)

    def data(self, ulim=20e9, offset=0):
        ''' return data separated by ulim as [x, y] '''
        return self.ssa.data(ulim, offset)

    def get_N_data(self):
        ''' return data for N as [x, y]. X includes offset '''
        return self.ssa.get_N_data()

    def get_N(self):
        return self.ssa.N

    def get_L(self):
        return self.ssa.L

    def set_N(self, N):
        self.ssa.N = N

    def set_L(self, L):
        self.ssa.L = L

    def get_data_offset(self):
        return self.ssa.data_offset

    def set_data_offset(self, val):
        self.ssa.data_offset = val

    def get_config_N(self):
        return self.cfg.ssa.data_range

    def get_config_data_offset(self):
        return self.cfg.common.data_offset

    def get_config_L(self):
        return self.cfg.ssa.window_size

    def u_eigen_vectors(self):
        if self.ssa is None:
            return []
        return self.ssa.u_eigen_vectors()

    def v_eigen_vectors(self):
        if self.ssa is None:
            return []
        return self.ssa.v_eigen_vectors()

    def s_svd(self):
        if self.ssa is None:
            return np.array([])
        return self.ssa.s_svd()

    def s_svd_opt(self):
        if self.ssa is None:
            return np.array([])
        return self.ssa.s_svd_opt()

    def get_poly_roots(self):
        if self.ssa is None:
            return np.array([])
        return self.ssa.get_poly_roots()

    def set_S(self, S):
        ''' set SSA 'S' vector '''
        if self.ssa is None:
            return
        self.ssa.set_S(S)

    def set_s_size(self, sz):
        ''' set SSA 'S' vector size '''
        if self.ssa is None:
            return
        self.ssa.set_s_size(sz)

    def calc_rest(self):
        ''' calculate data not used in predict '''
        if self.ssa is None:
            return
        if self.predict_errors is None:
            self.predict_errors = self.ssa.calc_rest()
        return self.predict_errors

    def predict(self):
        ''' make prediction step for data.
            @return pair 'x', 'y'
        '''
        return self.ssa.predict()

    def prepare_predict(self):
        if self.ssa is None:
            return
        try:
            self.ssa.prepare_predict()
        except ssa_algo.SSAError, e:
            raise PSSAError(e)

    def reset_predict(self):
        ''' reset predicted data.  '''
        return self.ssa.reset_predict()

    def calc_F1(self):
        return self.ssa.calc_F1()

    # TODO: should be empty grouping when S is empty
    def grouping_items(self, corr_vec):
        ''' return list of list of item indexes in SVD. Count of groups
            is limited by 5. Rest is grouped to the last group 6.
            @param corr_vec - vector of correlations of DFFT for U eigen
                              vectors. Size of vector is 1 less than
                              U vectors count - each corr value is for
                              two U vectors
        '''
        GROUP_BARRIER = 0.5
        GROUP_LIMIT = 5
        ret = []
        curr_group = []
        # update vector by zero value. It allow to include
        # last member of group
        corr_vec_up = np.hstack((corr_vec, np.array([0])))
        for idx in range(len(corr_vec_up)):
            self.log.debug("%d: corr_vec[%d]: %f" %(idx, idx, corr_vec_up[idx]))
            curr_group.append(idx)
            if corr_vec_up[idx] < GROUP_BARRIER:
                # fix group, create new one
                ret.append(curr_group)
                if len(ret) >= GROUP_LIMIT:
                    GROUP_BARRIER = -1
                curr_group = []
        # save last non empty group
        if len(curr_group):
            ret.append(curr_group)
        # TODO: sort by frequencies. Trend should be first, freq=1 - next, etc.
        self.log.info("grouped items: %s" % repr(ret))
        return ret

    def _check_ssa_keywords(self, fname, avail, kargs):
        ''' check that keywords parameter has keys only from 'avail'.
            Return tuple (True/False, err string)
        '''
        invalid = []
        for x in kargs.keys():
            if not x in avail:
                invalid.append(x)
        if len(invalid):
            e = "{}: - invalid arg(s):{}".format(fname, invalid)
            self.log.error(e)
            raise PSSAError(e)
        if 'N' in kargs.keys():
            N = int(kargs['N'])
        else:
            N = self.get_config_N()
        if 'L' in kargs.keys():
            L = int(kargs['L'])
        else:
            L = self.get_config_L()
        if 'offset' in kargs.keys():
            offset = int(kargs['offset'])
        else:
            offset = self.get_config_data_offset()
        if 'send_fin_msg' in kargs.keys():
            send_fin_msg = kargs['send_fin_msg']
        else:
            send_fin_msg = True
        if N < 0 or L < 0 or offset < 0:
            e = "at least one arg {} <= 0, impossible to calc ssa".format(
                    N, L , offset)
            self.log.error(e)
            raise PSSAError(e)
        if self.cfg.common.data_size < N:
            raise PSSAError('data size too small')
        if L >= N:
            raise PSSAError('window size(L) too large')
        return (N, L, offset, send_fin_msg)

    def __calculate_ssa(self, **kargs):
        ''' calculate SVD decomposition.
            keyword arguments:
                'N' data size;
                'L' - window size;
                'offset' - data offset;
                'send_fin_msg' - send 'final' message when calc done;
            When keyword arg is not set, value will be got from
            current ssa parameters.
        '''
        try:
            N, L, offset, send_fin_msg = self._check_ssa_keywords("__calculate_ssa",
                    ['N', 'L', 'offset', 'send_fin_msg'], kargs)
            self.ssa_log_msg.emit("algo: calc ssa started")
            self.ssa_step_start.emit(offset, N, L)
            self.ssa = None
            self.u_fft = None
            self.u_fft_pw = None
            self.predict_possible = False
            self.predict_errors = None
            self.u_correl = None
            self.w_correl = None
            self.predict_errors = None  # array of differences of F-F(1) or F(2)

            # create data
            if not self.data_src:
                self.data_src = data_source.create_data_source(self.cfg)

            u_fft = None      # matrix of U vector spektres
            u2_fft = None     # power of 2 of fft decomposition
            ssa_obj = None    # SSA object
            s_groups = []     # groups of eigenvectors

            # produce SSA from data
            self.ssa = ssa_algo.SSA(self.cfg, self.loglevel)
            self.ssa.configure(self.cfg)
            self.ssa.L = L
            self.ssa.N = N
            self.ssa.data_offset = offset
            self.ssa.ydata = self.data_src.make()
            hmatrix = self.ssa.prepare_hankel_matrix()
            self.ssa.make_ssa(hmatrix, 0)
            self.ssa.report()
            if self.preferred_rank != 0:
                self.set_s_size(self.preferred_rank)
            else:
                rank, w_min = self.rank_by_w_correl()
                self.ssa.set_s_size(rank)
            if send_fin_msg:
                self.prepare_predict()
                self.ssa_calc_finished.emit(offset, N, L)
            self.ssa_log_msg.emit("algo: ssa calc done")
#            self.yieldCurrentThread()
        except (ssa_algo.SSAErrorDataRange, PSSAError), err:
            self.log.error(err)
            self.ssa_log_error.emit(str(err))
            self.ssa_calc_finished.emit(0, 0, 0)
        except Exception, e:
            self.log.error('Exception: {}'.format(e))
            self.ssa_log_error.emit('Exception: {}'.format(e))
            self.ssa_calc_finished.emit(0, 0, 0)

    def __calculate_w_optimized_ssa(self, **kargs):
        try:
            N, L, offset, unused = self._check_ssa_keywords("__calculate_w_optimized_ssa()",
                    ['N', 'L', 'offset'], kargs)
            self.ssa_log_msg.emit("algo: w-opt ssa calc started")
            N_cnf = self.get_config_N()
            offset = self.get_config_data_offset()
            deltaN = min(10, max(2, (N_cnf / 20)))
            if N_cnf - deltaN * self.SSA_W_OPT_STEPS / 2 < 4:
                self.log.error("invalid calc parameters: N too small")
                self.ssa_calc_finished.emit(0, 0, 0)
                return False
            N0 = N_cnf - deltaN * (self.SSA_W_OPT_STEPS / 2)
            if offset - (N_cnf - N0) < 0:
                self.log.error("invalid calc parameters: offset too small")
                self.ssa_calc_finished.emit(0, 0, 0)
                return False
            # delta N should be even
            deltaN = deltaN & (~1)
            self.log.info("ssa optimized: N_cnf={}. N0={}, deltaN={}, steps={}".format(
                    N_cnf, N0, deltaN, self.SSA_W_OPT_STEPS))
            # ssa_probe is a list of [ssa, min_w_correl, idx_min_w_corr]
            ssa_probe = []
            for i in range(self.SSA_W_OPT_STEPS):
                probe_elem = []
                N = N0 + deltaN * i
                offs = offset - (N - N_cnf)
                self.ssa_log_msg.emit("## ssa calc[{}] started".format(i))
                self.log.info("ssa optimized, step {}, N:{}".format(i, N))
                self.__calculate_ssa(send_fin_msg=False, N=N, L=N/2, offset=offs)
                rank, w_min = self.rank_by_w_correl()
                ssa_probe.append([self.ssa, w_min, rank])
                self.log.info("w-correl min at [{}]:{}".format(rank, w_min))
                self.ssa_log_msg.emit("## ssa calc[{}] stopped".format(i))
                self.ssa_step_done.emit(rank, w_min)
            # select appropriate ssa by w_min
            w_min = 1e10
            ssa_len = -1
            i_opt = -1
            for i in range(len(ssa_probe)):
                if ssa_probe[i][1] < w_min:
                    w_min = ssa_probe[i][1]
                    i_opt = i
            rank = ssa_probe[i_opt][2]
            self.ssa = ssa_probe[i_opt][0]
            self.ssa.set_s_size(rank)
            self.log.info("selected ssa by w_correl {}, pass:{} - N:{}, L:{}, offset:{}, svd size:{}"
                    .format(w_min, rank, self.ssa.N, self.ssa.L,
                        self.ssa.data_offset, ssa_len))
            self.prepare_predict()
            self.ssa_log_msg.emit("algo: w-opt ssa calc finished")
            self.ssa_calc_finished.emit(self.ssa.data_offset, self.ssa.N, self.ssa.L)
        except (ssa_algo.SSAErrorDataRange, PSSAError), err:
            self.log.error(e)
            self.ssa_log_error.emit(e)
            self.ssa_calc_finished.emit(0, 0, 0)
        except Exception, e:
            self.log.error('Exception: {}'.format(e))
            self.ssa_log_error.emit('Exception: {}'.format(e))
            self.ssa_calc_finished.emit(0, 0, 0)


    def calc_u_fft_correl(self):
        ''' calculate fft correlation of U-vectors '''
        if not self.ssa:
            return
        self.u_fft = utils.fft_on_list(self.ssa.u_eigen_vectors())
        self.u_fft_pw = self.u_fft ** 2

        # calculate correlation of fft power pairs
        self.u_correl = utils.calc_fft_correlations(self.u_fft_pw)
        self.debug_array(self.u_correl, "correlation coeff")

        # choose groups by correlation coefficients.
        s_groups = self.grouping_items(self.u_correl)

        # create unit matices from groups above.
        # !!! Currently those groups unused !!!
        self.ssa.assign_groups(s_groups)
        s_opt_sz = len(self.ssa.s_svd_opt())
        self.ssa.create_group_matrices(s_opt_sz)
        self.ssa.merge_main_groups()

        # diagonal averaging of grouped matrixes.
        res_vectors = self.ssa.diag_average()

    def calc_w_unit_correl_matrix(self, rank_max, m_size):
        ''' calc omega-correlation matrix of X(i) from unit matrices.
            Matrix has simmetry around main diagonal and main diagonal is ones.
            @param rank_max - calculation depth
            @param m_size - matrix dimension X = Y
            result is self.w_correl matrix
        '''
        if not self.ssa:
            return
        if m_size < 1:
            raise PSSAError("calc_w_unit_correl_matrix(): count of U < 1")
        if m_size > len(self.s_svd()):
            e = "Could not calc w-correl matrix: calc size {} > {}".format(
                    m_size, len(self.s_svd()))
            self.log.error(e)
            raise PSSAError(e)
        self.log.info("calc_w_unit_correl_matrix(): rank_max: {}, m_size: {}"
                .format(rank_max, m_size))
        self.ssa.create_unit_matrices(m_size)
        u_funcs = self.ssa.diag_average_unit_matrices()
        self.log.debug("cacl_w_unit_correl_matrix(): u_funcs:")
        for f in u_funcs:
            self.debug_array(f, "****")
        self.w_correl = np.zeros(np.power(m_size, 2)).reshape(m_size, m_size)
        for r in range(rank_max):
            for c in range(r, m_size):
                if r == c:
                    self.w_correl[r, c] = 1
                    continue
                # note: correlation has range [-1, 1]. According to Golyandina,
                # ssa_an.pdf, example, absolute values is used
                self.w_correl[r, c] = abs(self.ssa.calc_w_correl(u_funcs[r], u_funcs[c]))
                self.w_correl[c, r] = self.w_correl[r, c]
        self.log.debug("omega-correlation matrix:")
        self.debug_matrix(self.w_correl, "omega-correlations")

    def calc_w_unit_correl_weights(self, rank_max):
        ''' calculate w_correl weights from self.w_correl.
            Calculation depth is bounded by rank_max
            Return array of correl weights. Array size is rank_max.
            x---++++    + - area of weight(i) calculation
            -x--++++
            --x-++++
          (i)--x++++
            ----x---
            -----x--
            ------x-
            -------x
              (i)
        '''
        if self.w_correl is None:
            return np.array([])
        res = np.zeros(rank_max)
        cols = self.w_correl.shape[1]
        for r in range(rank_max):
            weight = 0
            for i in range(r+1):
                for c in range(r+1, cols):
                    weight += self.w_correl[i,c]
            res[r] = weight
        return res

    def rank_by_w_correl(self):
        ''' find optimal rank by:
            2) calculate summ of correlations for each row of self.w_correl
            3) find index where those summ is minimal. This is a rank
            @return pair of (calculated rank of SVD, w-correlation for that rank))
            Note: rank >= 1
        '''
        # w-correlation matrix has simmetry about main diagonal.
        # Coefficients at main diagonal always 1(self correlation).
        # Coefficients at (i+1,j) where j=1(close to main diagonal) can
        # be viewed as correlation of cos/sin of the same frequency
        # when coefficient more 0.5 and as noise in other case.
        # Assumptions:
        # - calculation is made in one half of matrix,
        # - main diagonal excluded from calculation.
        # - for element (i,i) calculation made for elements (i+1, i+wd), where
        #   wd is a w-correlation depth.
        # - No more than wlen elements analyzed. Assume rank of S no more
        #   than rank_max.
        # - according to condition above, size of analized SVD is
        #   wd + rank_max.
        # - correlations should be calculated at early step because
        #   the same value is used ofter more than one time.
        if not self.ssa:
            return 0, 0
        # calc w-correl limit
        rank_max = min(len(self.s_svd()) - 5, self.SSA_RANK_MAX)
        if rank_max < 0:
            self.log.error("Could not calc w-correl - S too short")
        if rank_max + self.W_MIN_DEPTH > len(self.s_svd()):
            m_size = len(self.s_svd())
        else:
            m_size = rank_max + self.W_MIN_DEPTH
        if rank_max > len(self.s_svd()):
            rank_max = len(self.s_svd())

        if self.w_correl is None:
            self.calc_w_unit_correl_matrix(rank_max, m_size)
        weights = self.calc_w_unit_correl_weights(rank_max)
        log_array(weights, "weights of w-correlation")
        if self.USE_W_CORR_WEIGHTS:
            self.log.info("correction weights by square")
            for i in range(weights.size):
                square = (i + 1) * (m_size - i)
                weights[i] /= square
            log_array(weights, "normalized by square weights of w-correlation")
        w_min = 1e10
        rank = -1
        # first and second values excluded, because:
        # 1) anyway, this value in most cases in small;
        # 2) rank = 1 is not interested in most cases.
        if weights.size < 3:
            imin = 1
        else:
            imin = 3
        for i in range(imin, weights.size):
            if weights[i] < w_min:
                rank = i
                w_min = weights[i]
        return (rank + 1, w_min)

    def data_distrib(self):
        ''' find data distribution in some range.
            @param offset: data offset from begin
            @param size:   analized size of data.
            @return  list as [array-like x, array-like y]
        '''
        x, y = self.get_N_data()
        y_r = np.around(y, self.DATA_DECIMALS)
        y_r.dump("y_data")
        self.log.info("data ({})saved to y_data".format(y_r.size))
        return [np.array([]), np.array([])]

    def calc_m_multistart_err(self, M):
        ''' Estimate interval for F(1) by M-step multistart continue.
            Short description:
            1) Data for prediction has length K-M.
            2) Data for prediction is a F(1) values begins from F(1)[i],
            where i is 0..L-2 ('i' is a steps count).
            3) Prediction is made on 'M' length.
            Last value of prediction will be at N-1.
            @param M - step multistart value.
            @return  tuple of array-like 'x' and 'y'.

        '''
        if self.ssa is None:
            return np.array([]), np.array([])
        if M <= 0:
            raise PSSAError("calc_m_mutlistart_err(): M <= 0({})".format(M))
        K = self.ssa.N - self.ssa.L + 1
        x = []
        y = []
#        print "ssa.M: {}, L: {}, K: {}, ssa.predict_poly: {}".format(M, self.ssa.L, K,
#                len(self.ssa.predict_poly))
        for i in range(K - M):
#            print("calc_m_..., i:{}".format(i))
            self.ssa.reset_predict_by_F1(i)
            px = -1
            py = -1
            for j in range(M):
                px, py = self.ssa.predict()
            x.append(px)
            y.append(py)
        # 'y' is the predicted values. Next step - calculate difference between
        # those values and input data
        y_in = self.ssa.ydata[x[0]:x[0]+len(x)]
        y_diff = y_in - y
        return np.array(x), np.array(y_diff)

