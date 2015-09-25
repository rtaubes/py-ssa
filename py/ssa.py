#!/usr/bin/env python

import os.path
import sys
import argparse
import logging
sys.path.append(os.path.join(sys.path[0], "modules"))
import pssa
import numpy as np
import pylab as p
from coeff_dlg import CoeffDlg

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSignal  # ,QPoint
from ui_main_dlg import *
import sys
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from config import *


class PlotError(Exception):
    pass


class PlotDlg(QtGui.QDialog):
    COLORS = {'black':'#000000', 'blue':'#0000ff', 'red':'#ff0000',
            'green':'#00ff00', 'gray':'#585858'}

    fig_closing = pyqtSignal(int, int, int)

    def __init__(self, parent, title, view_type, x, y):
        super(PlotDlg, self).__init__(parent)
        self.log = logging.getLogger('MainWnd')
        self.resize(400, 600)
        self.setWindowTitle(title)
        self.setVisible(True)
        self.setModal(False)
        self.figure = plt.figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(
                                QtGui.QSizePolicy.Expanding,
                                QtGui.QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.view_type = view_type
        self.finished.connect(self.about_to_close)
        self.axes = self.figure.add_subplot(111)
        self.axes.hold(False)
        self.colors_lst = PlotDlg.COLORS.values()
        if parent:
            self.fig_closing.connect(parent.fig_closing)
        print "set PlotDlg to {}/{}".format(x, y)
        if x != 0:
            self.move(x, y)

    def about_to_close(self):
        p = self.pos()
        self.fig_closing.emit(self.view_type, p.x(), p.y())


class XYPlotDlg(PlotDlg):
    ''' plot figures at one axis.
        @param  parent - parent widget
            Sample for 2 lines: [[x00,x01,x02], [y01,y02,y03],
                               [x11,x12],[y11,y12]]
        @param  title  - title of figure
        @param  view_type - type of view
    '''
    def __init__(self, parent, title, view_type, x, y, marker=None):
        super(XYPlotDlg, self).__init__(parent, title, view_type, x, y)
        self.log = logging.getLogger("XYPlotDlg")
        self.lines2d = {}     # line name->Line2D
        self.marker = marker
        self.axes.grid(True)

    def add_line(self, line_name, **kw_args):
        ''' return Line2D object
            @param kw_args - Line2D keyword args.
            Example of Line2D arguments: color, label, linewidth, linestyle,
            xdata, ydata.
            Short about color:
            1) base colors: b(blue),g(green),r(red),c(cyan),m(magenta),y(yellow),
            k(black),w(white)
            2) '#xxxxxx' or (r, g, b), where each component in range [0,1].
            3) legal html names: 'red',... are supported.
            See Line2D documentation.
        '''
        if line_name in self.lines2d.keys():
            return self.lines2d[line_name]
        line = Line2D(**kw_args)
        self.axes.add_line(line)
        self.axes.autoscale(True)
        x0, x1 = self.axes.get_xlim()
        dx = (x1 - x0) / 50.0
        self.axes.set_xlim(x0-dx, x0+dx)
        self.lines2d[line_name] = line
        return line

    def get_line2d(self, line_name):
        if not line_name in self.lines2d.keys():
            self.log.error("could not find line2d by name {}".format(line_name))
            return None
        return self.lines2d[line_name]

    def draw(self, **data):
        ''' draw data set 'data'. Helper for add_line for line array.
            @param data - data array as line_name -> keywords parameter
                          for add_line().
            When line is not exists, it will be added
        '''
        xmin = 1e10
        xmax = -1e10
        ymin = 1e10
        ymax = -1e10
        for ln_name in data.keys():
            if ln_name == '_x_limits_':
                ymin = data[ln_name]['ymin']
                ymax = data[ln_name]['ymax']
                xmin = data[ln_name]['xmin']
                xmax = data[ln_name]['xmax']
                continue
            line_data = data[ln_name]
            ln2d = self.add_line(ln_name, **line_data)
        if xmin == 1e10:
            self.axes.relim()
            self.axes.autoscale(enable=True, tight=True, axis='both')
        else:
            self.axes.relim()
            self.axes.autoscale(enable=True, axis='y', tight=False)
            self.axes.set_xlim(xmin, xmax)
            self.axes.set_ylim(ymin, ymax)
            print "ymin/ymax: {}/{}".format(ymin, ymax)

    def redraw(self, **data):
        ''' redraw lines. Only 'xdata' and 'ydata' in 'data' allowed.
            @param  data - array line_name->[x, y]
        '''
        ymin = 1e10
        ymax = -1e10
        xmin = 1e10
        xmax = -1e10
        for ln_name in data.keys():
            if ln_name == '_x_limits_':
                ymin = data[ln_name]['ymin']
                ymax = data[ln_name]['ymax']
                xmin = data[ln_name]['xmin']
                xmax = data[ln_name]['xmax']
                continue
            if not ln_name in self.lines2d.keys():
                continue
            ln_data = data[ln_name]
            x = None
            y = None
            for key in ln_data.keys():
                if key == 'xdata':
                    x = ln_data['xdata']
                elif key == 'ydata':
                    y = ln_data['ydata']
                else:
                    self.log.warning("redraw(): parameter {} ignored"
                        .format(key))
            if x is None or y is None:
                self.log.error("redraw(): missing 'xdata' or 'ydata' parameter")
                return
            self.lines2d[ln_name].set_data(x, y)
        if xmin == 1e10:
            self.axes.relim()
            self.axes.autoscale(enable=True)
        else:
            self.axes.set_xlim(xmin, xmax)
        self.canvas.draw()

class WCorrelDlg(PlotDlg):
    ''' plot color table of matrix where each value range marked
        by some color
    '''
    def __init__(self, parent, title, view_type, x, y):
        super(WCorrelDlg, self).__init__(parent, title, view_type, x, y)
        self.log = logging.getLogger("WCorrelDlg")
        self.tbl = None

    def __cell_color(self, val):
        ''' return cell color as gray [0..1].
            @param val  value in range [0..1]
        '''
        parts = 20  # step is 1 / parts = 0.05
        color = np.floor(val * parts) / parts
        if color > 1:
            self.log.error("color > 1! arg:{} color:{}".format(val, color))
            return "0.0"
        color = 1.0 - color
        return "{:f}".format(color)

    def draw(self, data_tbl):
        ''' draw matrix as colorized table
            @param table - square matrix of values [0..1]
        '''
        self.axes.axis("off")
        r, c = data_tbl.shape
        if r != c:
            self.log.error("WCorrelDlg: table ({},{}) is not a square".
                    format(r, c))
            return
        colors = [[self.__cell_color(data_tbl[ln, col]) for col in range(r)]
            for ln in range(r)]
        self.log.debug("table colors:{}".format(colors))
        self.tbl = p.table(loc='center', cellColours=colors)

    def redraw(self, data_tbl):
        ''' redraw matrix table
            @param table - square matrix of values [0..1]
        '''
        r, c = data_tbl.shape
        if r != c:
            self.log.error("WCorrelDlg: table ({},{}) is not a square".
                    format(r, c))
            return
        colors = [[self.__cell_color(data_tbl[ln, col]) for col in range(r)]
            for ln in range(r)]
        self.tbl.update(cellColours=colors)
        self.canvas.draw()


class XYListPlotDlg(QtGui.QDialog):

    fig_closing = pyqtSignal(int, int, int)

    def __init__(self, parent, title, view_type, x, y):
        super(XYListPlotDlg, self).__init__(parent)
        self.log = logging.getLogger('XYLstPDlg')
        self.resize(400, 600)
        self.view_type = view_type
        self.setWindowTitle(title)
        self.setVisible(True)
        self.setModal(False)
        self.qlayout = QtGui.QHBoxLayout(self)
        self.setLayout(self.qlayout)
        self.qscroll = QtGui.QScrollArea(self)
        self.qscroll.setGeometry(QtCore.QRect(0, 0, 400, 600))
        self.qscroll.setFrameStyle(QtGui.QFrame.NoFrame)
        self.qlayout.addWidget(self.qscroll)

        self.qscrollContext = QtGui.QWidget()
        self.qscrollLayout = QtGui.QVBoxLayout(self.qscrollContext)
        self.qscrollLayout.setGeometry(QtCore.QRect(0, 0, 1000, 1000))

        self.qscroll.setWidget(self.qscrollContext)
        self.qscroll.setWidgetResizable(True)

        if parent:
            self.fig_closing.connect(parent.fig_closing)
        if x != 0:
            self.move(x, y)

    def about_to_close(self):
        p = self.pos()
        self.fig_closing.emit(self.view_type, p.x(), p.y())


    def draw(self, data):
        figs = len(data)
        for i in range(figs):
            fig_widget = QtGui.QWidget(self.qscrollContext)
            figure = Figure((4.0, 3.0), dpi=70)  # params: size(inches), dpi
            canvas = FigureCanvas(figure)
            canvas.setParent(fig_widget)
            axes = figure.add_subplot(111, title="[%d]" % (i))
            if self.view_type == MainDlg.FOURIE_VIEW:
                line = data[i]
                axes.vlines(range(len(line)), np.zeros(len(line)), line)
                # shift left for showing zero values
                x0, x1 = axes.get_xlim()
                x0 = x0 - (x1-x0)/20
                axes.set_xlim(x0, x1)
            else:
                axes.plot(data[i], 'r')
            axes.grid()

            canvas = FigureCanvas(figure)
            canvas.setParent(fig_widget)
            layout = QtGui.QVBoxLayout()
            layout.addWidget(canvas)
            fig_widget.setLayout(layout)
            # prevent the canvas to shrink beyond a point
            # original size looks like a good minimum sizw
            canvas.setMinimumSize(canvas.size() * 0.8)
            self.qscrollLayout.addWidget(fig_widget)
        self.qscrollContext.setLayout(self.qscrollLayout)

    def redraw(self, data):
        self.qscrollContext = QtGui.QWidget()
        self.qscrollLayout = QtGui.QVBoxLayout(self.qscrollContext)
        self.qscrollLayout.setGeometry(QtCore.QRect(0, 0, 1000, 1000))

        self.qscroll.setWidget(self.qscrollContext)
        self.qscroll.setWidgetResizable(True)
        self.draw(data)


class ActiveFigures(object):
    ''' manipulation of active figures '''
    def __init__(self):
        self._figures = []
        self._saved_positions = {}  # view type -> (x, y)

    def all(self):
        return self._figures

    def get(self, view_type):
        for f in self._figures:
            if f.view_type == view_type:
                return f
        return None

    def add(self, figure):
        self._figures.append(figure)

    def remove(self, view_type):
        f = self.get(view_type)
        if f:
            self._figures.remove(f)

    def save_position(self, view_type, x, y):
        self._saved_positions[view_type] = (x, y)

    def get_position(self, view_type):
        if view_type in self._saved_positions.keys():
            return self._saved_positions[view_type]
        return (0, 0)


class MainDlg(QtGui.QDialog):
    # order of widgets on dataSourceSw
    DATA_CFG_ORDER = [CSVSourceConfig.TYPE, LinearSourceConfig.TYPE,
            SinSourceConfig.TYPE, RandomSourceConfig.TYPE]
    # types of figures. Add appropriate pair to figure_glue for auto
    # redrawing after calculation of ssa.
    DATA_VIEW      = 1
    U_VIEW         = 2
    V_VIEW         = 3
    FOURIE_VIEW    = 4
    PREDICT_VIEW   = 5
    SLOG_VIEW      = 6
    FFT_CORREL_VIEW =7
    W_CORR_VIEW    = 8
    DATA_REST_VIEW = 9
    DATA_DISTRIB_VIEW = 10

    # data for redrawing figure

    def __init__(self, loglevel, parent=None):
        super(MainDlg, self).__init__(parent)
        # For redrawing. fig type -> figure data
        # figure data is list of:
        # - redrawing function
        # - True when figure will be shown always, False - when redraw
        #   only existed figure
        self.figure_glue = {
            self.DATA_VIEW       : [self.data_show_req, False],
            self.U_VIEW          : [ self.u_eigenvect_show_req, False],
            self.V_VIEW          : [self.v_eigenvect_show_req, False],
            self.FOURIE_VIEW     : [self.fourie_show_req, False],
            self.PREDICT_VIEW    : [self.predict_show_req, True],
            self.SLOG_VIEW       : [self.slog_show_req, False],
            self.FFT_CORREL_VIEW : [self.fft_correl_show_req, False],
            self.W_CORR_VIEW     : [self.w_correl_req, False],
            self.DATA_REST_VIEW  : [self.data_rest_show_req, False],
            self.DATA_DISTRIB_VIEW  : [self.data_distrib_show_req, False]
        }
        self.config_done = False
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.active_figures = ActiveFigures()
        self.fig_id = 0
        self.config_folder = ''
        self.loglevel = loglevel
        self.log = logging.getLogger('MainDlg')
        # "calc ssa" push_button
        self.ui.calcPb.clicked.connect(self.calc_ssa)
        # "data show" push button
        self.ui.dataPb.clicked.connect(self.data_show_req)
        # estimation of M-multystart error
        self.ui.mMulitStartPb.clicked.connect(self.m_multistart_err_req)
        # "data rest show" push button
        self.ui.dataRestPb.clicked.connect(self.data_rest_show_req)
        self.ui.varsDistrPb.clicked.connect(self.data_distrib_show_req)
        # "vars distribution" push-bottm
        self.ui.evVPb.clicked.connect(self.v_eigenvect_show_req)
        self.ui.evUPb.clicked.connect(self.u_eigenvect_show_req)
        self.ui.fftPb.clicked.connect(self.fourie_show_req)
        # w-correlation push button
        self.ui.wCorrelPb.clicked.connect(self.w_correl_req)
        self.ui.sourceTypeCb.currentIndexChanged.connect(self.source_type_changed)
        self.ui.csvFileOffsetSb.valueChanged.connect(self.source_offset_changed)
        self.ui.confOpenPb.clicked.connect(self.read_config)
        self.ui.confSavePb.clicked.connect(self.save_config)
        self.ui.NSb.valueChanged.connect(self.Nvalue_changed)
        self.ui.LSb.valueChanged.connect(self.Lvalue_changed)
        self.ui.dataSzSb.setMaximum(100000)
        self.ui.NSb.setMaximum(100000)
        self.ui.LSb.setMaximum(100000)
        self.ui.predictSb.setValue(10)
        self.ui.predictSb.setMaximum(100000)
        self.ui.predictSb.setSingleStep(5)
        self.ui.csvFileOffsetSb.setMaximum(100000)
        self.ui.dataSzSb.valueChanged.connect(self.data_size_changed)
        self.ui.sLogPb.clicked.connect(self.slog_show_req)
        self.ui.logLimitCb.addItems(ShowConfig.S_SHOW_LIMITS)
        self.ui.logLimitCb.currentIndexChanged.connect(self.s_show_lim_changed)
        # csv src config
        self.ui.csvFileDataCb.currentIndexChanged.connect(self.csv_datacols_changed)
        self.ui.csvFilePb.clicked.connect(self.csv_file_open)
        # linear src config
        self.ui.linearA1Sb.valueChanged.connect(self.linear_a1_changed)
        self.ui.linearA0Sb.valueChanged.connect(self.linear_a0_changed)
        # sinus src config
        self.ui.sinAmpSb.setMaximum(3)
        self.ui.sinAmpSb.setSingleStep(0.1)
        self.ui.sinAmpSb.valueChanged.connect(self.sin_amp_changed)
        self.ui.sinPeriodSb.setMaximum(100000)
        self.ui.sinPeriodSb.valueChanged.connect(self.sin_period_changed)
        self.ui.sinPhaseSb.setMaximum(359)
        self.ui.sinPhaseSb.setMinimum(0)
        self.ui.sinPhaseSb.valueChanged.connect(self.sin_phase_changed)
        # random src config
        self.ui.randomCacheChb.stateChanged.connect(self.random_cache_changed)
        self.ui.distribCb.currentIndexChanged.connect(self.random_distrib_changed)
        self.ui.randomMinSb.valueChanged.connect(self.random_min_changed)
        self.ui.randomMaxSb.valueChanged.connect(self.random_max_changed)
        # ...
        self.pssa = None
        self.disable_all()
        self.coeff_dlg = None
        self.conf = None
        self.need_recalc_ssa = False
        self.need_recreate_ssa = True

    def random_cache_changed(self, butt_state):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != RandomSourceConfig.TYPE:
            self.log.error("config source not RandomSourceConfig")
            return
        print 'random cache butt state:', butt_state
        self.conf.sources[idx].use_cached = self.ui.randomCacheChb.isChecked()
        self.log.debug("use_cached changed to {}".format(butt_state))
        self.need_recalc_ssa = True

    def random_distrib_changed(self, idx):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != RandomSourceConfig.TYPE:
            self.log.error("config source not RandomSourceConfig")
            return
        self.conf.sources[idx].distr_type = self.ui.distribCb.currentText()
        self.log.debug("random distrib changed to {}".format(self.ui.distribCb.currentText()))
        self.need_recalc_ssa = True

    def random_min_changed(self, val):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != RandomSourceConfig.TYPE:
            self.log.error("config source not RandomSourceConfig")
            return
        self.conf.sources[idx].lim_min = val

    def random_max_changed(self, val):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != RandomSourceConfig.TYPE:
            self.log.error("config source not RandomSourceConfig")
            return
        self.conf.sources[idx].lim_max = val

    def csv_datacols_changed(self, idx):
        pass

    def sin_amp_changed(self, val):
        ''' Note: value is float '''
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != SinSourceConfig.TYPE:
            self.log.error("config source not SinSourceConfig")
            return
        self.conf.sources[idx].amp = val
        self.log.debug("amp changed to {}".format(val))
        self.need_recalc_ssa = True

    def sin_period_changed(self, val):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != SinSourceConfig.TYPE:
            self.log.error("config source not SinSourceConfig")
            return
        self.conf.sources[idx].period = val
        self.log.debug("period changed to {}".format(val))
        self.need_recalc_ssa = True

    def sin_phase_changed(self, val):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != SinSourceConfig.TYPE:
            self.log.error("config source not SinSourceConfig")
            return
        self.conf.sources[idx].phase = val
        self.log.debug("phase changed to {}".format(val))
        self.need_recalc_ssa = True

    def coeff_dlg_closed(self, unused):
        self.coeff_dlg = None

    def linear_a1_changed(self, val):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != LinearSourceConfig.TYPE:
            self.log.error("config source not LinearSourceConfig")
            return
        self.conf.sources[idx].angle = val
        self.log.debug("angle changed to {}".format(val))
        self.need_recalc_ssa = True

    def linear_a0_changed(self, val):
        idx = self.ui.sourceTypeCb.currentIndex()
        if idx >= len(self.conf.sources):
            return
        if self.conf.sources[idx].TYPE != LinearSourceConfig.TYPE:
            self.log.error("config source not LinearSourceConfig")
            return
        self.conf.sources[idx].y0 = val
        self.log.debug("y0 changed to {}".format(val))
        self.need_recalc_ssa = True

    def source_offset_changed(self, val):
        self.conf.common.data_offset = val;

    def Nvalue_changed(self, val):
        self.conf.ssa.data_range = val
        self.need_recalc_ssa = True

    def Lvalue_changed(self, val):
        self.conf.ssa.window_size = val
        self.need_recalc_ssa = True

    def data_size_changed(self, val):
        self.need_recalc_ssa = True
        self.conf.common.data_size = val

    def read_config(self):
        fn = QtGui.QFileDialog.getOpenFileName(self, "Open config", "",
                "Configs (*.cfg)")
        if fn == '' or self.ui.configFileLe.text() == fn:
            return
        self.need_recreate_ssa= True
        self.config_done = False
        self.ui.configFileLe.setText(fn)
        self.conf = Config()
        self.conf.parse(fn)
        self.ui.dataSzSb.setValue(self.conf.common.data_size)
        self.ui.predictSb.setValue(self.conf.common.predict)
        self.ui.NSb.setValue(self.conf.ssa.data_range)
        self.ui.LSb.setValue(self.conf.ssa.window_size)
        self.ui.sourceTypeCb.clear()
        self.ui.sourceTypeCb.setCurrentIndex(-1)
        # sourceTypeCb indexes is the config.sources indexes too.
        # View type is defined by current config.source.
        # Views order of dataSourceSw is defined when form
        # created and should be the same as VIEW_ORDER
        # 1) Fill sourceTypeCb by source type/name
        for src in self.conf.sources:
            src_name = ""
            if src.TYPE == SinSourceConfig.TYPE:
                src_name = "sinusPage"
            elif src.TYPE == LinearSourceConfig.TYPE:
                src_name = "linearPage"
            elif src.TYPE == CSVSourceConfig.TYPE:
                src_name = "csvPage"
                self.log.debug("csvPage page added to config")
            elif src.TYPE == RandomSourceConfig.TYPE:
                src_name = "randomPage"
            src_name = "{}/{}".format(src.sname, src.TYPE)
            print "src_name:", src_name
            self.ui.sourceTypeCb.addItem(src_name)
        idx = self.conf.source_idx
        if idx == Config.COMBINED_IDX:
            self.ui.combinedSrcLbl.setText("Combined")
            idx = 0
        else:
            self.ui.combinedSrcLbl.setText("Separated")
        self.enable_all()
        self.log.debug("read_config(): idx:%d" % idx)
        self.ui.sourceTypeCb.setCurrentIndex(idx)
        self.ui.csvFileOffsetSb.setValue(self.conf.common.data_offset)
        self.config_done = True
        self.set_source_by_id(idx)
        self.ui.logLimitCb.setCurrentIndex(self.conf.show.get_idx())

    def s_show_lim_changed(self, idx):
        if idx >= len(ShowConfig.S_SHOW_LIMITS):
            return
        self.conf.show.s_show_limit = int(ShowConfig.S_SHOW_LIMITS[idx])

    def set_source_by_id(self, idx):
        ''' Show data source widget by index of config in conf.sources '''
        self.log.debug("set_source_by_id(): idx:%d" % idx)
        if idx >= len(self.conf.sources):
            msg = "no such source [%d] - ignored" % idx
            self.log.info(msg)
            return
        src_conf = self.conf.sources[idx]
        if src_conf.TYPE == SinSourceConfig.TYPE:
            self.ui.sinAmpSb.setValue(src_conf.amp)
            self.ui.sinPeriodSb.setValue(src_conf.period)
            self.ui.sinPhaseSb.setValue(src_conf.phase)
        elif src_conf.TYPE == LinearSourceConfig.TYPE:
            self.ui.linearA0Sb.setValue(src_conf.y0)
            self.ui.linearA1Sb.setValue(src_conf.angle)
        elif src_conf.TYPE == RandomSourceConfig.TYPE:
            self.ui.randomMinSb.setValue(src_conf.lim_min)
            self.ui.randomMaxSb.setValue(src_conf.lim_max)
            self.ui.randomCacheChb.setChecked(src_conf.use_cached)
        elif src_conf.TYPE == CSVSourceConfig.TYPE:
            self.ui.csvFileLe.setText(src_conf.filename)
            if src_conf.use == 'high':
                i = 0
            elif src_conf.use == 'low':
                i = 1
            elif src_conf.use == 'average':
                i = 2
            else:
                err = "unsupported use %s" % src_conf.use
                self.log.error(err)
                i = 2
        else:
            err = "unknown src type {}".format(src_conf.TYPE)
            self.log.error(err)
        # setup data widget idx by config idx
        dview_idx = -1
        for i in range(len(self.DATA_CFG_ORDER)):
            if self.DATA_CFG_ORDER[i] == src_conf.TYPE:
                dview_idx = i
                break
        if dview_idx == -1:
            self.log.error("could not define data view index form conf.idx {}" +
                    " and conf type {}".format(idx, src_conf.TYPE))
            dview_idx = 0
        self.ui.dataSourceSw.setCurrentIndex(dview_idx)

    def csv_file_open(self):
        fn = QtGui.QFileDialog.getOpenFileName(self, "Open config", "",
                "CSV (*.csv)")
        if fn == '' or self.ui.csvFileLe.text() == fn:
            return
        self.need_recalc_ssa = True
        self.ui.csvFileLe.setText(fn)

    def source_type_changed(self, item_no):
        # item_no is integer
        self.log.debug("source_type_changed(): item_no:%d, current:%d" %
                (item_no, self.ui.dataSourceSw.currentIndex()))
        if self.config_done:
            self.set_source_by_id(item_no)
            self.conf.source_idx = item_no
        self.need_recalc_ssa = True

    def save_config(self):
        pass

    def _redraw_figure(self, title, fig_type, **line_args):
        ''' replace data on existed figure or create new one '''
        if not self.pssa:
            return
        f = self.active_figures.get(fig_type)
        if not f:
            x, y = self.active_figures.get_position(fig_type)
            f = self.newFigure(title, fig_type, x, y)
            f.draw(**line_args)
            self.active_figures.add(f)
        else:
            f.redraw(**line_args)

    def w_correl_req(self):
        ''' show w-correlation '''
        if not self.pssa:
            self.log.info("could not show w-correl - no SSA")
            return
        if self.pssa.w_correl is None:
            self.log.info("w-correl don't calculated before - doing that")
            self.pssa.rank_by_w_correl()
            return
        lim = int(self.conf.show.s_show_limit)
        sz = min(lim, len(self.pssa.s_svd()))
        self.log.info("w_correl_req(), sz: {}".format(sz))
        # pssa.w_correl is np square matrix sz x sz
        f = self.active_figures.get(self.W_CORR_VIEW)
        if f:
            f.done(0)
            del(f)
            f = None
            self.active_figures.remove(self.W_CORR_VIEW)
        if not f:
            x, y = self.active_figures.get_position(self.W_CORR_VIEW)
            f = self.newWCorrelFigure("W-correl", self.W_CORR_VIEW, x, y)
            f.draw(self.pssa.w_correl)
            self.active_figures.add(f)

    def slog_show_req(self):
        ''' show log(s) '''
        if not self.pssa:
            return
        lim = int(self.conf.show.s_show_limit)
        y_raw = self.pssa.s_svd()[:lim]
        y = np.log10(y_raw)
        line_data = [range(y.size), y]
        data = {}
        lines = {}
        data['xdata'] = range(y.size)
        data['ydata'] = y
        data['marker'] = '*'
        lines['slog'] = data
        self._redraw_figure("Log(s)", self.SLOG_VIEW, **lines)

    def predict_show_req(self):
        if not self.pssa:
            return
        steps = self.ui.predictSb.value()
        lines = {}
        x_f1, y_f1 = self.pssa.calc_F1()
        xy_N = self.pssa.get_N_data()
        x_p = []
        y_p = []
        ymin = 1e10
        ymax = -1e10
        self.pssa.reset_predict()
        for i in range(steps):
            _x, _y = self.pssa.predict()
            x_p.append(_x)
            y_p.append(_y)

        src_len = len(xy_N[0]) + steps + 256
        offs = max(0, self.conf.common.data_offset - 4)
        x_src, y_src = self.pssa.data()
        dataB = {}
        dataB['xdata'] = x_src
        dataB['ydata'] = y_src
        dataB['color'] = '#000000'
        lines['source'] = dataB
        x_view_min = offs
        x_view_max = offs + self.pssa.get_N() + steps * 3
        x_view_max = min(x_view_max, len(x_src))
        ymin = min(y_src[x_view_min : x_view_max])
        ymax = max(y_src[x_view_min : x_view_max])

        dataC = {}
        dataC['xdata'] = xy_N[0]
        dataC['ydata'] = xy_N[1]
        dataC['color'] = 'r'
        dataC['linewidth'] = 2
        lines['N'] = dataC
        ymin = min(ymin, min(dataC['ydata']))
        ymax = max(ymax, max(dataC['ydata']))

        dataA = {}
        dataA['xdata'] = x_p
        dataA['ydata'] = y_p
        dataA['color'] = '#00ff00'  # green
        lines['predict'] = dataA
        ymin = min(ymin, min(dataA['ydata']))
        ymax = max(ymax, max(dataA['ydata']))

        dataD = {}
        dataD['xdata'] = x_f1
        dataD['ydata'] = y_f1
        dataD['color'] = 'b'
        lines['F1'] = dataD
        ymin = min(ymin, min(dataD['ydata']))
        ymax = max(ymax, max(dataD['ydata']))

        self.pssa.calc_rest()
        err_lim = max(self.pssa.predict_errors[1]) * 0.5

        dataE = {}
        dataE['xdata'] = x_p
        dataE['ydata'] = y_p + err_lim
        dataE['color'] = 'g'
        lines['predict_range_h'] = dataE
        ymin = min(ymin, min(dataE['ydata']))
        ymax = max(ymax, max(dataE['ydata']))

        dataG = {}
        dataG['xdata'] = x_p
        dataG['ydata'] = y_p - err_lim
        dataG['color'] = 'g'
        lines['predict_range_l'] = dataG
        ymin = min(ymin, min(dataG['ydata']))
        ymax = max(ymax, max(dataG['ydata']))

        dataF = {}
        dataF['ymin'] = ymin
        dataF['ymax'] = ymax
        dataF['xmin'] = x_view_min
        dataF['xmax'] = x_view_max
        lines['_x_limits_'] = dataF

        self._redraw_figure("Predict", self.PREDICT_VIEW, **lines)

    def data_distrib_show_req(self):
        line_data = self.pssa.data_distrib()
        ld = {}
        lines = {}
        ld['xdata'] = line_data[0]
        ld['ydata'] = line_data[1]
        ld['color'] = 'b'
        lines['distrib'] = ld
        self._redraw_figure("Data distrib", self.DATA_DISTRIB_VIEW, **lines)

    def m_multistart_err_req(self):
        lines = {}
        dataA = {}
        lineA = self.pssa.calc_rest()
        dataA['xdata'] = lineA[0]
        dataA['ydata'] = lineA[1]
        dataA['color'] = 'k'
        lines['multistart_err'] = dataA

        lineB = self.pssa.calc_m_multistart_err(M=10)
        dataB = {}
        dataB['xdata'] = lineB[0]
        dataB['ydata'] = lineB[1]
#        print("ssa.m_multistart..., x.shape: {}, len(x): {}, len(y): {}"
#                .format(line_data[0].shape, len(line_data[0]), line_data[1].size))

        dataB['color'] = 'g'
        lines['F(2)'] = dataB
        self._redraw_figure("Multistart error", self.DATA_REST_VIEW, **lines)

    def data_rest_show_req(self):
        line_data = self.pssa.calc_rest()
        ld = {}
        lines = {}
        ld['xdata'] = line_data[0]
        ld['ydata'] = line_data[1]
        ld['color'] = 'k'
        lines['rest'] = ld
        self._redraw_figure("Data rest", self.DATA_REST_VIEW, **lines)

    def data_show_req(self):
        line_data = self.pssa.data()
        ld = {}
        lines = {}
        ld['xdata'] = line_data[0]
        ld['ydata'] = line_data[1]
        ld['color'] = 'k'
        lines['data'] = ld
        self._redraw_figure("Data", self.DATA_VIEW, **lines)

    def fft_correl_show_req(self):
        ''' show fft correlations of fft(U) '''
        if not self.pssa:
            return
        self.log.info("calc u-correlation")
        self.pssa.calc_u_fft_correl()
        lim = int(self.conf.show.s_show_limit)
        y = self.pssa.u_correl[:lim]
        line_data = [range(y.size), y]
        self._redraw_figure("FFT correl", self.FFT_CORREL_VIEW, line_data, '*')

    def v_eigenvect_show_req(self):
        data = self.pssa.v_eigen_vectors()
        self._redraw_figure("V eigenvectors", self.V_VIEW, data)

    def u_eigenvect_show_req(self):
        data = self.pssa.u_eigen_vectors()
        if data is None:
            return
        self._redraw_figure("U eigenvectors", self.U_VIEW, data)

    def fourie_show_req(self):
        if self.pssa.u_fft_pw is None:
            self.log.info("fourie_show_req(): data is None")
            return
        if not self.pssa:
            return
        self._redraw_figure("FFT(U)", self.FOURIE_VIEW, self.pssa.u_fft_pw)

    def fig_closing(self, view_type, x, y):
        ''' slot about closing figure window '''
        print "closing %d..." % view_type
        self.active_figures.remove(view_type)
        self.active_figures.save_position(view_type, x, y)

    def enable_all(self):
        self.ui.dataPb.setEnabled(True)
        self.ui.sourceTypeCb.setEnabled(True)
        self.ui.confSavePb.setEnabled(True)
        self.ui.dataSzSb.setEnabled(True)
        self.ui.NSb.setEnabled(True)
        self.ui.LSb.setEnabled(True)
        self.ui.calcPb.setEnabled(True)
        self.ui.evVPb.setEnabled(True)
        self.ui.evUPb.setEnabled(True)
        self.ui.fftPb.setEnabled(True)
        self.ui.predictSb.setEnabled(True)
        self.ui.csvFileOffsetSb.setEnabled(True)

    def disable_all(self):
        self.ui.dataPb.setEnabled(False)
        self.ui.sourceTypeCb.setEnabled(False)
        self.ui.confSavePb.setEnabled(False)
        self.ui.dataSzSb.setEnabled(False)
        self.ui.NSb.setEnabled(False)
        self.ui.LSb.setEnabled(False)
        self.ui.calcPb.setEnabled(False)
        self.ui.evVPb.setEnabled(False)
        self.ui.evUPb.setEnabled(False)
        self.ui.fftPb.setEnabled(False)
        self.ui.predictSb.setEnabled(False)
        self.ui.csvFileOffsetSb.setEnabled(False)

    def add_logEd_msg(self, msg):
        self.ui.logEd.appendHtml("<font color='#0000ff'>(*)</font> {}"
                .format(msg))

    def add_logEd_blue_msg(self, msg):
        self.ui.logEd.appendHtml("<font color='#0000ff'>(*) {}</font>"
                .format(msg))

    def add_logEd_error(self, msg):
        self.ui.logEd.appendHtml("<font color='#ff0000'>(*)</font> {}"
                .format(msg))

    def sig_ssa_step_start(self, offset, N, L):
        ''' signal about current step of ssa optimization if starte.
            parameters - calculation parameters
        '''
        self.ui.logEd.appendHtml("<font color='#0000ff'>(*)</font> ssa start, off:{}, N:{}, L:{}"
                .format(offset, N, L))

    def sig_ssa_step_done(self, rank, w_val):
        ''' signal about current step of ssa optimization is done.
            parameters - ssa characteristics
        '''
        self.add_logEd_msg("ssa done, rank:{}, w-correl:{}".format(rank, w_val))

    def _ssa(self):
        ''' calc ssa by simple algo, rank from CoeffDlg is used '''
        if self.pssa is None:
            self.pssa = pssa.PSSA(self, self.conf, self.loglevel)
        # start ssa thread
        if not self.coeff_dlg is None:
            s_sz = self.coeff_dlg.calc_checked_s_parts()
            self.pssa.preferred_rank = s_sz
        else:
            self.pssa.preferred_rank = 0
        self.pssa.do_cmd("calculate_ssa")
        self.pssa.start()

    def _ssa_w_optimized(self):
        ''' calc ssa optimized by w-correlation '''
        if self.pssa is None:
            self.pssa = pssa.PSSA(self, self.conf, self.loglevel)
        self.pssa.do_cmd("calculate_w_optimized_ssa")
        self.pssa.start()

    def calc_ssa(self):
        self.disable_all()
        # When data has changed, recalc SSA and reset results.
        if self.need_recreate_ssa:
            self.pssa = None
            self.need_recreate_ssa = False
        if self.need_recalc_ssa or self.ssa.obj is None:
            self.sig_ssa_startstop_msg("start SSA calc")
            try:
                if self.ui.calcRankCb.isChecked():
                    self._ssa_w_optimized()
                else:
                    self._ssa()
            except pssa.PSSAError, e:
                QtGui.QMessageBox.critical(None, "SSA calc", str(e))
                self.enable_all()
                return

    def sig_ssa_startstop_msg(self, msg):
        self.add_logEd_blue_msg(msg)

    def sig_ssa_msg(self, msg):
        ''' signal with some message '''
        self.add_logEd_msg(msg)

    def sig_ssa_error(self, msg):
        ''' signal with some error '''
        self.add_logEd_error(msg)

    def sig_ssa_calc_finished(self, offset, N, L):
        ''' signal about ssa calc is finished.
            parameters - ssa optimized parameters for optimized algo
                         or initial parameters for one-step algo
        '''
        self.log.debug("ssa calc thread is about to finish. waiting...")
        self.pssa.wait()
        if offset == 0 and N ==0 and L == 0:
            # just error
            self.sig_ssa_error("SSA calc failed")
            self.enable_all()
            return
        self.sig_ssa_startstop_msg("SSA calc done. Opt offs:{}, N:{}, L:{}"
                .format(offset, N, L))
        self.log.info("ssa calc thread finished")
        if self.coeff_dlg is None:
            self.coeff_dlg = CoeffDlg(self)
        self.coeff_dlg.set_s_vector(self.pssa.s_svd(), self.pssa.s_svd_opt().size)
        self.coeff_dlg.set_poly_roots(self.pssa.get_poly_roots())
        self.coeff_dlg.finished.connect(self.coeff_dlg_closed)
        self.coeff_dlg.show()
        self.enable_all()
        # redraw figures
        for f_type in self.figure_glue.keys():
            if self.figure_glue[f_type][1]:
                # always draw
                self.figure_glue[f_type][0]()
                continue
            # redraw existed figures
            for fig in self.active_figures.all():
                if f_type == fig.view_type:
                    self.figure_glue[f_type][0]()

    def newFigure(self, title, view_type, x, y, marker=None):
        dlg = XYPlotDlg(self, title, view_type, x, y, marker)
        print "marker:", marker
        dlg.show()
        return dlg

    def newListFigure(self, title, view_type, x, y):
        ''' show list of figures.
            @param title - title of child window
            @param view_type - type of view.
        '''
        dlg = XYListPlotDlg(self, title, view_type, x, y)
        dlg.show()
        return dlg

    def newWCorrelFigure(self, title, view_type, x, y):
        ''' show w-correlation as table '''
        dlg = WCorrelDlg(self, title, view_type, x, y)
        dlg.show()
        return dlg


if __name__ == '__main__':
    prog_path = sys.path[0] # Path to this scripts independent from path
                            # where start is done.
    prog_name = sys.argv[0]
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-v', '--verbose', action='store_const',
            const=1, default=0, help="show debug info")
    parser.add_argument(
            '-t', '--term', action='store_const',
            const=0, default=1, \
            help="output messages to console instead file")
    args = parser.parse_args()
    lvl = logging.INFO
    if args.verbose == 1:
        lvl = logging.DEBUG
    log_fname = 'ssa.log'
    if args.term:
        logging.basicConfig(
                level=lvl,
#                format="%(asctime)s [%(levelname)8s] [%(name)12s] %(message)s",
    format="%(asctime)s [%(levelname)8s] [%(name)12s %(lineno)03d] %(message)s",
                filename=log_fname, filemode='w',
                datefmt='%H:%M:%S')
    else:
        logging.basicConfig(
                level=lvl,
                format="* [%(levelname)8s] [%(name)10s] %(message)s",
                datefmt='%H:%M:%S')
    log = logging.getLogger('main')
    app = QtGui.QApplication(sys.argv)
    dlg = MainDlg(lvl)
    dlg.show()
    sys.exit(app.exec_())
main()

