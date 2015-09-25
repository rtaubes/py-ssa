#!/usr/bin/env python
''' Dialog for coefficients and 'U' eigenvectors '''

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSignal
from ui_coeff_dlg import Ui_Dialog
import numpy as np
import logging

class CoeffDlg(QtGui.QDialog):

    need_recalc = pyqtSignal()

    def __init__(self, parent):
        super(CoeffDlg, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.u_vectors = None  # np.array
        self.u_checked = None  # list of bool
        self.log = logging.getLogger("CoeffDlg")
        self.setWindowTitle('vector S')
        self.ui.sSelectSb.setMinimum(0)
        self.ui.sSelectSb.sliderReleased.connect(self.s_by_slider)

    def s_by_slider(self):
        for i in range(self.ui.u_vecLw.count()):
            if i <= self.ui.sSelectSb.value():
                self.ui.u_vecLw.item(i).setCheckState(QtCore.Qt.Checked)
            else:
                self.ui.u_vecLw.item(i).setCheckState(0)

    def set_s_vector(self, S, s_used):
        ''' set 'S' vector data
            @param  S - S array like from SVD
            @param  s_used - count of elements used 's'
        '''
        self.ui.u_vecLw.clear()
        if S is None:
            return
        item_no = 0
        for i in range(len(S)):
            item = QtGui.QListWidgetItem("[{0:02d}] {1:f}".format(i, S[i]))
            item.setFlags(QtCore.Qt.ItemIsSelectable |
                    QtCore.Qt.ItemIsEnabled |
                    QtCore.Qt.ItemIsUserCheckable)
            if item_no < s_used:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(0)
            item_no += 1
            self.ui.u_vecLw.addItem(item)
        self.ui.sSelectSb.setMaximum(S.size - 1)
        self.ui.sSelectSb.setValue(s_used)

    def set_poly_roots(self, roots):
        self.ui.coeffLw.clear()
        if roots is None:
            return
        for r in roots:
            s = "{:<22f} ({:f})".format(r, np.linalg.norm(r))
            item = QtGui.QListWidgetItem(s)
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.ui.coeffLw.addItem(item)

    def calc_checked_s_parts(self):
        ''' returns count of checked items '''
        ret = 0
        for i in range(self.ui.u_vecLw.count()):
            if self.ui.u_vecLw.item(i).checkState() == QtCore.Qt.Checked:
                ret += 1
        self.log.info("SVD 'S': {0} checked".format(ret))
        return ret

    def set_coeff(self, coeff):
        self._coeff = coeff

