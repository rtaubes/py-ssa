# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'coeff.ui'
#
# Created: Sat Sep 20 00:35:41 2014
#      by: PyQt4 UI code generator 4.10.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(353, 596)
        self.gridLayout = QtGui.QGridLayout(Dialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.sSelectSb = QtGui.QSlider(Dialog)
        self.sSelectSb.setOrientation(QtCore.Qt.Vertical)
        self.sSelectSb.setInvertedAppearance(True)
        self.sSelectSb.setInvertedControls(False)
        self.sSelectSb.setObjectName(_fromUtf8("sSelectSb"))
        self.horizontalLayout_2.addWidget(self.sSelectSb)
        self.u_vecLw = QtGui.QListWidget(Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.u_vecLw.sizePolicy().hasHeightForWidth())
        self.u_vecLw.setSizePolicy(sizePolicy)
        self.u_vecLw.setMinimumSize(QtCore.QSize(125, 0))
        self.u_vecLw.setMaximumSize(QtCore.QSize(116, 16777215))
        self.u_vecLw.setObjectName(_fromUtf8("u_vecLw"))
        self.horizontalLayout_2.addWidget(self.u_vecLw)
        self.coeffLw = QtGui.QListWidget(Dialog)
        self.coeffLw.setObjectName(_fromUtf8("coeffLw"))
        self.horizontalLayout_2.addWidget(self.coeffLw)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))

