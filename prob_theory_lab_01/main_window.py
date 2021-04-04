# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_main_window(object):
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.table = QtWidgets.QTableWidget(self.centralwidget)
        self.table.setMinimumSize(QtCore.QSize(160, 0))
        self.table.setMaximumSize(QtCore.QSize(160, 16777215))
        self.table.setObjectName("table")
        self.table.setColumnCount(1)
        self.table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, item)
        self.table.horizontalHeader().setDefaultSectionSize(120)
        self.gridLayout.addWidget(self.table, 0, 1, 2, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.num_of_observ_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.num_of_observ_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.num_of_observ_in.setObjectName("num_of_observ_in")
        self.gridLayout_3.addWidget(self.num_of_observ_in, 3, 1, 1, 1)
        self.q_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.q_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.q_in.setObjectName("q_in")
        self.gridLayout_3.addWidget(self.q_in, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.n_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.n_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.n_in.setObjectName("n_in")
        self.gridLayout_3.addWidget(self.n_in, 2, 1, 1, 1)
        self.r_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.r_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.r_in.setObjectName("r_in")
        self.gridLayout_3.addWidget(self.r_in, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 3, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem, 0, 2, 1, 1)
        self.calc_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.calc_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        self.calc_btn.setObjectName("calc_btn")
        self.gridLayout_3.addWidget(self.calc_btn, 4, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 1, 0, 1, 1)
        main_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "Лабораторная работа по теории вероятностей"))
        item = self.table.horizontalHeaderItem(0)
        item.setText(_translate("main_window", "Время работы"))
        self.groupBox.setTitle(_translate("main_window", "Условие задачи"))
        self.label.setText(_translate("main_window", "<html><head/><body><p><span style=\" font-size:10pt;\">Устройство состоит из </span><span style=\" font-size:10pt; font-style:italic;\">N</span><span style=\" font-size:10pt;\"> &gt;&gt; 1 дублирующих приборов. Каждый следующий прибор включается после выхода из строя предыдущегою Время безотказной работы каждого прибора - положительная с.в. со средним </span><span style=\" font-size:10pt; font-style:italic;\">Q </span><span style=\" font-size:10pt;\">и дисперсией </span><span style=\" font-size:10pt; font-style:italic;\">R. </span><span style=\" font-size:10pt;\">Плотность распределения выбрать по аналогии с приведенными на рисунках на с. 15. С.в. </span><span style=\" font-size:10pt; font-style:italic;\">η - </span><span style=\" font-size:10pt;\">время безотказной работы всего устройтва.</span></p></body></html>"))
        self.groupBox_2.setTitle(_translate("main_window", "Параметры задачи"))
        self.num_of_observ_in.setText(_translate("main_window", "100"))
        self.q_in.setText(_translate("main_window", "1"))
        self.label_2.setText(_translate("main_window", "Q"))
        self.label_5.setText(_translate("main_window", "N"))
        self.n_in.setText(_translate("main_window", "5"))
        self.r_in.setText(_translate("main_window", "0.5"))
        self.label_3.setText(_translate("main_window", "R"))
        self.label_4.setText(_translate("main_window", "n"))
        self.calc_btn.setText(_translate("main_window", "PushButton"))