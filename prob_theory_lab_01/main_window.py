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
        main_window.resize(1155, 659)
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setMaximumSize(QtCore.QSize(16777215, 100))
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.num_chars_table = QtWidgets.QTableWidget(self.groupBox_3)
        self.num_chars_table.setMinimumSize(QtCore.QSize(685, 0))
        self.num_chars_table.setObjectName("num_chars_table")
        self.num_chars_table.setColumnCount(8)
        self.num_chars_table.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setItalic(False)
        item.setFont(font)
        self.num_chars_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(7, item)
        self.num_chars_table.horizontalHeader().setDefaultSectionSize(85)
        self.verticalLayout.addWidget(self.num_chars_table)
        self.gridLayout.addWidget(self.groupBox_3, 2, 1, 1, 1)
        self.table = QtWidgets.QTableWidget(self.centralwidget)
        self.table.setMinimumSize(QtCore.QSize(160, 0))
        self.table.setMaximumSize(QtCore.QSize(160, 16777215))
        self.table.setObjectName("table")
        self.table.setColumnCount(1)
        self.table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, item)
        self.table.horizontalHeader().setDefaultSectionSize(120)
        self.gridLayout.addWidget(self.table, 0, 3, 4, 1)
        self.bin_edges_table = QtWidgets.QTableWidget(self.centralwidget)
        self.bin_edges_table.setMinimumSize(QtCore.QSize(100, 0))
        self.bin_edges_table.setMaximumSize(QtCore.QSize(100, 16777215))
        self.bin_edges_table.setObjectName("bin_edges_table")
        self.bin_edges_table.setColumnCount(1)
        self.bin_edges_table.setRowCount(5)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setHorizontalHeaderItem(0, item)
        self.gridLayout.addWidget(self.bin_edges_table, 1, 2, 2, 1)
        self.m_rows_in = QtWidgets.QLineEdit(self.centralwidget)
        self.m_rows_in.setMaximumSize(QtCore.QSize(100, 16777215))
        self.m_rows_in.setObjectName("m_rows_in")
        self.gridLayout.addWidget(self.m_rows_in, 0, 2, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 2, 2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMaximumSize(QtCore.QSize(150, 300))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.calc_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.calc_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        self.calc_btn.setObjectName("calc_btn")
        self.gridLayout_3.addWidget(self.calc_btn, 4, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)
        self.n_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.n_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.n_in.setObjectName("n_in")
        self.gridLayout_3.addWidget(self.n_in, 2, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setMaximumSize(QtCore.QSize(120, 16777215))
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.r_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.r_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.r_in.setObjectName("r_in")
        self.gridLayout_3.addWidget(self.r_in, 1, 1, 1, 1)
        self.q_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.q_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.q_in.setObjectName("q_in")
        self.gridLayout_3.addWidget(self.q_in, 0, 1, 1, 1)
        self.num_of_observ_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.num_of_observ_in.setMaximumSize(QtCore.QSize(120, 16777215))
        self.num_of_observ_in.setObjectName("num_of_observ_in")
        self.gridLayout_3.addWidget(self.num_of_observ_in, 3, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 2, 0, 2, 1)
        self.plot_btn = QtWidgets.QPushButton(self.centralwidget)
        self.plot_btn.setObjectName("plot_btn")
        self.gridLayout.addWidget(self.plot_btn, 3, 2, 1, 1)
        main_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "Лабораторная работа по теории вероятностей"))
        self.groupBox_3.setTitle(_translate("main_window", "Числовые характеристики"))
        item = self.num_chars_table.horizontalHeaderItem(0)
        item.setText(_translate("main_window", "Eη"))
        item = self.num_chars_table.horizontalHeaderItem(1)
        item.setText(_translate("main_window", "🆇"))
        item = self.num_chars_table.horizontalHeaderItem(2)
        item.setText(_translate("main_window", "|Eη - 🆇|"))
        item = self.num_chars_table.horizontalHeaderItem(3)
        item.setText(_translate("main_window", "Dη"))
        item = self.num_chars_table.horizontalHeaderItem(4)
        item.setText(_translate("main_window", "S^2"))
        item = self.num_chars_table.horizontalHeaderItem(5)
        item.setText(_translate("main_window", "|Dη - S^2|"))
        item = self.num_chars_table.horizontalHeaderItem(6)
        item.setText(_translate("main_window", "🅼e"))
        item = self.num_chars_table.horizontalHeaderItem(7)
        item.setText(_translate("main_window", "🆁"))
        item = self.table.horizontalHeaderItem(0)
        item.setText(_translate("main_window", "Время работы"))
        item = self.bin_edges_table.horizontalHeaderItem(0)
        item.setText(_translate("main_window", "Интервал"))
        self.m_rows_in.setText(_translate("main_window", "5"))
        self.groupBox.setTitle(_translate("main_window", "Условие задачи"))
        self.label.setText(_translate("main_window", "<html><head/><body><p><span style=\" font-size:10pt;\">Устройство состоит из </span><span style=\" font-size:10pt; font-style:italic;\">N</span><span style=\" font-size:10pt;\"> &gt;&gt; 1 дублирующих приборов. Каждый следующий прибор включается после выхода из строя предыдущегою Время безотказной работы каждого прибора - положительная с.в. со средним </span><span style=\" font-size:10pt; font-style:italic;\">Q </span><span style=\" font-size:10pt;\">и дисперсией </span><span style=\" font-size:10pt; font-style:italic;\">R. </span><span style=\" font-size:10pt;\">Плотность распределения выбрать по аналогии с приведенными на рисунках на с. 15. С.в. </span><span style=\" font-size:10pt; font-style:italic;\">η - </span><span style=\" font-size:10pt;\">время безотказной работы всего устройтва.</span></p></body></html>"))
        self.groupBox_2.setTitle(_translate("main_window", "Параметры задачи"))
        self.label_3.setText(_translate("main_window", "R"))
        self.calc_btn.setText(_translate("main_window", "PushButton"))
        self.label_2.setText(_translate("main_window", "Q"))
        self.n_in.setText(_translate("main_window", "5"))
        self.label_4.setText(_translate("main_window", "n"))
        self.label_5.setText(_translate("main_window", "N"))
        self.r_in.setText(_translate("main_window", "0.5"))
        self.q_in.setText(_translate("main_window", "1"))
        self.num_of_observ_in.setText(_translate("main_window", "100"))
        self.plot_btn.setText(_translate("main_window", "Графики"))
